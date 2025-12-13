import os
import time
import random
import yaml
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import argparse

from datasets.student_feeder_backup import Feeder
from src.student_model___mlp import DistilRET


def get_parser():
    parser = argparse.ArgumentParser(description="Distilled R2ET Trainer with Validation & Early Stopping")

    parser.add_argument('--config', default='./config/train_cfg.yaml', help='path to the configuration file')
    parser.add_argument("--data_path", type=str, default='./datasets/student/', help="Root folder containing processed data")
    parser.add_argument("--stats_path", type=str, default="./stats_distill", help="Folder to store/load stats.npz")
    parser.add_argument("--max_length", type=int, default=120, help="Max sequence length")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lambda_dqs", type=float, default=1.0, help="Loss weight for delta_qs")
    parser.add_argument("--lambda_dqg", type=float, default=1.0, help="Loss weight for delta_qg")

    parser.add_argument("--work_dir", type=str, default="./work_dir/distill", help="Folder to save checkpoints/logs")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(model, optimizer, resume_path, device):
    print(f"=> Loading checkpoint from {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val", None)
    print(f"=> Resumed from epoch {start_epoch}, best_val={best_val}")
    return start_epoch, best_val


def masked_l1_loss(pred, target, mask, eps=1e-6):
    m = mask.unsqueeze(-1).unsqueeze(-1)
    diff = torch.abs(pred - target) * m
    return diff.sum() / (m.sum() * pred.size(-1) + eps)

@torch.no_grad()
def eval_one_epoch(epoch, model, loader, device, args):
    model.eval()
    v_loss = v_dqs = v_dqg = 0.0

    for batch in loader:
        quat_src, skel_src, shape_src, dqs_teacher, dqg_teacher, mask = batch

        quat_src = quat_src.to(device).float()
        skel_src = skel_src.to(device).float()
        shape_src = shape_src.to(device).float()
        dqs_teacher = dqs_teacher.to(device).float()
        dqg_teacher = dqg_teacher.to(device).float()
        mask = mask.to(device).float()

        pred_dqs, pred_dqg = model(quat_src, skel_src, shape_src)

        loss_dqs = masked_l1_loss(pred_dqs, dqs_teacher, mask)
        loss_dqg = masked_l1_loss(pred_dqg, dqg_teacher, mask)
        loss = args.lambda_dqs * loss_dqs + args.lambda_dqg * loss_dqg

        v_loss += loss.item()
        v_dqs += loss_dqs.item()
        v_dqg += loss_dqg.item()

    print(f"[Val] Epoch {epoch} Loss {v_loss/len(loader):.6f}")
    return v_loss / len(loader), v_dqs / len(loader), v_dqg / len(loader)


def main(args):
    init_seed(args.seed)
    device = args.device
    print("Using device:", device)

    os.makedirs(args.work_dir, exist_ok=True)

    # ----- Dataset -----
    test_dataset = Feeder(
        data_path=args.data_path,
        stats_path=args.stats_path,
        max_length=args.max_length,
        num_joint=22,
        mode='test'
    )

    n_test = len(test_dataset)
    print(f"Test samples: {n_test}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    
    # ----- Model -----
    retarget_net = DistilRET()    
    retarget_net = retarget_net.to(device)

    _, prev_best = load_model(retarget_net, None, args.resume, device)


    # ----- history -----
    history = {
        "epoch": [],
        "test_loss": [],
        "test_loss_dqs": [],
        "test_loss_dqg": [],
    }

    # ----------------- Test Loop -----------------
    test_loss, test_dqs, test_dqg = eval_one_epoch(
        None, retarget_net, test_loader, device, args
    )

    # history
    history["epoch"].append(None)
    history["test_loss"].append(float(test_loss))
    history["test_loss_dqs"].append(float(test_dqs))
    history["test_loss_dqg"].append(float(test_dqg))

    # save json
    hist_path = os.path.join(args.work_dir, "history_eval_100.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert k in key
        parser.set_defaults(**default_arg)
    args = parser.parse_args()
    main(args)