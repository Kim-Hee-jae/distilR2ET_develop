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
from collections import OrderedDict


import argparse

from datasets.student_feeder import Feeder
from src.student_model_v2 import DistilledRetNet


def get_parser():
    parser = argparse.ArgumentParser(description="Distilled R2ET Trainer with Validation & Early Stopping")

    parser.add_argument('--config', default='./config/student_config.yaml', help='path to the configuration file')
    
    parser.add_argument("--data_path", type=str, default='./datasets/student/', help="Root folder containing processed data")
    parser.add_argument("--stats_path", type=str, default="./stats_distill", help="Folder to store/load stats.npz")
    
    parser.add_argument("--max_length", type=int, default=120, help="Max sequence length")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lambda_dqs", type=float, default=1.0, help="Loss weight for delta_qs")
    parser.add_argument("--lambda_dqg", type=float, default=1.0, help="Loss weight for delta_qg")

    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum delta for improvement")
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_loss_dqs", "val_loss_dqg"],
        help="Validation metric to monitor"
    )

    parser.add_argument("--work_dir", type=str, default="./work_dir/distill", help="Folder to save checkpoints/logs")
    parser.add_argument("--save_interval", type=int, default=10, help="Checkpoint interval")
    parser.add_argument("--resume_weights", type=str, default="", help="Checkpoint to resume")

    parser.add_argument("--teacher_weights", type=str, default="./pretrain/shape_aware.pt", help="Path to pre-trained R2ET ret model weights (.pt)")
    parser.add_argument("--freeze_encoder", action="store_true", help="Enable encoder freezing")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(state, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save(state, path)
    print(f"=> Saved checkpoint to {path}")


def load_model(model, optimizer, resume_path, teacher_path, freeze_encoder, device):
    start_epoch, best_val = None, None
    if resume_path:
        print(f"=> Loading checkpoint from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", None)
        print(f"=> Resumed from epoch {start_epoch}, best_val={best_val}")
    elif teacher_path:
        print(f"=> Loading checkpoint from {teacher_path}")
        teacher_state = torch.load(teacher_path, map_location=device)
        init_state = model.state_dict()
        for key in list(teacher_state.keys()):
            value = teacher_state.pop(key)
            if 'delta_dec' in key and ('q_encoder' in key or 'skel_encoder' in key):
                key = key.replace('module.', '').replace('delta_dec', 'delta_qs_dec')
                teacher_state[key] = value
        init_state.update(teacher_state)
        model.load_state_dict(init_state)
        print(f"=> Initialized encoders from teacher")
    else:
        print("=> Raw Start")

    if freeze_encoder:
        for key, param in model.named_parameters():
            if 'delta_qs_dec' in key and ('q_encoder' in key or 'skel_encoder' in key):
                param.requires_grad = False
 

    return start_epoch, best_val


def masked_l1_loss(pred, target, mask, eps=1e-6):
    m = mask.unsqueeze(-1).unsqueeze(-1)
    diff = torch.sqaure(pred - target) * m
    return diff.sum() / (m.sum() * pred.size(-1) + eps)


def train_one_epoch(epoch, model, loader, optimizer, device, args):
    model.train()
    t_loss = t_dqs = t_dqg = 0.0
    start_time = time.time()

    for it, batch in enumerate(loader):
        quat_src, skel_src, shape_src, dqs_teacher, dqg_teacher, quat_tgt, mask = batch
        quat_src = quat_src.to(device).float()
        skel_src = skel_src.to(device).float()
        shape_src = shape_src.to(device).float()
        dqs_teacher = dqs_teacher.to(device).float()
        dqg_teacher = dqg_teacher.to(device).float()
        quat_tgt = quat_tgt.to(device).float()
        mask = mask.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        pred_quat = model(quat_src, skel_src, shape_src)

        loss = masked_l1_loss(pred_quat, quat_tgt, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t_loss += loss.item()

        if (it + 1) % 10 == 0:
            print(
                f"[Train] Epoch {epoch} Iter {it+1}/{len(loader)} "
                f"Loss {t_loss/(it+1):.6f}"
            )

    return t_loss / len(loader)


@torch.no_grad()
def eval_one_epoch(epoch, model, loader, device, args, mode='Val'):
    model.eval()
    v_loss = v_dqs = v_dqg = 0.0

    for batch in loader:
        quat_src, skel_src, shape_src, dqs_teacher, dqg_teacher, quat_tgt, mask = batch

        quat_src = quat_src.to(device).float()
        skel_src = skel_src.to(device).float()
        shape_src = shape_src.to(device).float()
        dqs_teacher = dqs_teacher.to(device).float()
        dqg_teacher = dqg_teacher.to(device).float()
        quat_tgt = quat_tgt.to(device).float()
        mask = mask.to(device).float()

        pred_quat = model(quat_src, skel_src, shape_src)

        loss = masked_l1_loss(pred_quat, quat_tgt, mask)

        v_loss += loss.item()

    print(f"[{mode}] Epoch {epoch} Loss {v_loss/len(loader):.6f}")
    return v_loss / len(loader)


def main(args):
    init_seed(args.seed)
    device = args.device
    print("Using device:", device)

    os.makedirs(args.work_dir, exist_ok=True)

    full_dataset = Feeder(
        data_path=args.data_path,
        stats_path=args.stats_path,
        max_length=args.max_length,
        num_joint=22,
    )

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)
    test_ds  = Feeder(
        data_path=args.data_path,
        stats_path=args.stats_path,
        max_length=args.max_length,
        num_joint=22,
        mode='test'
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, drop_last=False)

    model = DistilledRetNet().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    start_epoch = 0
    best_val = float("inf")
    prev_epoch, prev_best = load_model(model, optimizer, args.resume_weights, args.teacher_weights, args.freeze_encoder, device)
    if prev_epoch is not None:
        start_epoch = prev_epoch
    if prev_best is not None:
        best_val = prev_best


    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
    }

    patience_counter = 0

    for epoch in range(start_epoch, args.num_epoch):
        train_loss= train_one_epoch(epoch, model, train_loader, optimizer, device, args)
        val_loss = eval_one_epoch(epoch, model, val_loader, device, args, mode='Val')
        test_loss = eval_one_epoch(epoch, model, test_loader, device, args, mode='Test')

        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["test_loss"].append(float(test_loss))

        with open(os.path.join(args.work_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.monitor == "val_loss":
            monitor_value = val_loss
        elif args.monitor == "test_loss":
            monitor_value = test_loss
        else:
            monitor_value = train_loss

        improved = monitor_value + args.min_delta < best_val
        if improved:
            best_val = monitor_value
            patience_counter = 0
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            }
            save_model(state, args.work_dir, "best")
        else:
            if args.early_stopping:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break

        if (epoch + 1) % args.save_interval == 0:
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            }
            save_model(state, args.work_dir, f"epoch_{epoch+1:03d}")


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