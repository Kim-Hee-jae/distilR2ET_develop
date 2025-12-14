# -*- coding: cp949 -*-
# metrics/compare_inference_speed.py

import os
import sys
import time
import argparse
import json

import yaml
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from inference_bvh import load_from_bvh
from src.model_shape_aware import RetNet
from src.model_student import DistilledRetNet


WARMUP_ITERS = 10


def load_yaml_config(path):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def build_teacher_model(cfg, device):
    ret_args = cfg.get("ret_model_args", {})
    model = RetNet(**ret_args).to(device)

    weight_path = cfg.get("weights", None)
    if weight_path is None:
        raise ValueError("Teacher config must contain 'weights' field.")

    ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    # DataParallel 저장된 경우 module. prefix 제거
    new_state = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def build_student_model(cfg, device, phase_override="test"):
    """
    phase_override:
      - "test"         -> limited_input=True
      - "test_no_skip" -> limited_input=False
    """
    ret_args = cfg.get("ret_model_args", {})
    limited_input = (phase_override == "test")

    model = DistilledRetNet(
        limited_input=limited_input,
        **ret_args,
    ).to(device)

    weight_path = cfg.get("weights", None)
    if weight_path is None:
        raise ValueError("Student config must contain 'weights' field.")

    ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    new_state = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def prepare_inputs_from_bvh(cfg, device, seq_len):
    load_cfg = cfg.get("load_inp_data", None)
    if load_cfg is None:
        raise ValueError("Config must contain 'load_inp_data' block.")

    (
        inp_seq,
        inpskel,
        tgtskel,
        inp_shape,
        tgt_shape,
        inpquat,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        global_mean,
        global_std,
        tgtanim,
        tgtname,
        tgtftime,
        inpanim,
        inpname,
        inpftime,
        inpjoints,
        tgtjoints,
    ) = load_from_bvh(device, **load_cfg)

    T = inp_seq.shape[1]
    if T < seq_len:
        rep = int(np.ceil(seq_len / T))

        def tile_time(x):
            x_np = x.detach().cpu().numpy()
            x_np = np.tile(x_np, (1, rep, *([1] * (x_np.ndim - 2))))
            x_np = x_np[:, :seq_len]
            return torch.from_numpy(x_np).to(x.device)

        inp_seq = tile_time(inp_seq)
        inpskel = tile_time(inpskel)
        tgtskel = tile_time(tgtskel)
        inpquat = tile_time(inpquat)
    else:
        inp_seq = inp_seq[:, :seq_len]
        inpskel = inpskel[:, :seq_len]
        tgtskel = tgtskel[:, :seq_len]
        inpquat = inpquat[:, :seq_len]

    parents = np.array(
        [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20],
        dtype=np.int64,
    )

    common_inputs = [
        inp_seq,
        None,
        inpskel,
        tgtskel,
        inp_shape,
        tgt_shape,
        inpquat,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        parents,
    ]

    return common_inputs, seq_len


def benchmark_model(
    model,
    inputs,
    k,
    phase,
    device,
    warmup_iters=WARMUP_ITERS,
    bench_iters=50,
):
    full_inputs = tuple(inputs + [k, phase])

    model.eval()
    seq_len = inputs[0].shape[1]

    print(f"  Warmup iterations: {warmup_iters}")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(*full_inputs)

    times_ms = []

    with torch.no_grad():
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            for _ in range(bench_iters):
                starter.record()
                _ = model(*full_inputs)
                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender)
                per_frame = elapsed / seq_len
                times_ms.append(per_frame)
        else:
            for _ in range(bench_iters):
                t0 = time.perf_counter()
                _ = model(*full_inputs)
                t1 = time.perf_counter()
                elapsed = (t1 - t0) * 1000.0
                per_frame = elapsed / seq_len
                times_ms.append(per_frame)

    times_ms = np.array(times_ms, dtype=np.float64)
    avg_ms_per_frame = float(times_ms.mean())
    std_ms_per_frame = float(times_ms.std())
    fps = 1000.0 / avg_ms_per_frame if avg_ms_per_frame > 0 else float("inf")
    return avg_ms_per_frame, std_ms_per_frame, fps


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def select_device_from_config_or_arg(teacher_cfg, device_arg):
    if device_arg is not None:
        dev_str = str(device_arg).lower()
        if "cpu" in dev_str or not torch.cuda.is_available():
            return torch.device("cpu")
        if dev_str.startswith("cuda"):
            return torch.device(device_arg)
        if dev_str.isdigit():
            return torch.device(f"cuda:{dev_str}")
        return torch.device("cuda:0")

    dev_cfg = teacher_cfg.get("device", 0)

    if isinstance(dev_cfg, (list, tuple)):
        dev_cfg = dev_cfg[0]

    if isinstance(dev_cfg, str):
        dev_str = dev_cfg.lower()
        if "cpu" in dev_str or not torch.cuda.is_available():
            return torch.device("cpu")
        else:
            if dev_str.startswith("cuda"):
                return torch.device(dev_cfg)
            elif dev_str.isdigit():
                return torch.device(f"cuda:{dev_str}")
            else:
                return torch.device("cuda:0")
    else:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(dev_cfg)}")
        else:
            return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Compare inference speed of R2ET (teacher) and student models (test / test_no_skip)."
    )
    parser.add_argument(
        "--teacher_config",
        type=str,
        required=True,
        help="YAML config for teacher (original R2ET) inference.",
    )
    parser.add_argument(
        "--student_config",
        type=str,
        required=True,
        help="YAML config for student model inference.",
    )
    parser.add_argument(
        "--bench_iters", type=int, default=50, help="Benchmark iterations."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override device. e.g., "cpu", "cuda", "cuda:0", "1". If not set, use config.',
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=120,
        help="Sequence length (T) used for benchmarking.",
    )
    args = parser.parse_args()

    teacher_cfg = load_yaml_config(args.teacher_config)
    student_cfg = load_yaml_config(args.student_config)

    device = select_device_from_config_or_arg(teacher_cfg, args.device)
    print(f"Using device: {device}")
    print(f"Warmup iterations fixed to {WARMUP_ITERS}")
    print(f"Benchmark iterations: {args.bench_iters}")
    print(f"Sequence length (T): {args.seq_len}")

    common_inputs, seq_len = prepare_inputs_from_bvh(teacher_cfg, device, args.seq_len)
    print(f"Prepared sequence length for benchmarking: T = {seq_len}")

    teacher_model = build_teacher_model(teacher_cfg, device)
    student_model_test = build_student_model(student_cfg, device, phase_override="test")
    student_model_noskip = build_student_model(
        student_cfg, device, phase_override="test_no_skip"
    )

    teacher_params = count_parameters(teacher_model)
    student_params_test = count_parameters(student_model_test)
    student_params_noskip = count_parameters(student_model_noskip)

    teacher_k = float(teacher_cfg.get("k", 0.8))
    teacher_phase = teacher_cfg.get("phase", "test")

    student_k = float(student_cfg.get("k", teacher_k))
    phase_test = "test"
    phase_noskip = "test_no_skip"

    print("\n=== Benchmarking Teacher (R2ET) ===")
    t_mean, t_std, t_fps = benchmark_model(
        teacher_model,
        common_inputs,
        teacher_k,
        teacher_phase,
        device,
        warmup_iters=WARMUP_ITERS,
        bench_iters=args.bench_iters,
    )
    print(
        f"Teacher: {t_mean:.4f} ± {t_std:.4f} ms / frame  ({t_fps:.2f} FPS) "
        f"| Params: {teacher_params:,}"
    )

    print("\n=== Benchmarking Student (phase = test) ===")
    s1_mean, s1_std, s1_fps = benchmark_model(
        student_model_test,
        common_inputs,
        student_k,
        phase_test,
        device,
        warmup_iters=WARMUP_ITERS,
        bench_iters=args.bench_iters,
    )
    print(
        f"Student(test): {s1_mean:.4f} ± {s1_std:.4f} ms / frame  ({s1_fps:.2f} FPS) "
        f"| Params: {student_params_test:,}"
    )

    print("\n=== Benchmarking Student (phase = test_no_skip) ===")
    s2_mean, s2_std, s2_fps = benchmark_model(
        student_model_noskip,
        common_inputs,
        student_k,
        phase_noskip,
        device,
        warmup_iters=WARMUP_ITERS,
        bench_iters=args.bench_iters,
    )
    print(
        f"Student(test_no_skip): {s2_mean:.4f} ± {s2_std:.4f} ms / frame  ({s2_fps:.2f} FPS) "
        f"| Params: {student_params_noskip:,}"
    )

    print(f"\n=== Summary (T = {seq_len}) ===")
    print(
        f"Teacher:           {t_mean:.4f} ± {t_std:.4f} ms / frame | "
        f"{t_fps:.2f} FPS | Params: {teacher_params:,}"
    )
    print(
        f"Student(test):     {s1_mean:.4f} ± {s1_std:.4f} ms / frame | "
        f"{s1_fps:.2f} FPS | Params: {student_params_test:,}"
    )
    print(
        f"Student(no_skip):  {s2_mean:.4f} ± {s2_std:.4f} ms / frame | "
        f"{s2_fps:.2f} FPS | Params: {student_params_noskip:,}"
    )

    device_kind = "gpu" if device.type == "cuda" else "cpu"
    summary_dir = os.path.join(os.path.dirname(__file__), "summary")
    os.makedirs(summary_dir, exist_ok=True)

    json_path = os.path.join(
        summary_dir,
        f"compare_inference_speed_{seq_len}_{device_kind}.json",
    )

    summary = {
        "seq_len": seq_len,
        "device": str(device),
        "device_kind": device_kind,
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": args.bench_iters,
        "models": {
            "teacher": {
                "mean_ms_per_frame": t_mean,
                "std_ms_per_frame": t_std,
                "fps": t_fps,
                "num_params": teacher_params,
            },
            "student_test": {
                "mean_ms_per_frame": s1_mean,
                "std_ms_per_frame": s1_std,
                "fps": s1_fps,
                "num_params": student_params_test,
            },
            "student_test_no_skip": {
                "mean_ms_per_frame": s2_mean,
                "std_ms_per_frame": s2_std,
                "fps": s2_fps,
                "num_params": student_params_noskip,
            },
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved JSON summary to: {json_path}")


if __name__ == "__main__":
    main()
