# -*- coding:cp949 -*-

import os, sys
import json
from pathlib import Path
import numpy as np
sys.path.append("./outside-code")
import BVH as BVH
import Animation

# 기본 설정

BASE = r"c:\Users\Taeyong_Sim\Desktop\Motion Retargeting\R2ET\results"  # inference 결과 폴더
data_config_path = r"c:\Users\Taeyong_Sim\Desktop\Motion Retargeting\R2ET\config\data_config.json"  # data split 파일 경로

# skeleton에서 발 joint index
FOOT_JOINTS = {
    "left":  [8, 9],     # 왼발 관련 joint index
    "right": [12, 13],   # 오른발 관련 joint index
}

# BVH 로더

def load_bvh_positions(bvh_path):
    try:
        anim = BVH.load(bvh_path)[0]
        joints = Animation.positions_global(anim)  # (T, J, 3)
        return joints
    except Exception as e:
        print(f"[WARN] BVH load failed: {bvh_path} -> {e}")
        return None

# 지표 계산 함수
def compute_mse(pos_a, pos_b):
    """
    pos_a, pos_b: (T, J, 3)
    길이가 다르면 공통 구간까지만 사용
    """
    T = min(pos_a.shape[0], pos_b.shape[0])
    a = pos_a[:T]
    b = pos_b[:T]
    mse = np.mean((a - b) ** 2)
    return mse

def get_height(joints):
    """
    joints: (J, 3)
    캐릭터 키 계산
    """
    segs = [
        (5, 4),
        (4, 3),
        (3, 2),
        (2, 1),
        (1, 0),
        (0, 6),
        (6, 7),
        (7, 8),
        (8, 9),
    ]
    
    lengths = []
    for i, j in segs:
        diff = joints[i] - joints[j]
        l = np.sqrt((diff ** 2).sum(axis=-1))  
        lengths.append(l)
    
    height = np.sum(np.stack(lengths, axis=0), axis=0)  
    return height

def compute_foot_contact_mask(positions, fid_l):
    """
    positions: (T, J, 3)
    fid_l: 발에 해당하는 joint index 리스트
    반환: (T-1, len(fid_l)) 의 0/1 float 마스크
    """
    char_height = get_height(positions[0])
    velfactor, heightfactor = np.array([0.10, 0.10]), np.array([9.0, 3.0])

    feet_l_x = ((positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) / char_height * 180) ** 2
    feet_l_y = ((positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) / char_height * 180) ** 2
    feet_l_z = ((positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) / char_height * 180) ** 2
    feet_l_h = positions[:-1, fid_l, 1] / char_height * 180
    feet_l = (
        ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
    ).astype(float)

    return feet_l  # 0.0 / 1.0

def compute_foot_height_metric(pred_pos, gt_pos, foot_indices):
    """
    GT가 foot contact인 프레임에서,
    pred_pos의 해당 발 joints의 y좌표 절댓값 평균 계산

    pred_pos, gt_pos: (T, J, 3)
    foot_indices: [j0, j1, ...]
    반환: scalar (평균 |y|)
    """
    # GT 기준 contact 마스크 (T-1, F)
    gt_contact_mask = compute_foot_contact_mask(gt_pos, foot_indices)  # 0/1 float
    gt_contact = gt_contact_mask > 0.5  # bool

    # pred의 y좌표 (contact 마스크의 시간축(T-1)에 맞춰서)
    pred_y = pred_pos[:-1, foot_indices, 1]  # (T-1, F)

    # GT가 contact인 위치만 골라서 abs(y) 측정
    if not np.any(gt_contact):
        # contact 프레임이 없다면 0으로 처리
        return 0.0

    vals = np.abs(pred_y[gt_contact])  # flatten
    return float(np.mean(vals))


def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    m = float(np.nanmean(arr))
    if arr.size > 1:
        s = float(np.nanstd(arr, ddof=1))
    else:
        s = 0.0
    return m, s


# split JSON 처리
def load_split_config(path: str):
    split_path = Path(path)
    assert split_path.is_file(), f"{split_path} not found"

    with split_path.open("r", encoding="utf-8") as f:
        data_config = json.load(f)
    return data_config

def classify_sequence(char_name: str, motion_name: str, data_config: dict) -> str:
    """
    char_name, motion_name을 받아서 아래 4가지 카테고리 중 하나를 리턴:
      - 'seen_char_seen_motion'
      - 'seen_char_unseen_motion'
      - 'unseen_char_seen_motion'
      - 'unseen_char_unseen_motion'
    """
    seen_chars   = set(data_config.get("seen_char", []))
    unseen_chars = set(data_config.get("unseen_char", []))
    # motion: absolute path -> 파일명 추출
    seen_motions   = set([_.split("\\")[-1][:-4] for _ in data_config.get("seen_motion", [])])
    unseen_motions = set([_.split("\\")[-1][:-4] for _ in data_config.get("unseen_motion", [])])

    is_char_seen = char_name in seen_chars
    if not is_char_seen and char_name in unseen_chars:
        is_char_seen = False

    is_motion_seen = motion_name in seen_motions
    if not is_motion_seen and motion_name in unseen_motions:
        is_motion_seen = False

    if is_char_seen and is_motion_seen:
        return "seen_char_seen_motion"
    elif is_char_seen and not is_motion_seen:
        return "seen_char_unseen_motion"
    elif (not is_char_seen) and is_motion_seen:
        return "unseen_char_seen_motion"
    else:
        return "unseen_char_unseen_motion"


# 메인 루프
def main():
    data_config = load_split_config(data_config_path)

    # 카테고리별 결과 저장
    buckets = {
        "seen_char_seen_motion": [],
        "seen_char_unseen_motion": [],
        "unseen_char_seen_motion": [],
        "unseen_char_unseen_motion": [],
    }

    base_path = Path(BASE)

    for char_dir in base_path.iterdir():
        if not char_dir.is_dir():
            continue

        char_name = char_dir.name

        # 모션 이름별로 세 가지 파일(_gt, _dr2et, _r2et)을 정리
        motion_dict: dict[str, dict[str, Path]] = {}

        for f in char_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() != ".bvh":
                continue

            stem = f.stem  # 예: walk01_gt
            if stem.endswith("_gt"):
                motion_name = stem[:-3]       # "_gt" 길이 3
                var = "gt"
            elif stem.endswith("_dr2et"):
                motion_name = stem[:-6]       # "_dr2et" 길이 6
                var = "dr2et"                 # student
            elif stem.endswith("_r2et"):
                motion_name = stem[:-5]       # "_r2et" 길이 5
                var = "r2et"                  # teacher
            else:
                continue

            if motion_name not in motion_dict:
                motion_dict[motion_name] = {}
            motion_dict[motion_name][var] = f

        # 각 motion_name에 대해 세 파일이 다 있는 경우만 처리
        for motion_name, files in motion_dict.items():
            if not all(k in files for k in ["gt", "dr2et", "r2et"]):
                print(f"[WARN] {char_name}/{motion_name}: gt/dr2et/r2et 중 일부 없음 -> 스킵")
                continue

            gt_path     = files["gt"]
            stud_path   = files["dr2et"]
            teach_path  = files["r2et"]

            gt_pos    = load_bvh_positions(str(gt_path))     # (T, J, 3)
            stud_pos  = load_bvh_positions(str(stud_path))
            teach_pos = load_bvh_positions(str(teach_path))

            # 1) joint MSE (GT 기준)
            stud_mse  = compute_mse(stud_pos,  gt_pos)
            teach_mse = compute_mse(teach_pos, gt_pos)

            # 1-1) student - teacher 간 MSE 추가
            stud_teach_mse = compute_mse(stud_pos, teach_pos)

            # 2) foot contact metric
            #    GT가 foot contact인 프레임에서의 |y| 평균 (왼발/오른발 각각)
            left_idx  = FOOT_JOINTS["left"]
            right_idx = FOOT_JOINTS["right"]

            stud_left_y   = compute_foot_height_metric(stud_pos,  gt_pos, left_idx)
            stud_right_y  = compute_foot_height_metric(stud_pos,  gt_pos, right_idx)
            teach_left_y  = compute_foot_height_metric(teach_pos, gt_pos, left_idx)
            teach_right_y = compute_foot_height_metric(teach_pos, gt_pos, right_idx)

            # seen/unseen 캐릭터/모션 분류
            bucket_name = classify_sequence(char_name, motion_name, data_config)

            seq_id = f"{char_name}/{motion_name}"
            result = {
                "seq": seq_id,
                "char": char_name,
                "motion": motion_name,
                "student": {
                    "mse_to_gt": stud_mse,
                    "left_contact_abs_y":  stud_left_y,
                    "right_contact_abs_y": stud_right_y,
                },
                "teacher": {
                    "mse_to_gt": teach_mse,
                    "left_contact_abs_y":  teach_left_y,
                    "right_contact_abs_y": teach_right_y,
                },
                "student_teacher_mse": stud_teach_mse,
            }
            buckets[bucket_name].append(result)

            # 개별 시퀀스 출력
            print(f"=== {seq_id} ({bucket_name}) ===")
            print(f"  Student MSE to GT:  {stud_mse:.6f}")
            print(f"  Teacher MSE to GT:  {teach_mse:.6f}")
            print(f"  Student-Teacher MSE: {stud_teach_mse:.6f}")
            print(f"  Student Left (foot contact |y|): {(stud_left_y + stud_right_y)/2:.4f}")
            print(f"  Teacher Right (foot contact |y|): {(teach_left_y + teach_right_y)/2:.4f}")
            # print(f"    Student Left  |y| (GT contact): {stud_left_y:.4f}")
            # print(f"    Student Right |y| (GT contact): {stud_right_y:.4f}")
            # print(f"    Teacher Left  |y| (GT contact): {teach_left_y:.4f}")
            # print(f"    Teacher Right |y| (GT contact): {teach_right_y:.4f}")
            print()

    # 카테고리별 평균 요약

    summary_out = {}
    print("\n========== SUMMARY by SPLIT ==========")
    for bucket_name, seq_list in buckets.items():
        if not seq_list:
            print(f"{bucket_name}: (no sequences)")
            continue

        stud_mses_to_gt   = [r["student"]["mse_to_gt"] for r in seq_list]
        teach_mses_to_gt  = [r["teacher"]["mse_to_gt"] for r in seq_list]
        stud_teach_mses   = [r["student_teacher_mse"] for r in seq_list]

        stud_left_abs_y   = [r["student"]["left_contact_abs_y"]  for r in seq_list]
        stud_right_abs_y  = [r["student"]["right_contact_abs_y"] for r in seq_list]
        teach_left_abs_y  = [r["teacher"]["left_contact_abs_y"]  for r in seq_list]
        teach_right_abs_y = [r["teacher"]["right_contact_abs_y"] for r in seq_list]

        # 시퀀스 단위 mean / std
        stud_mse_mean,  stud_mse_std  = mean_std(stud_mses_to_gt)
        teach_mse_mean, teach_mse_std = mean_std(teach_mses_to_gt)
        st_mse_mean,    st_mse_std    = mean_std(stud_teach_mses)

        stud_contact_all = stud_left_abs_y + stud_right_abs_y
        teach_contact_all = teach_left_abs_y + teach_right_abs_y
        stud_cont_mean,  stud_cont_std  = mean_std(stud_contact_all)
        teach_cont_mean, teach_cont_std = mean_std(teach_contact_all)

        # 출력
        print(f"\n[{bucket_name}]  (#seq={len(seq_list)})")
        print(f"  Student  MSE to GT mean/std:   {stud_mse_mean:.6f} / {stud_mse_std:.6f}")
        print(f"  Teacher  MSE to GT mean/std:   {teach_mse_mean:.6f} / {teach_mse_std:.6f}")
        print(f"  Student-Teacher MSE mean/std:  {st_mse_mean:.6f} / {st_mse_std:.6f}")
        print(f"  Student  mean/std (foot contact |y|): {stud_cont_mean:.4f} / {stud_cont_std:.4f}")
        print(f"  Teacher  mean/std (foot contact |y|): {teach_cont_mean:.4f} / {teach_cont_std:.4f}")
        # print(f"  Student  Left  |y| mean (GT contact): {np.mean(stud_left_abs_y):.4f}")
        # print(f"  Student  Right |y| mean (GT contact): {np.mean(stud_right_abs_y):.4f}")
        # print(f"  Teacher  Left  |y| mean (GT contact): {np.mean(teach_left_abs_y):.4f}")
        # print(f"  Teacher  Right |y| mean (GT contact): {np.mean(teach_right_abs_y):.4f}")

        # JSON 저장용 구조
        summary_out[bucket_name] = {
            "num_sequences": len(seq_list),

            "student_mse_to_gt": {
                "mean": stud_mse_mean,
                "std":  stud_mse_std,
            },
            "teacher_mse_to_gt": {
                "mean": teach_mse_mean,
                "std":  teach_mse_std,
            },
            "student_teacher_mse": {
                "mean": st_mse_mean,
                "std":  st_mse_std,
            },

            "student_foot_contact": {
                "mean": stud_cont_mean,
                "std":  stud_cont_std,
            },
            "teacher_foot_contact": {
                "mean": teach_cont_mean,
                "std":  teach_cont_std,
            },
        }

    # 결과 저장
    out_path = Path("./metrics/summary/eval_summary.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to {out_path}")

if __name__ == "__main__":
    main()
