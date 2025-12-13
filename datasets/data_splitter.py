 # -*- coding: cp949 -*-
import json
import random
from pathlib import Path


def make_random_split(
    dataset_dir: str,
    unseen_pct: float,
    out_path: str,
    seed: int = 42,
    exts=None,
):
    """
    dataset_dir 안의 파일들 중 unseen_pct (%)를 랜덤으로 뽑아서
    unseen / train 리스트를 out_path(json)에 저장한다.

    Args:
        dataset_dir: 모션 데이터가 들어 있는 루트 폴더
        unseen_pct: 전체에서 unseen으로 뽑을 비율 (예: 20.0)
        out_path: 결과를 저장할 json 경로
        seed: 랜덤 시드 (재현용)
        exts: 사용할 파일 확장자 리스트 (예: ['.bvh', '.npy'])
              None이면 모든 파일 사용
    """
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.is_dir(), f"{dataset_dir} is not a directory"

    # 파일 모으기
    if exts is None:
        files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    else:
        exts = {e.lower() for e in exts}
        files = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in exts]

    files = sorted(files)
    n_total = len(files)
    if n_total == 0:
        raise ValueError(f"No files found in {dataset_dir}")

    # 랜덤 셔플 후 n%를 unseen으로
    random.seed(seed)
    random.shuffle(files)

    n_unseen = int(n_total * (unseen_pct / 100.0))
    unseen_files = files[:n_unseen]
    train_files = files[n_unseen:]

    result = {
        "dataset_dir": str(dataset_dir),
        "total_files": n_total,
        "unseen_ratio": unseen_pct,
        "n_seen": len(train_files),
        "n_unseen": len(unseen_files),
        "seed": seed,
        "seen_motion": [str(p) for p in train_files],
        "unseen_motion": [str(p) for p in unseen_files],
        "seen_char": ["Abe", "Adam", "Alien Soldier", "Crypto", "Demon T Wiezzorek", "Exo Gray", "James", "Leonard"],
        "unseen_char": ["Steve"],
        "target_char": "input",
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved split to {out_path}")
    print(f"  total  : {n_total}")
    print(f"  seen  : {len(train_files)}")
    print(f"  unseen : {len(unseen_files)}")
    

if __name__ == "__main__":
    # python split_motion.py
    make_random_split(
        dataset_dir="./datasets/mixamo/char/Abe",
        unseen_pct=20.0,  # 전체의 20%를 unseen으로
        out_path="./config/data_config.json",
        seed=42,
        exts=[".bvh"],  # 필요 없으면 None
    )
