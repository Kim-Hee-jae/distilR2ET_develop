#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bvh_self_contact_final.py

Robust BVH parser + safe FK + self-contact evaluation (L_rep, L_att)
- ./data 안의 모든 .bvh 파일에 대해 평가
"""

import os
import re
import glob
import json
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import math

# ---------------------------
# Configuration: joint whitelist
# ---------------------------
JOINTS_LIST = [
    "Hip", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]

# ---------------------------
# Utilities
# ---------------------------
def normalize_name(name: str) -> str:
    s = name.replace("mixamorig:", "").replace("mixamo_", "")
    s = s.replace(":", "").strip()
    return s

def array_to_homogeneous(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

###############################################
#  🔥 내부용 BVH 파서
###############################################
class BVH_Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.offset = np.zeros(3, dtype=np.float32)
        self.channels: List[str] = []
        self.children: List["BVH_Node"] = []
        self.parent = parent
        self.index: Optional[int] = None
        self.channel_indices: List[int] = []

class BVH:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.nodes: List[BVH_Node] = []
        self.root: Optional[BVH_Node] = None
        self.frames: Optional[np.ndarray] = None
        self.frame_time: float = 1/60
        self.num_frames: int = 0

        self._parse_hierarchy()
        self._map_channels()
        self._parse_motion()

    ##############################################################
    # 1) HIERARCHY Parsing
    ##############################################################
    def _parse_hierarchy(self):
        with open(self.filepath, 'r', encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        stack: List[BVH_Node] = []
        node: Optional[BVH_Node] = None
        i = 0

        while i < len(lines):
            l = lines[i].strip()

            if l.startswith("ROOT") or l.startswith("JOINT"):
                name = l.split()[1]
                parent = stack[-1] if stack else None
                node = BVH_Node(name, parent)
                node.index = len(self.nodes)
                self.nodes.append(node)

                if parent:
                    parent.children.append(node)
                else:
                    self.root = node

                stack.append(node)
                i += 1
                continue

            if l.startswith("OFFSET") and node is not None:
                nums = list(map(float, l.split()[1:4]))
                node.offset = np.array(nums, dtype=np.float32)

            if "CHANNELS" in l and node is not None:
                p = l.split()
                n = int(p[1])
                node.channels = p[2:2+n]

            # End Site 처리: 채널/자식 없음 → 블록 스킵
            if l.upper().startswith("END SITE") or l.startswith("End") or l.startswith("END"):
                i += 1
                while i < len(lines) and "}" not in lines[i]:
                    i += 1
                i += 1
                continue

            # 블록 닫힘 처리
            if l.startswith("}"):
                if stack:
                    stack.pop()
                i += 1
                continue

            if l.strip().upper().startswith("MOTION"):
                break

            i += 1

    ##############################################################
    # 2) Channel indexing
    ##############################################################
    def _map_channels(self):
        idx = 0
        for node in self.nodes:
            node.channel_indices = list(range(idx, idx + len(node.channels)))
            idx += len(node.channels)

    ##############################################################
    # 3) MOTION Parsing
    ##############################################################
    def _parse_motion(self):
        with open(self.filepath, 'r', encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        m = re.search(r"Frames:\s+(\d+)", txt)
        ft = re.search(r"Frame Time:\s+([\d\.]+)", txt)
        if not m or not ft:
            raise RuntimeError("BVH MOTION header parse failed")

        self.num_frames = int(m.group(1))
        self.frame_time = float(ft.group(1))

        motion_raw = txt.split("Frame Time:")[1].strip()
        rows = motion_raw.split("\n")[1:]  # 첫 줄은 숫자 바로 뒤 줄부터가 프레임 데이터

        frame_data = []
        for l in rows:
            cols = l.strip().split()
            if len(cols) > 0:
                frame_data.append(list(map(float, cols)))

        self.frames = np.array(frame_data, dtype=np.float32)
        if self.frames.shape[0] < self.num_frames:
            # 방어: 실제 프레임 수보다 데이터가 적으면 맞춰줌
            self.num_frames = self.frames.shape[0]

    ##############################################################
    # 4) SAFE Forward Kinematics - pure Python
    ##############################################################
    def forward_kinematics(self, frame_idx: int, debug: bool = False) -> Dict[int, np.ndarray]:
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError("frame_idx out of range")

        frame = self.frames[frame_idx].tolist()
        joint_positions: Dict[int, np.ndarray] = {}

        def mat3_mul_vec3(M, v):
            return [
                M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
                M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
                M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
            ]

        def mat3_mul_mat3(A, B):
            C = [[0.0, 0.0, 0.0] for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for k in range(3):
                        s += A[i][k] * B[k][j]
                    C[i][j] = s
            return C

        def euler_to_R(angles_deg, order="ZYX"):
            try:
                ax, ay, az = float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2])
            except Exception:
                return [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]

            rx = math.radians(ax)
            ry = math.radians(ay)
            rz = math.radians(az)

            cx, sx = math.cos(rx), math.sin(rx)
            cy, sy = math.cos(ry), math.sin(ry)
            cz, sz = math.cos(rz), math.sin(rz)

            Rx = [
                [1.0, 0.0, 0.0],
                [0.0, cx, -sx],
                [0.0, sx,  cx],
            ]
            Ry = [
                [ cy, 0.0, sy],
                [ 0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ]
            Rz = [
                [cz, -sz, 0.0],
                [sz,  cz, 0.0],
                [0.0, 0.0, 1.0],
            ]

            if order == "ZYX":
                return mat3_mul_mat3(mat3_mul_mat3(Rx, Ry), Rz)
            else:
                return mat3_mul_mat3(mat3_mul_mat3(Rz, Ry), Rx)

        stack: List[Tuple[BVH_Node, List[float], List[List[float]]]] = [
            (self.root, [0.0, 0.0, 0.0],
             [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        ]
        visited = set()
        steps = 0
        MAX_STEPS = len(self.nodes) * 10 + 10

        if debug:
            print(f"[FK] Frame {frame_idx} start (nodes={len(self.nodes)})")

        while stack:
            node, parent_pos, parent_rot = stack.pop()
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)

            steps += 1
            if steps > MAX_STEPS:
                if debug:
                    print(f"[FK-WARN] too many steps ({steps}), break")
                break

            off = node.offset.tolist() if isinstance(node.offset, np.ndarray) else list(node.offset)
            local_pos = [float(off[0]), float(off[1]), float(off[2])]
            local_rot = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]

            if node.channel_indices:
                try:
                    vals = [frame[i] for i in node.channel_indices]
                    # Mixamo root: 6채널(XYZpos + ZYXrot), 나머지: 3채널(rot) 인 경우가 많음
                    if len(vals) >= 6 and any("position" in c.lower() for c in node.channels):
                        # root
                        local_pos = [vals[0], vals[1], vals[2]]
                        local_rot = euler_to_R(vals[3:6], order="ZYX")
                    elif len(vals) >= 3:
                        local_rot = euler_to_R(vals[:3], order="ZYX")
                except Exception as e:
                    if debug:
                        print(f"[FK-ERR] channel read failed at '{node.name}': {e}")

            try:
                rotated = mat3_mul_vec3(parent_rot, local_pos)
                gpos = [
                    parent_pos[0] + rotated[0],
                    parent_pos[1] + rotated[1],
                    parent_pos[2] + rotated[2],
                ]
                grot = mat3_mul_mat3(parent_rot, local_rot)
            except Exception as e:
                if debug:
                    print(f"[FK-ERR] transform failed at '{node.name}': {e}")
                gpos = parent_pos[:]
                grot = [row[:] for row in parent_rot]

            joint_positions[node.index] = np.array(gpos, dtype=np.float32)

            for c in node.children:
                stack.append((c, gpos, grot))

        if debug:
            print(f"[FK] Frame {frame_idx} done: joints={len(joint_positions)}, steps={steps}")

        return joint_positions

# ---------------------------
# Self-Contact Evaluator (R2ET 스타일 근사)
# ---------------------------
class SelfContactEvaluator:
    def __init__(
        self,
        bvh: BVH,
        joints_whitelist: List[str],
        joint_radius: float = 5.0,
        contact_band: float = 10.0,
        num_nearest: int = 4
    ):
        """
        joint_radius, contact_band는 BVH 스케일에 맞게 조절 필요.
        (현재 FK 예제 기준, 캐릭 높이 ~40 정도 → radius 3~6, band 8~15 정도 적당)
        """
        self.bvh = bvh
        self.joint_radius = float(joint_radius)
        self.contact_band = float(contact_band)
        self.num_nearest = int(num_nearest)

        norm_targets = {normalize_name(n) for n in joints_whitelist}
        self.active_indices: List[int] = []

        for node in bvh.nodes:
            if normalize_name(node.name) in norm_targets:
                self.active_indices.append(node.index)

        if not self.active_indices:
            # whitelist에 아무것도 안 걸리면 전체 사용
            self.active_indices = [n.index for n in bvh.nodes]
            print(f"  [Eval] whitelist 매칭 실패 → 전체 {len(self.active_indices)}개 joint 사용")
        else:
            print(f"  [Eval] whitelist 기반 active joints: {len(self.active_indices)}개")

    def evaluate_frame(self, t: int, debug: bool = False) -> Tuple[float, float, int, int]:
        """
        한 프레임에 대해:
        - L_rep: penetration(관통) 정도 평균
        - L_att: contact band 내 근접 정도 평균
        - rep_count, att_count: 각 항목 개수
        """
        jp = self.bvh.forward_kinematics(t, debug=debug)
        active_jp = {idx: jp[idx] for idx in self.active_indices if idx in jp}

        if len(active_jp) < 2:
            return 0.0, 0.0, 0, 0

        # 인덱스를 리스트로 정렬해서 pair 순회
        ids = sorted(active_jp.keys())
        L_rep = 0.0
        L_att = 0.0
        rep_count = 0
        att_count = 0

        R2 = 2.0 * self.joint_radius  # 각 joint를 반지름 r 구로 보고, 중심 거리에서 2r를 빼서 signed distance

        for i_idx, i in enumerate(ids):
            pi = active_jp[i]
            # 간단하게: 바로 뒤 joint들만 보거나, 혹은 전체 다 볼 수도 있음.
            # 여기서는 self.num_nearest에 따라 제한을 둘 수도 있지만, 우선은 전체 pair 사용.
            for j in ids[i_idx+1:]:
                pj = active_jp[j]
                d = float(np.linalg.norm(pi - pj))
                if np.isnan(d):
                    continue
                d_signed = d - R2

                if d_signed < 0.0:
                    # 관통: 거리가 2*r보다 작음
                    L_rep += (-d_signed)
                    rep_count += 1
                elif d_signed < self.contact_band:
                    # 접촉 band 내 근접
                    L_att += (self.contact_band - d_signed) / self.contact_band
                    att_count += 1

        L_rep_mean = (L_rep / rep_count) if rep_count > 0 else 0.0
        L_att_mean = (L_att / att_count) if att_count > 0 else 0.0

        if debug:
            print(f"[EVAL] frame {t}: L_rep={L_rep_mean:.6f} (count={rep_count}), "
                  f"L_att={L_att_mean:.6f} (count={att_count})")

        return L_rep_mean, L_att_mean, rep_count, att_count

    def evaluate_sequence(self) -> Dict:
        T = self.bvh.num_frames
        Lr_list: List[float] = []
        La_list: List[float] = []
        rep_counts: List[int] = []
        att_counts: List[int] = []

        print(f"\n  평가 중: ", end="", flush=True)
        start_all = time.time()
        step = max(1, T // 20)

        for t in range(T):
            if t % step == 0:
                print(f"{t}/{T} ", end="", flush=True)

            try:
                lr, la, rc, ac = self.evaluate_frame(t, debug=(t == 0))
                Lr_list.append(lr)
                La_list.append(la)
                rep_counts.append(rc)
                att_counts.append(ac)
            except Exception as e:
                print(f"\n  Error at frame {t}: {e} — skipping frame.")
                Lr_list.append(0.0)
                La_list.append(0.0)
                rep_counts.append(0)
                att_counts.append(0)

        elapsed = time.time() - start_all
        print(f"{T}/{T} ({elapsed:.2f}초)")

        L_rep_mean = float(np.mean(Lr_list)) if Lr_list else 0.0
        L_att_mean = float(np.mean(La_list)) if La_list else 0.0
        total_rep_frames = int(np.sum(np.array(rep_counts) > 0))
        total_att_frames = int(np.sum(np.array(att_counts) > 0))

        return {
            "L_rep_mean": L_rep_mean,
            "L_att_mean": L_att_mean,
            "L_rep_list": Lr_list,
            "L_att_list": La_list,
            "rep_count_list": rep_counts,
            "att_count_list": att_counts,
            "total_rep_frames": total_rep_frames,
            "total_att_frames": total_att_frames,
            "elapsed_time": elapsed,
        }

# ---------------------------
# Batch Processor
# ---------------------------
class BatchProcessor:
    def __init__(self, data_dir: str, joints_whitelist: List[str]):
        self.data_dir = data_dir
        self.joints_whitelist = joints_whitelist

    def get_bvh_files(self) -> List[str]:
        pattern = os.path.join(self.data_dir, "*.bvh")
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(self.data_dir, "*.BVH")
            files = sorted(glob.glob(pattern))
        return files

    def process_file(self, path: str) -> Dict:
        print(f"\n▶ 처리: {os.path.basename(path)}")
        try:
            bvh = BVH(path)
            print(f"  nodes total: {len(bvh.nodes)}, frames: {bvh.num_frames}, frame_time: {bvh.frame_time:.4f}")
            print(f"nodes total: {len(bvh.nodes)}, frames: {bvh.num_frames}, frame_time: {bvh.frame_time:.4f}")
            print("⚠ FK는 full skeleton 기반으로 진행되고, joint whitelist는 self-contact 평가에서만 사용됩니다.")
            for n in bvh.nodes[:min(8, len(bvh.nodes))]:
                print(f"    node[{n.index}] name='{n.name}' channels={n.channels} ch_idx={n.channel_indices}")

            # joint_radius, contact_band는 필요에 따라 조정 가능
            evaluator = SelfContactEvaluator(
                bvh,
                self.joints_whitelist,
                joint_radius=5.0,
                contact_band=10.0,
                num_nearest=4,
            )
            res = evaluator.evaluate_sequence()

            return {
                "file": os.path.basename(path),
                "status": "success",
                "L_rep_mean": res["L_rep_mean"],
                "L_att_mean": res["L_att_mean"],
                "L_total": res["L_rep_mean"] + res["L_att_mean"],
                "total_rep_frames": res["total_rep_frames"],
                "total_att_frames": res["total_att_frames"],
                "frames_total": bvh.num_frames,
                "elapsed": res["elapsed_time"],
            }
        except Exception as e:
            return {
                "file": os.path.basename(path),
                "status": "error",
                "error": str(e)
            }

    def run(self):
        files = self.get_bvh_files()
        print("\n" + "="*100)
        print("【BVH 자기접촉(Self-Contact) 배치 평가 - robust 수정판】")
        print("="*100)
        print(f"\n📁 데이터 디렉토리: {self.data_dir}")
        print(f"📊 발견된 BVH 파일: {len(files)}개\n")
        if not files:
            print("⚠️ BVH 파일을 찾을 수 없습니다.")
            return {}

        results = []
        for i, p in enumerate(files, 1):
            print(f"  [{i:2d}/{len(files)}] {os.path.basename(p):50s} ", end="", flush=True)
            r = self.process_file(p)
            results.append(r)
            if r.get("status") == "success":
                print(f"✓ ({r['elapsed']:.1f}s) L_rep={r['L_rep_mean']:.6f}  L_att={r['L_att_mean']:.6f}  L_total={r['L_total']:.6f}")
            else:
                print(f"✗ Error: {r.get('error')}")

        # 요약 출력
        ok = [x for x in results if x.get("status") == "success"]
        if ok:
            print("\n" + "-"*120)
            print(f"{'파일명':<50} {'L_rep':>12} {'L_att':>12} {'L_total':>12} {'프레임수':>8}")
            print("-"*120)
            for r in ok:
                print(f"{r['file']:<50} {r['L_rep_mean']:>12.6f} {r['L_att_mean']:>12.6f} {r['L_total']:>12.6f} {r['frames_total']:>8d}")
            print("-"*120)

        with open("results.json", "w", encoding="utf-8") as fo:
            json.dump(results, fo, indent=2, ensure_ascii=False)
        print("\n완료: results.json 저장됨.")
        return results

# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    # 확인할 bvh 데이터 경로 설정
    bp = BatchProcessor(
        "./inference/student/hd_64/1202_1_5",   
        JOINTS_LIST
    )
    bp.run()