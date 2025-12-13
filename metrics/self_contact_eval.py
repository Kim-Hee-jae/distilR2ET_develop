#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R2ET 스타일 Self-Contact 평가 모듈

함수:
    evaluate_bvh_self_contact_r2et_style(bvh_path: str, npz_path: str) -> dict

반환 예:
    {
        "L_rep_mean": float,   # limb-body repulsive term (self-collision 억제)
        "L_att_mean": float,   # hand-body attractive term (self-contact 유지)
        "frames": int,         # 평가된 프레임 수
    }

NPZ 포맷(예시, Claire.npz):
    - rest_vertices    (V, 3)
    - skinning_weights (V, J)
    - skeleton         (J, 3)
    - joint_names      (J,)
    - vertex_part      (V,)  # 각 vertex가 어떤 joint 파트에 속하는지 (0~J-1)

주의: 여기서는 distance-field 대신 body vertex에 대한 최근접거리로 근사함.
"""

import os
import re
import math
import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# 1) BVH 파서 + FK
# ============================================================

class BVH:
    """
    간단한 Mixamo 스타일 BVH 로더
    - HIERARCHY: joint 트리/offset/channels 파싱
    - MOTION   : frame별 채널 값 파싱
    - fk(frame_idx) → {joint_idx: (pos(3,), rot(3x3))}
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "r", errors="ignore") as f:
            self.lines = f.read().splitlines()
        self.nodes = []
        self._parse_hierarchy()
        self._parse_motion()

    def _parse_hierarchy(self):
        lines = self.lines
        stack = []
        idx = 0
        motion_start = None

        for i, line in enumerate(lines):
            if "MOTION" in line:
                motion_start = i
                break

            stripped = line.strip()

            if stripped.startswith("ROOT") or stripped.startswith("JOINT"):
                name = stripped.split()[-1]
                parent = stack[-1] if stack else None
                node = {
                    "name": name,
                    "parent": parent,     # dict 또는 None
                    "offset": np.zeros(3, dtype=np.float32),
                    "channels": [],
                    "idx": idx
                }
                self.nodes.append(node)
                idx += 1

            elif stripped.startswith("OFFSET") and self.nodes:
                parts = stripped.split()
                self.nodes[-1]["offset"] = np.array(
                    list(map(float, parts[1:4])),
                    dtype=np.float32
                )

            elif stripped.startswith("CHANNELS") and self.nodes:
                parts = stripped.split()
                chs = parts[2:]   # 예: ['Xposition','Yposition','Zposition','Zrotation','Yrotation','Xrotation']
                self.nodes[-1]["channels"] = chs

            # 계층 스택
            if "{" in stripped and self.nodes:
                stack.append(self.nodes[-1])
            if "}" in stripped and stack:
                stack.pop()

        if motion_start is None:
            raise RuntimeError("BVH: MOTION 섹션을 찾지 못했습니다.")
        self.motion_start = motion_start

    def _parse_motion(self):
        lines = self.lines

        frames_line = None
        frame_time_line = None
        for i in range(self.motion_start, len(lines)):
            if "Frames:" in lines[i]:
                frames_line = i
            elif "Frame Time:" in lines[i]:
                frame_time_line = i
                break

        if frames_line is None or frame_time_line is None:
            raise RuntimeError("Frames / Frame Time 라인을 찾지 못했습니다.")

        self.num_frames = int(lines[frames_line].split()[-1])
        self.frame_time = float(lines[frame_time_line].split()[-1])

        # joint별 채널 인덱스 맵
        self.channel_map = []
        total_ch = 0
        for n in self.nodes:
            n_ch = len(n["channels"])
            self.channel_map.append((total_ch, total_ch + n_ch))
            total_ch += n_ch

        # MOTION 숫자 파싱
        numeric_text = "\n".join(lines[frame_time_line + 1 :])
        tokens = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", numeric_text)
        arr = np.array(list(map(float, tokens)), dtype=np.float32)

        if arr.size < self.num_frames * total_ch:
            self.num_frames = arr.size // total_ch

        arr = arr[: self.num_frames * total_ch]
        self.frames = arr.reshape(self.num_frames, total_ch)

        print(f"[BVH] Loaded '{self.path}' | joints={len(self.nodes)}, frames={self.num_frames}")

    @staticmethod
    def _euler_z_y_x(z, y, x):
        """
        Z-Y-X 순서의 Euler 각 → 3x3 회전 행렬
        (순수 파이썬 + math만 사용)
        """
        z = math.radians(z)
        y = math.radians(y)
        x = math.radians(x)

        cz, sz = math.cos(z), math.sin(z)
        cy, sy = math.cos(y), math.sin(y)
        cx, sx = math.cos(x), math.sin(x)

        Rz = [
            [cz, -sz, 0.0],
            [sz,  cz, 0.0],
            [0.0, 0.0, 1.0],
        ]
        Ry = [
            [ cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ]
        Rx = [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx,  cx],
        ]

        def matmul3(A, B):
            return [
                [
                    A[r][0] * B[0][c] +
                    A[r][1] * B[1][c] +
                    A[r][2] * B[2][c]
                    for c in range(3)
                ]
                for r in range(3)
            ]

        return matmul3(matmul3(Rz, Ry), Rx)

    def fk(self, frame_idx: int):
        """
        Forward Kinematics
        - frame_idx 프레임에 대해 각 joint의 global pos/rot 계산
        - 반환: {joint_idx: (pos(np.array(3,)), rot(np.array(3,3)))}
        """
        f = self.frames[frame_idx]
        out = {}

        def matvec3(R, v):
            return [
                R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
                R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
                R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
            ]

        def matmul3(A, B):
            return [
                [
                    A[r][0]*B[0][c] +
                    A[r][1]*B[1][c] +
                    A[r][2]*B[2][c]
                    for c in range(3)
                ]
                for r in range(3)
            ]

        for i, n in enumerate(self.nodes):
            name   = n["name"]
            chs    = n["channels"]
            offset = n["offset"].astype(float).tolist()

            pos = offset[:]
            R   = [[1.0,0.0,0.0],
                   [0.0,1.0,0.0],
                   [0.0,0.0,1.0]]

            s, e = self.channel_map[i]
            vals = f[s:e]

            if len(chs) == 6:
                tx, ty, tz = float(vals[0]), float(vals[1]), float(vals[2])
                rz, ry, rx = float(vals[3]), float(vals[4]), float(vals[5])
                pos = [tx, ty, tz]
                R   = self._euler_z_y_x(rz, ry, rx)
            elif len(chs) == 3:
                rz, ry, rx = float(vals[0]), float(vals[1]), float(vals[2])
                R = self._euler_z_y_x(rz, ry, rx)

            parent = n["parent"]
            if parent is not None:
                pidx = parent["idx"]
                ppos, pR = out[pidx]

                pR_list = pR.tolist()
                pos_world = matvec3(pR_list, pos)
                pos = [
                    float(ppos[0] + pos_world[0]),
                    float(ppos[1] + pos_world[1]),
                    float(ppos[2] + pos_world[2]),
                ]
                R = matmul3(pR_list, R)

            out[i] = (
                np.array(pos, dtype=np.float32),
                np.array(R,   dtype=np.float32),
            )

        return out


# ============================================================
# 2) LBS (Linear Blend Skinning)
# ============================================================

def compute_joint_transforms_for_frame(bvh: BVH,
                                       frame_idx: int,
                                       joint_names,
                                       bvh_mapping,
                                       skeleton_rest):
    """
    joint_names: npz 기준 joint 이름 (J,)
    skeleton_rest: (J,3) rest pose joint 위치
    bvh_mapping: npz joint_name → BVH joint index
    반환: (J,4,4) joint transform 행렬
    """
    fk_res = bvh.fk(frame_idx)
    J = len(joint_names)
    B = np.zeros((J, 4, 4), dtype=np.float32)

    def matvec3(R_list, v_list):
        return [
            R_list[0][0]*v_list[0] + R_list[0][1]*v_list[1] + R_list[0][2]*v_list[2],
            R_list[1][0]*v_list[0] + R_list[1][1]*v_list[1] + R_list[1][2]*v_list[2],
            R_list[2][0]*v_list[0] + R_list[2][1]*v_list[1] + R_list[2][2]*v_list[2],
        ]

    for j, name in enumerate(joint_names):
        bvh_idx = bvh_mapping[name]
        pos, R = fk_res[bvh_idx]
        p_r = skeleton_rest[j].astype(np.float32)

        B[j, :3, :3] = R

        R_list  = R.tolist()
        p_r_list = p_r.tolist()
        Rpr = matvec3(R_list, p_r_list)

        B[j, 0, 3] = pos[0] - Rpr[0]
        B[j, 1, 3] = pos[1] - Rpr[1]
        B[j, 2, 3] = pos[2] - Rpr[2]
        B[j, 3, :] = [0.0, 0.0, 0.0, 1.0]

    return B


def deform_vertices(rest_vertices: np.ndarray,
                    skin_weights: np.ndarray,
                    B: np.ndarray):
    """
    NumPy 고수준 연산 없이 구현한 LBS
    rest_vertices: (V,3)
    skin_weights : (V,J)
    B            : (J,4,4)
    """
    V = rest_vertices.shape[0]
    J = skin_weights.shape[1]

    v_def = np.zeros((V, 3), dtype=np.float32)

    for j in range(J):
        wj = skin_weights[:, j]
        if np.allclose(wj, 0):
            continue

        Bj = B[j]
        Rj = Bj[:3, :3]
        tj = Bj[:3, 3]

        R = Rj.tolist()
        t = tj.tolist()

        for v in range(V):
            if wj[v] == 0:
                continue

            x, y, z = rest_vertices[v]

            vx = R[0][0]*x + R[0][1]*y + R[0][2]*z
            vy = R[1][0]*x + R[1][1]*y + R[1][2]*z
            vz = R[2][0]*x + R[2][1]*y + R[2][2]*z

            px = vx + t[0]
            py = vy + t[1]
            pz = vz + t[2]

            w = wj[v]
            v_def[v, 0] += w * px
            v_def[v, 1] += w * py
            v_def[v, 2] += w * pz

    return v_def


# ============================================================
# 3) R2ET 스타일 Self-Contact metric
# ============================================================

def min_distances(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """
    src: (Ns,3), tgt: (Nt,3)
    src의 각 점에 대해 tgt까지의 최근접 거리
    """
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return np.zeros((src.shape[0],), dtype=np.float32)
    tree = cKDTree(tgt)
    dists, _ = tree.query(src, k=1)
    return dists.astype(np.float32)


def self_contact_frame_metrics(
    v_def: np.ndarray,
    body_vs: np.ndarray,
    limb_vs: np.ndarray,
    hand_vs: np.ndarray,
    repulsive_threshold: float = 2.0,
    attractive_band: float = 10.0,
):
    """
    한 프레임에 대해 self-contact 근사 metric 계산

    - body_vs: body 파트 vertex 인덱스
    - limb_vs: 팔/다리 파트 vertex 인덱스 (관통 억제 대상)
    - hand_vs: 손 파트 vertex 인덱스 (contact 유지 대상)

    L_rep:
      pen = max(0, repulsive_threshold - d_limb)
      L_rep = mean(pen)

    L_att:
      mask = d_hand < attractive_band
      val  = (attractive_band - d_hand) / attractive_band
      L_att = mean(val[mask])
    """
    body = v_def[body_vs]
    limb = v_def[limb_vs]
    hand = v_def[hand_vs]

    d_limb = min_distances(limb, body)
    d_hand = min_distances(hand, body)

    pen = np.maximum(0.0, repulsive_threshold - d_limb)
    L_rep = float(pen.mean()) if pen.size > 0 else 0.0

    mask = d_hand < attractive_band
    if mask.any():
        val = (attractive_band - d_hand[mask]) / attractive_band
        L_att = float(val.mean())
    else:
        L_att = 0.0

    return L_rep, L_att


# ============================================================
# 4) 전체 평가 함수 (외부에서 호출)
# ============================================================

def evaluate_bvh_self_contact_r2et_style(bvh_path: str, npz_path: str,
                                         repulsive_threshold: float = 10.0,
                                         attractive_band: float = 60.0) -> dict:
    """
    bvh_path, npz_path를 받아 R2ET 스타일 self-contact metric 반환
    반환:
        {
            "L_rep_mean": float,
            "L_att_mean": float,
            "frames": int,
        }
    """

    data = np.load(npz_path, allow_pickle=True)
    rest_vertices    = data["rest_vertices"]
    skinning_weights = data["skinning_weights"]
    skeleton_rest    = data["skeleton"]
    joint_names      = [j for j in data["joint_names"]]
    vertex_part      = data["vertex_part"]

    V = rest_vertices.shape[0]
    J = len(joint_names)
    print(f"[SELF-CONTACT] '{os.path.basename(bvh_path)}' / vertices={V}, joints={J}")

    # joint_name → id
    joint_name_to_id = {name: i for i, name in enumerate(joint_names)}

    # body: Hips~Head, hand: LeftHand/RightHand, limb: 그 외
    body_joint_ids = [joint_name_to_id[n] for n in ["Hips","Spine","Spine1","Spine2","Neck","Head"]]
    hand_joint_ids = [joint_name_to_id["LeftHand"], joint_name_to_id["RightHand"]]
    all_joint_ids  = list(range(J))
    limb_joint_ids = [j for j in all_joint_ids if j not in body_joint_ids + hand_joint_ids]

    body_vs = np.where(np.isin(vertex_part, body_joint_ids))[0]
    hand_vs = np.where(np.isin(vertex_part, hand_joint_ids))[0]
    limb_vs = np.where(np.isin(vertex_part, limb_joint_ids))[0]

    print(f"  - body vertices : {len(body_vs)}")
    print(f"  - limb vertices : {len(limb_vs)}")
    print(f"  - hand vertices : {len(hand_vs)}")

    # BVH 로드 + joint 이름 매핑
    bvh = BVH(bvh_path)

    def normalize(name):
        return name.replace("mixamorig:", "").lower()

    bvh_map = {normalize(n["name"]): i for i, n in enumerate(bvh.nodes)}

    bvh_mapping = {}
    for jn in joint_names:
        key = normalize(jn)
        if key not in bvh_map:
            raise KeyError(f"[ERR] BVH에 존재하지 않는 joint: {jn}")
        bvh_mapping[jn] = bvh_map[key]

    # 프레임 루프
    Lr_list = []
    La_list = []

    for fidx in range(bvh.num_frames):
        B = compute_joint_transforms_for_frame(
            bvh,
            fidx,
            joint_names,
            bvh_mapping,
            skeleton_rest
        )
        v_def = deform_vertices(rest_vertices, skinning_weights, B)

        Lr, La = self_contact_frame_metrics(
            v_def,
            body_vs,
            limb_vs,
            hand_vs,
            repulsive_threshold=repulsive_threshold,
            attractive_band=attractive_band,
        )

        Lr_list.append(Lr)
        La_list.append(La)

    if len(Lr_list) == 0:
        L_rep_mean = float("nan")
    else:
        L_rep_mean = float(np.mean(Lr_list))

    if len(La_list) == 0:
        L_att_mean = float("nan")
    else:
        L_att_mean = float(np.mean(La_list))

    return {
        "L_rep_mean": L_rep_mean,
        "L_att_mean": L_att_mean,
        "frames": len(Lr_list),
    }
