#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-Penetration 평가 모듈

함수:
    evaluate_bvh_file(bvh_path: str,
                      mesh_npz_path: str,
                      penetration_margin: float = 0.01) -> PenetrationReport

PenetrationReport.summary.mean_ratio 를
"관통된 limb vertices 비율(프레임 평균)" 로 사용하면 된다.

NPZ 포맷(예시):
    - rest_vertices       (V,3)
    - rest_body_vertices  (Nb,3)
    - rest_arm_vertices   (Na,3)
    - rest_faces          (F,3)
    - skinning_weights    (V,J)
    - skeleton            (J,3)
    - joint_names         (J,)
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import re
from pathlib import Path
from scipy.spatial import cKDTree


# ================== 상수/설정 ==================

R2ET_JOINTS_LIST = [
    "Hip",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]

TORSO_JOINTS = ["Hip", "Spine", "Spine1", "Spine2", "Neck", "Head"]
LIMB_JOINTS = {
    "left_arm": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    "right_arm": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
    "left_leg": ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"],
    "right_leg": ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"],
}


# ================== dataclass 정의 ==================

@dataclass(frozen=True)
class LimbPenetration:
    name: str
    penetrating: int
    total: int
    ratio: float

@dataclass(frozen=True)
class FramePenetration:
    frame_index: int
    total_penetrating: int
    total_vertices: int
    total_ratio: float
    limb_stats: List[LimbPenetration]

@dataclass(frozen=True)
class PenetrationSummary:
    mean_ratio: float
    max_ratio: float
    max_ratio_frame: int
    mean_ratio_by_limb: Dict[str, float]
    penetration_margin: float

@dataclass(frozen=True)
class PenetrationReport:
    frames: List[FramePenetration]
    summary: PenetrationSummary

    def to_dict(self) -> Dict:
        return {
            "summary": asdict(self.summary),
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "total_penetrating": f.total_penetrating,
                    "total_vertices": f.total_vertices,
                    "total_ratio": f.total_ratio,
                    "limb_stats": [asdict(l) for l in f.limb_stats],
                }
                for f in self.frames
            ],
        }


# ================== 유틸 함수 ==================

def normalize_name(name: str) -> str:
    s = name.replace("mixamorig:", "").replace("mixamo_", "")
    s = s.replace(":", "").strip()
    return s

def load_mesh_data(npz_path: str) -> Dict:
    data = np.load(npz_path, allow_pickle=True)
    rest_vertices = data['rest_vertices'].astype(np.float32)
    rest_body_vertices = data['rest_body_vertices'].astype(np.float32)
    rest_arm_vertices = data['rest_arm_vertices'].astype(np.float32)
    
    body_indices = []
    arm_indices = []
    
    for i, v in enumerate(rest_vertices):
        body_dists = np.linalg.norm(rest_body_vertices - v, axis=1)
        arm_dists = np.linalg.norm(rest_arm_vertices - v, axis=1)
        
        min_body_dist = np.min(body_dists)
        min_arm_dist = np.min(arm_dists)
        
        if min_body_dist < min_arm_dist and min_body_dist < 1e-3:
            body_indices.append(i)
        elif min_arm_dist < 1e-3:
            arm_indices.append(i)
    
    return {
        'rest_vertices': rest_vertices,
        'rest_faces': data['rest_faces'],
        'skinning_weights': data['skinning_weights'].astype(np.float32),
        'joint_names': [str(n) for n in data['joint_names']],
        'skeleton': data['skeleton'].astype(np.float32),
        'body_indices': np.array(body_indices, dtype=np.int32),
        'arm_indices': np.array(arm_indices, dtype=np.int32),
    }


# ================== BVH 파서 ==================

class BVH_Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.offset = np.zeros(3, dtype=np.float32)
        self.channels = []
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

            if l.upper().startswith("END SITE") or l.startswith("End") or l.startswith("END"):
                i += 1
                while i < len(lines) and "}" not in lines[i]:
                    i += 1
                i += 1
                continue

            if l.startswith("}"):
                if stack:
                    stack.pop()
                i += 1
                continue

            if l.strip().upper().startswith("MOTION"):
                break

            i += 1

    def _map_channels(self):
        idx = 0
        for node in self.nodes:
            node.channel_indices = list(range(idx, idx + len(node.channels)))
            idx += len(node.channels)

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
        rows = motion_raw.split("\n")[1:]

        frame_data = []
        for l in rows:
            cols = l.strip().split()
            if len(cols) > 0:
                frame_data.append(list(map(float, cols)))

        self.frames = np.array(frame_data, dtype=np.float32)

        if self.frames.shape[0] < self.num_frames:
            self.num_frames = self.frames.shape[0]

    def forward_kinematics_transforms(self, frame_idx: int, mesh_joint_names: List[str], debug: bool = False) -> Dict[str, np.ndarray]:
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError("frame_idx out of range")

        frame = self.frames[frame_idx].tolist()
        joint_transforms: Dict[str, np.ndarray] = {}

        def euler_to_R(angles_deg, order="ZYX"):
            try:
                ax, ay, az = float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2])
            except Exception:
                return np.eye(3, dtype=np.float32)

            rx = math.radians(ax)
            ry = math.radians(ay)
            rz = math.radians(az)

            cx, sx = math.cos(rx), math.sin(rx)
            cy, sy = math.cos(ry), math.sin(ry)
            cz, sz = math.cos(rz), math.sin(rz)

            Rx = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cx, -sx],
                [0.0, sx,  cx],
            ], dtype=np.float32)

            Ry = np.array([
                [ cy, 0.0, sy],
                [ 0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ], dtype=np.float32)

            Rz = np.array([
                [cz, -sz, 0.0],
                [sz,  cz, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

            if order == "ZYX":
                return Rx @ Ry @ Rz
            else:
                return Rz @ Ry @ Rx

        def build_transform(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rot
            T[:3, 3] = pos
            return T

        bvh_joint_map: Dict[str, BVH_Node] = {}
        for node in self.nodes:
            norm_name = normalize_name(node.name)
            bvh_joint_map[norm_name] = node

        mesh_joint_to_bvh: Dict[str, Optional[BVH_Node]] = {}
        for mesh_joint_name in mesh_joint_names:
            norm_mesh_name = normalize_name(mesh_joint_name)
            best_match = None
            for bvh_name, bvh_node in bvh_joint_map.items():
                if norm_mesh_name == bvh_name or norm_mesh_name in bvh_name or bvh_name in norm_mesh_name:
                    best_match = bvh_node
                    break
            mesh_joint_to_bvh[mesh_joint_name] = best_match

        stack: List[Tuple[BVH_Node, np.ndarray, np.ndarray]] = [
            (self.root, np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
        ]

        visited = set()
        steps = 0
        MAX_STEPS = len(self.nodes) * 10 + 10

        while stack:
            node, parent_pos, parent_rot = stack.pop()

            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)

            steps += 1
            if steps > MAX_STEPS:
                break

            local_pos = node.offset.copy()
            local_rot = np.eye(3, dtype=np.float32)

            if node.channel_indices:
                try:
                    vals = [frame[i] for i in node.channel_indices]

                    if len(vals) >= 6 and any("position" in c.lower() for c in node.channels):
                        local_pos = np.array([vals[0], vals[1], vals[2]], dtype=np.float32)
                        local_rot = euler_to_R(vals[3:6], order="ZYX")
                    elif len(vals) >= 3:
                        local_rot = euler_to_R(vals[:3], order="ZYX")
                except Exception:
                    pass

            rotated = parent_rot @ local_pos
            gpos = parent_pos + rotated
            grot = parent_rot @ local_rot

            norm_name = normalize_name(node.name)
            for mesh_joint_name, bvh_node in mesh_joint_to_bvh.items():
                if bvh_node == node:
                    T = build_transform(grot, gpos)
                    joint_transforms[mesh_joint_name] = T

            for c in node.children:
                stack.append((c, gpos, grot))

        return joint_transforms


# ================== LBS ==================

def apply_skinning(rest_vertices: np.ndarray, skinning_weights: np.ndarray, 
                   joint_transforms: Dict[str, np.ndarray], 
                   rest_skeleton: np.ndarray, joint_names: List[str]) -> np.ndarray:
    num_vertices = rest_vertices.shape[0]
    num_joints = len(joint_names)
    
    skinned_vertices = np.zeros_like(rest_vertices)
    
    vertices_homogeneous = np.ones((num_vertices, 4), dtype=np.float32)
    vertices_homogeneous[:, :3] = rest_vertices
    
    for j_idx, joint_name in enumerate(joint_names):
        if joint_name not in joint_transforms:
            continue
        
        T = joint_transforms[joint_name]
        
        rest_joint_pos = rest_skeleton[j_idx]
        rest_joint_T = np.eye(4, dtype=np.float32)
        rest_joint_T[:3, 3] = rest_joint_pos
        
        relative_T = T @ np.linalg.inv(rest_joint_T)
        
        weights = skinning_weights[:, j_idx:j_idx+1]
        
        transformed = (relative_T @ vertices_homogeneous.T).T[:, :3]
        
        skinned_vertices += weights * transformed
    
    return skinned_vertices


# ================== Self Penetration Evaluator ==================

class SelfPenetrationEvaluator:
    def __init__(
        self,
        bvh: BVH,
        mesh_data: Dict,
        torso_joints: List[str],
        limb_joints: Dict[str, List[str]],
        penetration_margin: float = 0.01,
    ) -> None:
        self.bvh = bvh
        self.mesh_data = mesh_data
        self.penetration_margin = float(penetration_margin)
        
        self.rest_vertices = mesh_data['rest_vertices']
        self.skinning_weights = mesh_data['skinning_weights']
        self.joint_names = mesh_data['joint_names']
        self.rest_skeleton = mesh_data['skeleton']
        self.body_indices = mesh_data['body_indices']
        self.arm_indices = mesh_data['arm_indices']
        
        self.torso_vertex_indices = self.body_indices.tolist()
        
        rest_arm_vertices = self.rest_vertices[self.arm_indices]
        left_arm_mask = rest_arm_vertices[:, 0] < np.median(rest_arm_vertices[:, 0])
        right_arm_mask = ~left_arm_mask
        
        left_arm_indices = self.arm_indices[left_arm_mask].tolist()
        right_arm_indices = self.arm_indices[right_arm_mask].tolist()
        
        rest_leg_vertices = self.rest_vertices[
            ~np.isin(np.arange(len(self.rest_vertices)),
                     np.concatenate([self.body_indices, self.arm_indices]))
        ]
        if len(rest_leg_vertices) > 0:
            leg_indices_all = np.setdiff1d(
                np.arange(len(self.rest_vertices)),
                np.concatenate([self.body_indices, self.arm_indices])
            )
            left_leg_mask = rest_leg_vertices[:, 0] < np.median(rest_leg_vertices[:, 0])
            right_leg_mask = ~left_leg_mask
            
            left_leg_indices = leg_indices_all[left_leg_mask].tolist()
            right_leg_indices = leg_indices_all[right_leg_mask].tolist()
        else:
            left_leg_indices = []
            right_leg_indices = []
        
        self.limb_vertex_indices: Dict[str, List[int]] = {}
        if left_arm_indices:
            self.limb_vertex_indices['left_arm'] = left_arm_indices
        if right_arm_indices:
            self.limb_vertex_indices['right_arm'] = right_arm_indices
        if left_leg_indices:
            self.limb_vertex_indices['left_leg'] = left_leg_indices
        if right_leg_indices:
            self.limb_vertex_indices['right_leg'] = right_leg_indices
        
        if len(self.torso_vertex_indices) == 0:
            raise ValueError(f"torso vertex를 찾을 수 없음")
        if not self.limb_vertex_indices:
            raise ValueError(f"limb vertex를 찾을 수 없음")

        print(f"  [PenEval] Torso vertices: {len(self.torso_vertex_indices)}개")
        for limb_name, indices in self.limb_vertex_indices.items():
            print(f"  [PenEval] {limb_name}: {len(indices)}개 vertices")

    def _points_penetrate_torso(self, points: np.ndarray, torso_tree: cKDTree) -> np.ndarray:
        if len(points) == 0:
            return np.array([], dtype=bool)
        
        distances, _ = torso_tree.query(points, k=1)
        return distances < self.penetration_margin

    def evaluate(self) -> PenetrationReport:
        frame_results: List[FramePenetration] = []
        total_ratio = 0.0
        limb_ratio_accum = {name: 0.0 for name in self.limb_vertex_indices}
        max_ratio = -1.0
        max_ratio_frame = -1

        print(f"\n  [PenEval] 평가 중: ", end="", flush=True)
        step = max(1, self.bvh.num_frames // 20)

        for frame_idx in range(self.bvh.num_frames):
            if frame_idx % step == 0:
                print(f"{frame_idx}/{self.bvh.num_frames} ", end="", flush=True)

            try:
                joint_transforms = self.bvh.forward_kinematics_transforms(
                    frame_idx, self.joint_names, debug=False
                )
                
                skinned_vertices = apply_skinning(
                    self.rest_vertices,
                    self.skinning_weights,
                    joint_transforms,
                    self.rest_skeleton,
                    self.joint_names
                )
                
                torso_vertices = skinned_vertices[self.torso_vertex_indices]

                if len(torso_vertices) < 3:
                    frame_results.append(
                        FramePenetration(
                            frame_index=frame_idx,
                            total_penetrating=0,
                            total_vertices=0,
                            total_ratio=0.0,
                            limb_stats=[],
                        )
                    )
                    continue

                torso_tree = cKDTree(torso_vertices)

                limb_stats = []
                penetrating_sum = 0
                total_limb_vertices = 0

                for limb_name, limb_indices in self.limb_vertex_indices.items():
                    limb_vertices = skinned_vertices[limb_indices]
                    total_limb_vertices += len(limb_vertices)

                    penetrating_mask = self._points_penetrate_torso(limb_vertices, torso_tree)
                    penetrating_count = int(np.sum(penetrating_mask))

                    ratio = penetrating_count / len(limb_vertices) if len(limb_vertices) > 0 else 0.0
                    penetrating_sum += penetrating_count

                    limb_stats.append(
                        LimbPenetration(
                            name=limb_name,
                            penetrating=penetrating_count,
                            total=len(limb_vertices),
                            ratio=ratio,
                        )
                    )
                    limb_ratio_accum[limb_name] += ratio

                total_ratio_frame = penetrating_sum / total_limb_vertices if total_limb_vertices > 0 else 0.0
                total_ratio += total_ratio_frame

                if total_ratio_frame > max_ratio:
                    max_ratio = total_ratio_frame
                    max_ratio_frame = frame_idx

                frame_results.append(
                    FramePenetration(
                        frame_index=frame_idx,
                        total_penetrating=penetrating_sum,
                        total_vertices=total_limb_vertices,
                        total_ratio=total_ratio_frame,
                        limb_stats=limb_stats,
                    )
                )
            except Exception as e:
                print(f"\n  Error at frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                frame_results.append(
                    FramePenetration(
                        frame_index=frame_idx,
                        total_penetrating=0,
                        total_vertices=0,
                        total_ratio=0.0,
                        limb_stats=[],
                    )
                )

        print(f"{self.bvh.num_frames}/{self.bvh.num_frames}")

        frame_count = len(frame_results)
        summary = PenetrationSummary(
            mean_ratio=total_ratio / frame_count if frame_count else 0.0,
            max_ratio=max_ratio if max_ratio >= 0 else 0.0,
            max_ratio_frame=max_ratio_frame,
            mean_ratio_by_limb={
                name: limb_ratio_accum[name] / frame_count if frame_count else 0.0
                for name in self.limb_vertex_indices
            },
            penetration_margin=self.penetration_margin,
        )
        return PenetrationReport(frames=frame_results, summary=summary)


# ================== 외부에서 호출할 함수 ==================

def evaluate_bvh_file(
    bvh_path: str,
    mesh_npz_path: str,
    torso_joints: Optional[List[str]] = None,
    limb_joints: Optional[Dict[str, List[str]]] = None,
    penetration_margin: float = 0.01,
) -> PenetrationReport:
    """
    bvh_path + mesh_npz_path 에 대해 관통 비율 평가
    PenetrationReport.summary.mean_ratio 를 self-penetration metric으로 사용
    """
    if torso_joints is None:
        torso_joints = TORSO_JOINTS
    if limb_joints is None:
        limb_joints = LIMB_JOINTS

    bvh = BVH(bvh_path)
    mesh_data = load_mesh_data(mesh_npz_path)
    evaluator = SelfPenetrationEvaluator(
        bvh,
        mesh_data,
        torso_joints=torso_joints,
        limb_joints=limb_joints,
        penetration_margin=penetration_margin,
    )
    return evaluator.evaluate()
