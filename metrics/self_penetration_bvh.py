from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Optional, Tuple
import numpy as np
import math
import re
import time
from pathlib import Path

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

def normalize_name(name: str) -> str:
    s = name.replace("mixamorig:", "").replace("mixamo_", "")
    s = s.replace(":", "").strip()
    return s

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

                    if len(vals) >= 6 and any("position" in c.lower() for c in node.channels):
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

class SelfPenetrationEvaluator:
    def __init__(
        self,
        bvh: BVH,
        torso_joints: List[str],
        limb_joints: Dict[str, List[str]],
        penetration_margin: float = 5.0,
        torso_radius: float = 15.0,
    ) -> None:
        self.bvh = bvh
        self.penetration_margin = float(penetration_margin)
        self.torso_radius = float(torso_radius)

        norm_torso = {normalize_name(n) for n in torso_joints}
        self.torso_indices: List[int] = []
        self.torso_names: List[str] = []
        for node in bvh.nodes:
            norm_name = normalize_name(node.name)
            if norm_name in norm_torso:
                self.torso_indices.append(node.index)
                self.torso_names.append(norm_name)

        self.limb_indices: Dict[str, List[int]] = {}
        self.limb_names: Dict[str, List[str]] = {}
        for limb_name, joint_names in limb_joints.items():
            norm_targets = {normalize_name(n) for n in joint_names}
            indices = []
            names = []
            for node in bvh.nodes:
                norm_name = normalize_name(node.name)
                if norm_name in norm_targets:
                    indices.append(node.index)
                    names.append(norm_name)
            if indices:
                self.limb_indices[limb_name] = indices
                self.limb_names[limb_name] = names

        if not self.torso_indices:
            all_joint_names = [normalize_name(n.name) for n in bvh.nodes]
            raise ValueError(f"torso joint를 찾을 수 없음. 찾은 joint들: {all_joint_names[:20]}")
        if not self.limb_indices:
            all_joint_names = [normalize_name(n.name) for n in bvh.nodes]
            raise ValueError(f"limb joint를 찾을 수 없음. 찾은 joint들: {all_joint_names[:20]}")

        print(f"  [Eval] Torso joints: {len(self.torso_indices)}개 - {self.torso_names[:3]}...")
        for limb_name, names in self.limb_names.items():
            print(f"  [Eval] {limb_name}: {len(self.limb_indices[limb_name])}개 - {names[:2]}...")

    def _compute_min_distance_to_torso(self, point: np.ndarray, torso_positions: np.ndarray) -> float:
        if len(torso_positions) == 0:
            return float('inf')
        
        distances = np.linalg.norm(torso_positions - point, axis=1)
        return float(np.min(distances))
    
    def _point_penetrates_torso(self, point: np.ndarray, torso_positions: np.ndarray) -> bool:
        min_dist = self._compute_min_distance_to_torso(point, torso_positions)
        
        if np.isnan(min_dist) or min_dist == float('inf'):
            return False
        
        threshold = self.torso_radius + self.penetration_margin
        
        return min_dist < threshold

    def evaluate(self) -> PenetrationReport:
        frame_results: List[FramePenetration] = []
        total_ratio = 0.0
        limb_ratio_accum = {name: 0.0 for name in self.limb_indices}
        max_ratio = -1.0
        max_ratio_frame = -1

        print(f"\n  평가 중: ", end="", flush=True)
        step = max(1, self.bvh.num_frames // 20)

        for frame_idx in range(self.bvh.num_frames):
            if frame_idx % step == 0:
                print(f"{frame_idx}/{self.bvh.num_frames} ", end="", flush=True)

            try:
                joint_positions = self.bvh.forward_kinematics(frame_idx, debug=(frame_idx == 0))

                torso_positions = np.array([
                    joint_positions[idx] for idx in self.torso_indices if idx in joint_positions
                ], dtype=np.float32)

                if len(torso_positions) < 2:
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

                limb_stats = []
                penetrating_sum = 0
                total_limb_joints = 0

                for limb_name, limb_indices in self.limb_indices.items():
                    limb_joint_positions = [
                        joint_positions[idx] for idx in limb_indices if idx in joint_positions
                    ]

                    if not limb_joint_positions:
                        continue

                    limb_positions = np.array(limb_joint_positions, dtype=np.float32)
                    total_limb_joints += len(limb_positions)

                    penetrating_count = 0
                    for pos in limb_positions:
                        if self._point_penetrates_torso(pos, torso_positions):
                            penetrating_count += 1

                    ratio = penetrating_count / len(limb_positions) if len(limb_positions) > 0 else 0.0
                    penetrating_sum += penetrating_count

                    limb_stats.append(
                        LimbPenetration(
                            name=limb_name,
                            penetrating=penetrating_count,
                            total=len(limb_positions),
                            ratio=ratio,
                        )
                    )
                    limb_ratio_accum[limb_name] += ratio

                total_ratio_frame = penetrating_sum / total_limb_joints if total_limb_joints > 0 else 0.0
                total_ratio += total_ratio_frame

                if total_ratio_frame > max_ratio:
                    max_ratio = total_ratio_frame
                    max_ratio_frame = frame_idx

                frame_results.append(
                    FramePenetration(
                        frame_index=frame_idx,
                        total_penetrating=penetrating_sum,
                        total_vertices=total_limb_joints,
                        total_ratio=total_ratio_frame,
                        limb_stats=limb_stats,
                    )
                )
            except Exception as e:
                print(f"\n  Error at frame {frame_idx}: {e}")
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
                for name in self.limb_indices
            },
            penetration_margin=self.penetration_margin,
        )
        return PenetrationReport(frames=frame_results, summary=summary)

def evaluate_bvh_file(
    bvh_path: str,
    torso_joints: Optional[List[str]] = None,
    limb_joints: Optional[Dict[str, List[str]]] = None,
    penetration_margin: float = 5.0,
    torso_radius: float = 15.0,
) -> PenetrationReport:
    if torso_joints is None:
        torso_joints = TORSO_JOINTS
    if limb_joints is None:
        limb_joints = LIMB_JOINTS

    bvh = BVH(bvh_path)
    evaluator = SelfPenetrationEvaluator(
        bvh,
        torso_joints=torso_joints,
        limb_joints=limb_joints,
        penetration_margin=penetration_margin,
        torso_radius=torso_radius,
    )
    return evaluator.evaluate()

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        bvh_files = [
            "Steve_to_Claire_martelo 3.bvh",
            "Claire_gt_martelo 3.bvh"
        ]
        
        existing_files = []
        for f in bvh_files:
            if Path(f).exists():
                existing_files.append(f)
        
        if not existing_files:
            print("사용법:")
            print("  python self_penetration_eval.py <bvh_path> [penetration_margin] [torso_radius]")
            print("  python self_penetration_eval.py  (현재 디렉토리의 BVH 파일 자동 평가)")
            print("\n예시:")
            print("  python self_penetration_eval.py 'Steve_to_Claire_martelo 3.bvh'")
            print("  python self_penetration_eval.py 'Claire_gt_martelo 3.bvh' 5.0 15.0")
            sys.exit(1)
        
        bvh_files = existing_files
        penetration_margin = 5.0
        torso_radius = 15.0
    else:
        bvh_path = sys.argv[1]
        penetration_margin = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        torso_radius = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
        
        if not Path(bvh_path).exists():
            print(f"오류: BVH 파일을 찾을 수 없습니다: {bvh_path}")
            sys.exit(1)
        
        bvh_files = [bvh_path]

    print("=" * 100)
    print("【BVH 자기 관통(Self-Penetration) 평가】")
    print("=" * 100)
    
    results = {}
    
    for bvh_path in bvh_files:
        try:
            print(f"\n▶ 처리 중: {bvh_path}")
            start_time = time.time()
            
            bvh = BVH(bvh_path)
            print(f"  nodes total: {len(bvh.nodes)}, frames: {bvh.num_frames}, frame_time: {bvh.frame_time:.4f}")
            
            report = evaluate_bvh_file(
                bvh_path,
                penetration_margin=penetration_margin,
                torso_radius=torso_radius
            )
            
            elapsed = time.time() - start_time
            result = report.to_dict()
            results[bvh_path] = result
            
            summary = result["summary"]
            print(f"\n  평균 관통 비율: {summary['mean_ratio']:.6f}")
            print(f"  최대 관통 비율: {summary['max_ratio']:.6f} (프레임 {summary['max_ratio_frame']})")
            print(f"  프레임 수: {len(result['frames'])}")
            print(f"  소요 시간: {elapsed:.2f}초")
            
            if summary['mean_ratio_by_limb']:
                print("  팔다리별 평균:")
                for limb, ratio in summary['mean_ratio_by_limb'].items():
                    print(f"    {limb}: {ratio:.6f}")
                    
        except Exception as e:
            print(f"  ✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) > 1:
        print("\n" + "-" * 100)
        print(f"{'파일명':<50} {'평균 비율':>15} {'최대 비율':>15} {'프레임 수':>10}")
        print("-" * 100)
        for path, result in results.items():
            summary = result["summary"]
            filename = Path(path).name
            print(f"{filename:<50} {summary['mean_ratio']:>15.6f} {summary['max_ratio']:>15.6f} {len(result['frames']):>10d}")
        print("-" * 100)
    
    print("\n평가 완료!")
    
    if len(results) == 1:
        print("\n상세 결과:")
        print(json.dumps(list(results.values())[0], indent=2, ensure_ascii=False))
