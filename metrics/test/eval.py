from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Optional, Tuple
import numpy as np
import math
import re
import time
from pathlib import Path
from scipy.spatial import cKDTree

try:
    import trimesh
    TRIMESH_AVAILABLE = True
    print("Imported")
except ImportError:
    TRIMESH_AVAILABLE = False

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
class LimbLoss:
    name: str
    lrep: float
    latt: float

@dataclass(frozen=True)
class FrameLoss:
    frame_index: int
    lrep_total: float
    latt_total: float
    lrep_by_limb: Dict[str, float]
    latt_by_limb: Dict[str, float]
    limb_stats: List[LimbLoss]

@dataclass(frozen=True)
class LossSummary:
    mean_lrep: float
    mean_latt: float
    total_lrep: float
    total_latt: float
    mean_lrep_by_limb: Dict[str, float]
    mean_latt_by_limb: Dict[str, float]

@dataclass(frozen=True)
class LossReport:
    frames: List[FrameLoss]
    summary: LossSummary

    def to_dict(self) -> Dict:
        return {
            "summary": asdict(self.summary),
            "frames": [asdict(f) for f in self.frames],
        }

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
                except Exception as e:
                    if debug:
                        print(f"[FK-ERR] channel read failed at '{node.name}': {e}")

            try:
                rotated = parent_rot @ local_pos
                gpos = parent_pos + rotated
                grot = parent_rot @ local_rot
            except Exception as e:
                if debug:
                    print(f"[FK-ERR] transform failed at '{node.name}': {e}")
                gpos = parent_pos.copy()
                grot = parent_rot.copy()

            norm_name = normalize_name(node.name)
            for mesh_joint_name, bvh_node in mesh_joint_to_bvh.items():
                if bvh_node == node:
                    T = build_transform(grot, gpos)
                    joint_transforms[mesh_joint_name] = T

            for c in node.children:
                stack.append((c, gpos, grot))

        if debug:
            print(f"[FK] Frame {frame_idx} done: transforms={len(joint_transforms)}, steps={steps}")

        return joint_transforms

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

def get_bounding_boxes(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if vertices.ndim == 2:
        vertices = vertices[None, :, :]
    
    min_vals = np.min(vertices, axis=1)
    max_vals = np.max(vertices, axis=1)
    
    boxes_min = min_vals[:, None, :]
    boxes_max = max_vals[:, None, :]
    
    boxes = np.stack([boxes_min, boxes_max], axis=1)
    return boxes

def normalize_vertices(vertices: np.ndarray, scale_factor: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if vertices.ndim == 2:
        vertices = vertices[None, :, :]
    
    boxes = get_bounding_boxes(vertices)
    boxes_center = boxes.mean(axis=1)[:, None, :]
    boxes_scale = ((1 + scale_factor) * 0.5 * (boxes[:, 1] - boxes[:, 0]).max(axis=-1)[:, None, None])
    
    vertices_centered = vertices - boxes_center
    vertices_centered_scaled = vertices_centered / boxes_scale
    
    return vertices_centered_scaled, boxes_center, boxes_scale

def compute_signed_distance_simple(points: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> np.ndarray:
    if not TRIMESH_AVAILABLE:
        mesh_tree = cKDTree(mesh_vertices)
        distances, _ = mesh_tree.query(points, k=1)
        return distances.astype(np.float32)
    
    try:
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        signed_distances = mesh.nearest.signed_distance(points)
        return signed_distances.astype(np.float32)
    except:
        mesh_tree = cKDTree(mesh_vertices)
        distances, _ = mesh_tree.query(points, k=1)
        return distances.astype(np.float32)

def compute_sdf_grid_simple(vertices: np.ndarray, faces: np.ndarray, grid_size: int = 32) -> np.ndarray:
    if not TRIMESH_AVAILABLE:
        return None
    
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        bounds_min, bounds_max = mesh.bounds
        
        x = np.linspace(bounds_min[0], bounds_max[0], grid_size)
        y = np.linspace(bounds_min[1], bounds_max[1], grid_size)
        z = np.linspace(bounds_min[2], bounds_max[2], grid_size)
        
        grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        
        sdf_values = mesh.nearest.signed_distance(grid_points)
        sdf_grid = sdf_values.reshape(grid_size, grid_size, grid_size)
        
        return sdf_grid
    except:
        return None

def sample_sdf_grid(points: np.ndarray, sdf_grid: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    if sdf_grid is None:
        return np.zeros(len(points), dtype=np.float32)
    
    grid_size = sdf_grid.shape[0]
    
    points_normalized = (points - bounds_min) / (bounds_max - bounds_min + 1e-8)
    points_normalized = np.clip(points_normalized, 0, 1)
    
    indices = (points_normalized * (grid_size - 1)).astype(np.int32)
    indices = np.clip(indices, 0, grid_size - 1)
    
    sdf_values = sdf_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
    return sdf_values.astype(np.float32)

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
        
        rest_leg_vertices = self.rest_vertices[~np.isin(np.arange(len(self.rest_vertices)), np.concatenate([self.body_indices, self.arm_indices]))]
        if len(rest_leg_vertices) > 0:
            leg_indices_all = np.setdiff1d(np.arange(len(self.rest_vertices)), np.concatenate([self.body_indices, self.arm_indices]))
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

        rest_vertices_all = np.arange(len(self.rest_vertices))
        head_and_body = np.setdiff1d(rest_vertices_all, np.concatenate([
            self.limb_vertex_indices.get('left_arm', []),
            self.limb_vertex_indices.get('right_arm', []),
            self.limb_vertex_indices.get('left_leg', []),
            self.limb_vertex_indices.get('right_leg', [])
        ]))
        
        head_joint_idx = None
        for i, name in enumerate(self.joint_names):
            if name == "Head":
                head_joint_idx = i
                break
        
        if head_joint_idx is not None:
            head_center = self.rest_skeleton[head_joint_idx]
            head_distances = np.linalg.norm(self.rest_vertices[head_and_body] - head_center, axis=1)
            head_threshold = np.percentile(head_distances, 30)
            head_mask = head_distances < head_threshold
            self.head_vertex_indices = head_and_body[head_mask].tolist()
            self.body_only_indices = head_and_body[~head_mask].tolist()
        else:
            self.head_vertex_indices = []
            self.body_only_indices = head_and_body.tolist()
        
        self.arm_end_joints = ["LeftHand", "RightHand"]
        self.arm_end_vertex_indices = []
        for limb_name in ['left_arm', 'right_arm']:
            if limb_name in self.limb_vertex_indices:
                limb_verts = self.rest_vertices[self.limb_vertex_indices[limb_name]]
                if len(limb_verts) > 0:
                    if limb_name == 'left_arm':
                        end_mask = limb_verts[:, 0] < np.percentile(limb_verts[:, 0], 20)
                    else:
                        end_mask = limb_verts[:, 0] > np.percentile(limb_verts[:, 0], 80)
                    end_indices = np.array(self.limb_vertex_indices[limb_name])[end_mask].tolist()
                    self.arm_end_vertex_indices.extend(end_indices)

        print(f"  [Eval] Body vertices: {len(self.body_only_indices)}개")
        print(f"  [Eval] Head vertices: {len(self.head_vertex_indices)}개")
        print(f"  [Eval] Torso (body+head) vertices: {len(self.torso_vertex_indices)}개")
        for limb_name, indices in self.limb_vertex_indices.items():
            print(f"  [Eval] {limb_name}: {len(indices)}개 vertices")

    def _compute_rep_loss_for_limb(
        self, 
        limb_vertices_scaled: np.ndarray,
        body_vertices_scaled: np.ndarray,
        head_vertices_scaled: np.ndarray,
        ifth: bool = False,
        threshold: float = 3.0
    ) -> float:
        if len(limb_vertices_scaled) == 0:
            return 0.0
        
        if not TRIMESH_AVAILABLE:
            body_tree = cKDTree(body_vertices_scaled)
            head_tree = cKDTree(head_vertices_scaled)
            body_dists, _ = body_tree.query(limb_vertices_scaled, k=1)
            head_dists, _ = head_tree.query(limb_vertices_scaled, k=1)
            min_dists = np.minimum(body_dists, head_dists)
            phi_vals = np.maximum(0, -min_dists) * 1000.0
            phi_val = np.mean(phi_vals)
        else:
            try:
                body_pc = trimesh.points.PointCloud(vertices=body_vertices_scaled)
                body_mesh = body_pc.convex_hull
                body_vertices_ch = np.array(body_mesh.vertices, dtype=np.float32)
                body_faces_ch = np.array(body_mesh.faces, dtype=np.int32)
                
                head_pc = trimesh.points.PointCloud(vertices=head_vertices_scaled)
                head_mesh = head_pc.convex_hull
                head_vertices_ch = np.array(head_mesh.vertices, dtype=np.float32)
                head_faces_ch = np.array(head_mesh.faces, dtype=np.int32)
                
                body_sdf_grid = compute_sdf_grid_simple(body_vertices_ch, body_faces_ch, grid_size=32)
                head_sdf_grid = compute_sdf_grid_simple(head_vertices_ch, head_faces_ch, grid_size=32)
                
                body_bounds_min = body_vertices_ch.min(axis=0)
                body_bounds_max = body_vertices_ch.max(axis=0)
                head_bounds_min = head_vertices_ch.min(axis=0)
                head_bounds_max = head_vertices_ch.max(axis=0)
                
                body_sdf_vals = sample_sdf_grid(limb_vertices_scaled, body_sdf_grid, body_bounds_min, body_bounds_max)
                head_sdf_vals = sample_sdf_grid(limb_vertices_scaled, head_sdf_grid, head_bounds_min, head_bounds_max)
                
                total_sdf = body_sdf_vals + head_sdf_vals
                phi_vals = np.maximum(0, -total_sdf) * 1000.0
                phi_val = np.mean(phi_vals)
            except:
                body_tree = cKDTree(body_vertices_scaled)
                head_tree = cKDTree(head_vertices_scaled)
                body_dists, _ = body_tree.query(limb_vertices_scaled, k=1)
                head_dists, _ = head_tree.query(limb_vertices_scaled, k=1)
                min_dists = np.minimum(body_dists, head_dists)
                phi_vals = np.maximum(0, -min_dists) * 1000.0
                phi_val = np.mean(phi_vals)
        
        if ifth and phi_val < threshold:
            return 0.0
        
        return float(phi_val)
    
    def _compute_att_loss_for_limb(
        self,
        arm_end_vertices_scaled: np.ndarray,
        body_vertices_scaled: np.ndarray
    ) -> float:
        if len(arm_end_vertices_scaled) == 0:
            return 0.0
        
        if not TRIMESH_AVAILABLE:
            body_tree = cKDTree(body_vertices_scaled)
            dists, _ = body_tree.query(arm_end_vertices_scaled, k=1)
            phi_vals = np.maximum(0, dists) * 1000.0
            phi_val = np.mean(phi_vals)
        else:
            try:
                body_pc = trimesh.points.PointCloud(vertices=body_vertices_scaled)
                body_mesh = body_pc.convex_hull
                body_vertices_ch = np.array(body_mesh.vertices, dtype=np.float32)
                body_faces_ch = np.array(body_mesh.faces, dtype=np.int32)
                
                body_sdf_grid = compute_sdf_grid_simple(body_vertices_ch, body_faces_ch, grid_size=32)
                body_bounds_min = body_vertices_ch.min(axis=0)
                body_bounds_max = body_vertices_ch.max(axis=0)
                
                body_sdf_vals = sample_sdf_grid(arm_end_vertices_scaled, body_sdf_grid, body_bounds_min, body_bounds_max)
                phi_vals = np.maximum(0, body_sdf_vals) * 1000.0
                phi_val = np.mean(phi_vals)
            except:
                body_tree = cKDTree(body_vertices_scaled)
                dists, _ = body_tree.query(arm_end_vertices_scaled, k=1)
                phi_vals = np.maximum(0, dists) * 1000.0
                phi_val = np.mean(phi_vals)
        
        return float(phi_val)

    def _compute_distances_to_torso(self, points: np.ndarray, torso_vertices: np.ndarray) -> np.ndarray:
        if len(points) == 0 or len(torso_vertices) == 0:
            return np.array([], dtype=np.float32)
        
        torso_tree = cKDTree(torso_vertices)
        distances, _ = torso_tree.query(points, k=1)
        
        if isinstance(distances, np.ndarray):
            return distances.astype(np.float32)
        else:
            return np.array([distances], dtype=np.float32)
    
    def _compute_lrep(self, distances: np.ndarray, thresh_l: float = 0.05) -> np.ndarray:
        psi_r = np.clip(thresh_l - distances, a_min=0.0, a_max=None) ** 2
        return psi_r
    
    def _compute_latt(self, distances: np.ndarray, thresh_h: float = 0.2) -> np.ndarray:
        psi_a = np.clip(distances - thresh_h, a_min=0.0, a_max=None) ** 2
        return psi_a
    
    def _points_penetrate_torso(self, points: np.ndarray, torso_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(points) == 0:
            return np.array([], dtype=bool), np.array([], dtype=np.float32)
        
        distances = self._compute_distances_to_torso(points, torso_vertices)
        lrep_values = self._compute_lrep(distances, thresh_l=self.penetration_margin)
        
        penetrating_mask = lrep_values > 1e-6
        
        return penetrating_mask, lrep_values

    def evaluate(self) -> LossReport:
        frame_results: List[FrameLoss] = []
        
        lrep_accum = {name: 0.0 for name in self.limb_vertex_indices}
        latt_accum = 0.0
        lrep_total_accum = 0.0
        latt_total_accum = 0.0

        print(f"\n  평가 중: ", end="", flush=True)
        step = max(1, self.bvh.num_frames // 20)

        for frame_idx in range(self.bvh.num_frames):
            if frame_idx % step == 0:
                print(f"{frame_idx}/{self.bvh.num_frames} ", end="", flush=True)

            try:
                joint_transforms = self.bvh.forward_kinematics_transforms(
                    frame_idx, self.joint_names, debug=(frame_idx == 0)
                )
                
                skinned_vertices = apply_skinning(
                    self.rest_vertices,
                    self.skinning_weights,
                    joint_transforms,
                    self.rest_skeleton,
                    self.joint_names
                )
                
                vertices_centered_scaled, _, _ = normalize_vertices(
                    skinned_vertices[np.newaxis, :, :], scale_factor=0.2
                )
                vertices_centered_scaled = vertices_centered_scaled[0]
                
                body_vertices_scaled = vertices_centered_scaled[self.body_only_indices] if len(self.body_only_indices) > 0 else np.array([], dtype=np.float32).reshape(0, 3)
                head_vertices_scaled = vertices_centered_scaled[self.head_vertex_indices] if len(self.head_vertex_indices) > 0 else np.array([], dtype=np.float32).reshape(0, 3)
                
                if len(body_vertices_scaled) < 3 and len(head_vertices_scaled) < 3:
                    frame_results.append(
                        FrameLoss(
                            frame_index=frame_idx,
                            lrep_total=0.0,
                            latt_total=0.0,
                            lrep_by_limb={},
                            latt_by_limb={},
                            limb_stats=[],
                        )
                    )
                    continue

                limb_stats = []
                lrep_by_limb = {}
                latt_by_limb = {}
                
                thresholds = {
                    'left_arm': 3.0,
                    'right_arm': 6.0,
                    'left_leg': 10.0,
                    'right_leg': 15.0,
                }

                for limb_name, limb_indices in self.limb_vertex_indices.items():
                    limb_vertices_scaled = vertices_centered_scaled[limb_indices]
                    
                    lrep_val = self._compute_rep_loss_for_limb(
                        limb_vertices_scaled,
                        body_vertices_scaled,
                        head_vertices_scaled,
                        ifth=False,
                        threshold=thresholds.get(limb_name, 3.0)
                    )
                    
                    lrep_by_limb[limb_name] = lrep_val
                    lrep_accum[limb_name] += lrep_val
                    lrep_total_accum += lrep_val
                    
                    limb_stats.append(
                        LimbLoss(
                            name=limb_name,
                            lrep=lrep_val,
                            latt=0.0,
                        )
                    )

                if len(self.arm_end_vertex_indices) > 0:
                    arm_end_vertices_scaled = vertices_centered_scaled[self.arm_end_vertex_indices]
                    latt_val = self._compute_att_loss_for_limb(
                        arm_end_vertices_scaled,
                        body_vertices_scaled
                    )
                    latt_by_limb['arm_end'] = latt_val
                    latt_accum += latt_val
                    latt_total_accum += latt_val
                else:
                    latt_val = 0.0

                frame_results.append(
                    FrameLoss(
                        frame_index=frame_idx,
                        lrep_total=sum(lrep_by_limb.values()),
                        latt_total=latt_val,
                        lrep_by_limb=lrep_by_limb,
                        latt_by_limb=latt_by_limb,
                        limb_stats=limb_stats,
                    )
                )
            except Exception as e:
                print(f"\n  Error at frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                frame_results.append(
                    FrameLoss(
                        frame_index=frame_idx,
                        lrep_total=0.0,
                        latt_total=0.0,
                        lrep_by_limb={},
                        latt_by_limb={},
                        limb_stats=[],
                    )
                )

        print(f"{self.bvh.num_frames}/{self.bvh.num_frames}")

        frame_count = len(frame_results)
        summary = LossSummary(
            mean_lrep=lrep_total_accum / frame_count if frame_count > 0 else 0.0,
            mean_latt=latt_total_accum / frame_count if frame_count > 0 else 0.0,
            total_lrep=lrep_total_accum,
            total_latt=latt_total_accum,
            mean_lrep_by_limb={
                name: lrep_accum[name] / frame_count if frame_count > 0 else 0.0
                for name in self.limb_vertex_indices
            },
            mean_latt_by_limb={},
        )
        return LossReport(frames=frame_results, summary=summary)

def evaluate_bvh_file(
    bvh_path: str,
    mesh_npz_path: str,
    torso_joints: Optional[List[str]] = None,
    limb_joints: Optional[Dict[str, List[str]]] = None,
    penetration_margin: float = 0.01,
) -> LossReport:
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

if __name__ == "__main__":
    import sys
    import json

    bvh_to_mesh = {
        "Steve_to_Claire_martelo 3.bvh": "Steve.npz",
        "Claire_gt_martelo 3.bvh": "Claire.npz",
    }

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
            print("  python self_penetration_eval.py <bvh_path> <mesh_npz_path> [penetration_margin]")
            print("  python self_penetration_eval.py  (현재 디렉토리의 BVH 파일 자동 평가)")
            print("\n예시:")
            print("  python self_penetration_eval.py 'Steve_to_Claire_martelo 3.bvh' 'Steve.npz'")
            print("  python self_penetration_eval.py 'Claire_gt_martelo 3.bvh' 'Claire.npz' 0.01")
            sys.exit(1)
        
        bvh_files = existing_files
        penetration_margin = 1.0
    else:
        bvh_path = sys.argv[1]
        mesh_npz_path = sys.argv[2] if len(sys.argv) > 2 else None
        penetration_margin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
        
        if not Path(bvh_path).exists():
            print(f"오류: BVH 파일을 찾을 수 없습니다: {bvh_path}")
            sys.exit(1)
        
        if mesh_npz_path and not Path(mesh_npz_path).exists():
            print(f"오류: Mesh 파일을 찾을 수 없습니다: {mesh_npz_path}")
            sys.exit(1)
        
        bvh_files = [bvh_path]
        if mesh_npz_path:
            bvh_to_mesh[bvh_path] = mesh_npz_path

    print("=" * 100)
    print("【BVH Lrep/Latt Loss 평가 - R2ET 기반】")
    print("=" * 100)
    
    results = {}
    
    for bvh_path in bvh_files:
        try:
            print(f"\n▶ 처리 중: {bvh_path}")
            start_time = time.time()
            
            mesh_path = bvh_to_mesh.get(bvh_path)
            if not mesh_path or not Path(mesh_path).exists():
                print(f"  ⚠ Mesh 파일을 찾을 수 없습니다: {mesh_path}")
                print(f"  사용 가능한 mesh 파일: {list(bvh_to_mesh.values())}")
                continue
            
            print(f"  Mesh 파일: {mesh_path}")
            
            bvh = BVH(bvh_path)
            print(f"  nodes total: {len(bvh.nodes)}, frames: {bvh.num_frames}, frame_time: {bvh.frame_time:.4f}")
            
            report = evaluate_bvh_file(
                bvh_path,
                mesh_path,
                penetration_margin=penetration_margin
            )
            
            elapsed = time.time() - start_time
            result = report.to_dict()
            results[bvh_path] = result
            
            summary = result["summary"]
            print(f"\n  평균 Lrep: {summary['mean_lrep']:.6f}")
            print(f"  평균 Latt: {summary['mean_latt']:.6f}")
            print(f"  총 Lrep: {summary['total_lrep']:.6f}")
            print(f"  총 Latt: {summary['total_latt']:.6f}")
            print(f"  프레임 수: {len(result['frames'])}")
            print(f"  소요 시간: {elapsed:.2f}초")
            
            if summary['mean_lrep_by_limb']:
                print("  팔다리별 평균 Lrep:")
                for limb, lrep in summary['mean_lrep_by_limb'].items():
                    print(f"    {limb}: {lrep:.6f}")
                    
        except Exception as e:
            print(f"  ✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) > 1:
        print("\n" + "-" * 100)
        print(f"{'파일명':<50} {'평균 Lrep':>15} {'평균 Latt':>15} {'프레임 수':>10}")
        print("-" * 100)
        for path, result in results.items():
            summary = result["summary"]
            filename = Path(path).name
            print(f"{filename:<50} {summary['mean_lrep']:>15.6f} {summary['mean_latt']:>15.6f} {len(result['frames']):>10d}")
        print("-" * 100)
    
    print("\n평가 완료!")
    
    if len(results) == 1:
        print("\n상세 결과:")
        print(json.dumps(list(results.values())[0], indent=2, ensure_ascii=False))
