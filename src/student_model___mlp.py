import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_quat(q, eps=1e-8):
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / norm


class DistilRET(nn.Module):
    """
    Light distilled R2ET model without Transformer blocks.
    """

    def __init__(
        self,
        num_joint: int = 22,
        skel_feat_dim: int = 32,
        quat_feat_dim: int = 64,
        shape_feat_dim: int = 32,
        latent_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_joint = num_joint

        self.skel_encoder = nn.Sequential(
            nn.Linear(3, skel_feat_dim),
            nn.ReLU(),
            nn.Linear(skel_feat_dim, skel_feat_dim),
            nn.ReLU(),
        )

        self.quat_encoder = nn.Sequential(
            nn.Linear(4, quat_feat_dim),
            nn.ReLU(),
            nn.Linear(quat_feat_dim, quat_feat_dim),
            nn.ReLU(),
        )

        self.shape_encoder = nn.Sequential(
            nn.Linear(3, shape_feat_dim),
            nn.ReLU(),
            nn.Linear(shape_feat_dim, shape_feat_dim),
            nn.ReLU(),
        )

        stage1_in_dim = quat_feat_dim + skel_feat_dim

        self.stage1_mlp = nn.Sequential(
            nn.Linear(stage1_in_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        self.stage1_head = nn.Linear(latent_dim, 4)

        stage2_in_dim = latent_dim + shape_feat_dim

        self.stage2_mlp = nn.Sequential(
            nn.Linear(stage2_in_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        self.stage2_head = nn.Linear(latent_dim, 4)

    def forward(self, quat_src, skel_src, shape_src):
        B, T, J, _ = quat_src.shape
        assert J == self.num_joint

        skel_feat = self.skel_encoder(skel_src)
        skel_feat_time = skel_feat.unsqueeze(1).expand(B, T, J, -1)

        quat_feat = self.quat_encoder(quat_src)

        stage1_in = torch.cat([quat_feat, skel_feat_time], dim=-1)
        h1 = self.stage1_mlp(stage1_in)
        delta_qs = normalize_quat(self.stage1_head(h1))

        shape_feat = self.shape_encoder(shape_src)
        shape_feat_time = shape_feat.unsqueeze(1).expand(B, T, J, -1)

        stage2_in = torch.cat([h1, shape_feat_time], dim=-1)
        h2 = self.stage2_mlp(stage2_in)
        delta_qg = normalize_quat(self.stage2_head(h2))

        return delta_qs, delta_qg
