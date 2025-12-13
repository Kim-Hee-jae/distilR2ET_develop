import torch
import torch.nn as nn
import math


from src.ops import qlinear


# Blocks
class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, n, 3, h, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0, :, :, :, :], qkv[1, :, :, :, :], qkv[2, :, :, :, :]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)

        out = self.nn1(out)
        out = self.do1(out)

        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Residual(
                                Attention(dim, mlp_dim, heads=heads, dropout=dropout)
                            ),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )
        else:
            assert depth == 1
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Attention(dim, mlp_dim, heads=heads, dropout=dropout),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=22):
        if dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(dim)
            )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, n, self.dim)``
        """

        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, 0 : emb.size(1), :]
        emb = self.dropout(emb)
        return emb


# Encoder
class QuatEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, hidden_channels, kp):
        super(QuatEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(4, token_channels)
        self.trans1 = Transformer(token_channels, 1, 4, hidden_channels, 1 - kp) # dim, depth, heads, mlp_dim, dropout

    def forward(self, pose_t):
        # pose_t: bs joint, 4
        token_q = self.token_linear(pose_t)
        embed_q = self.trans1(token_q)

        return embed_q


class SkelEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super(SkelEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(3, token_channels)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, skel):
        # bs = skel.shape[0]
        token_s = self.token_linear(skel)
        embed_s = self.trans1(token_s)

        return embed_s


class ShapeEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super(ShapeEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(3, token_channels)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, shape):
        token_s = self.token_linear(shape)
        embed_s = self.trans1(token_s)

        return embed_s
 

# Decoder
# delta qs    
class DeltaSkelDecoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, hidden_channels, kp):
        super(DeltaSkelDecoder, self).__init__()

        self.num_joint = num_joint
        self.q_encoder = QuatEncoder(num_joint, token_channels, hidden_channels, kp)
        self.skel_encoder = SkelEncoder(num_joint, token_channels, embed_channels, kp)
        self.pos_encoder = PositionalEncoding(
            1 - kp, hidden_channels + embed_channels
        )

        self.embed_linear = nn.Linear(
            hidden_channels + embed_channels, embed_channels
        )
        self.embed_acti = nn.ReLU()
        self.embed_drop = nn.Dropout(1 - kp)
        self.delta_linear = nn.Linear(embed_channels, 4)

    def forward(self, q_t, skelA):
        q_embed = self.q_encoder(q_t)
        skelA_embed = self.skel_encoder(skelA)

        cat_embed = torch.cat([q_embed, skelA_embed], dim=-1)  # bs n c
        pos_embed = self.pos_encoder(cat_embed)

        embed = self.embed_drop(self.embed_acti(self.embed_linear(pos_embed)))
        deltaq_t = self.delta_linear(embed)

        return deltaq_t


# delta qg
class DeltaShapeDecoder(nn.Module):
    def __init__(self, num_joint, hidden_channels, kp):
        super(DeltaShapeDecoder, self).__init__()

        self.num_joint = num_joint

        self.joint_linear1 = nn.Linear((3 + 4) * self.num_joint, hidden_channels)
        self.joint_acti1 = nn.ReLU()
        self.joint_drop1 = nn.Dropout(p=1 - kp)
        self.joint_linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.joint_acti2 = nn.ReLU()
        self.joint_drop2 = nn.Dropout(p=1 - kp)

        self.delta_linear = qlinear(hidden_channels, 4 * num_joint)

    def forward(self, shapeA, x): # x = deltaq_t
        bs = shapeA.shape[0]
        x_cat = torch.cat([shapeA, x], dim=-1)
        x_cat = x_cat.view((bs, -1))

        x_embed = self.joint_drop1(self.joint_acti1(self.joint_linear1(x_cat)))
        x_embed = self.joint_drop2(self.joint_acti2(self.joint_linear2(x_embed)))
        deltaq_t = self.delta_linear(x_embed)
        deltaq_t = torch.reshape(deltaq_t, [bs, self.num_joint, 4])

        return deltaq_t
    

# Model
class DistilledRetNet(nn.Module):
    def __init__(
        self,
        num_joint=22,
        token_channels=64,
        hidden_channels_p=256,
        embed_channels_p=128,
        kp=0.8,
    ):
        super(DistilledRetNet, self).__init__()
        self.num_joint = num_joint
        self.delta_qs_dec = DeltaSkelDecoder(
            num_joint, token_channels, embed_channels_p, hidden_channels_p, kp
        )
        self.delta_qg_dec = DeltaShapeDecoder(
            num_joint, hidden_channels_p, kp
        )

    def forward(
        self,
        quatA,
        skelA,
        shapeA
    ):
        '''
        quatA : bs T joints 4
        skelA : bs joints 3
        shapeA : bs joints 3
        '''

        bs, T = quatA.size(0), quatA.size(1)
  
        quatB = []

        """ mapping from A to B frame by frame"""
        for t in range(T):
            quatA_t = quatA[:, t, :, :]  # motion copy

            # delta qs
            delta_qs_t = self.delta_qs_dec(quatA_t, skelA)

            # delta qg
            delta_qg_t = self.delta_qg_dec(shapeA, delta_qs_t)
            quatB.append(delta_qg_t)
            

        # stack all frames
        quatB = torch.stack(quatB, dim=1)  # shape: (batch_size, T, 22, 4)

        return quatB