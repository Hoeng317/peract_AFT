
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce

from helpers.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, Conv3DUpsampleBlock

# AFTSimple 클래스 정의
class AFTSimple(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.to_q = nn.Linear(dim, self.hidden_dim)
        self.to_k = nn.Linear(dim, self.hidden_dim)
        self.to_v = nn.Linear(dim, self.hidden_dim)
        self.project = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)

        # AFT 방식의 가중치 계산
        weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(Q)
        Yt = torch.mul(Q_sig, weights)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt

# PreNorm 클래스 정의
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if self.norm_context is not None:
            context = kwargs.get('context')
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)

# FeedForward 클래스 정의
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# Attention 클래스 정의
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

# PerceiverIO adapted for 6-DoF manipulation
class PerceiverVoxelLangEncoder(nn.Module):
    def __init__(
            self,
            depth,                    # number of self-attention layers
            iterations,               # number cross-attention iterations (PerceiverIO uses just 1)
            voxel_size,               # N voxels per side (size: N*N*N)
            initial_dim,              # 10 dimensions - dimension of the input sequence to be encoded
            low_dim_size,             # 4 dimensions - proprioception: {gripper_open, left_finger, right_finger, timestep}
            layer=0,
            num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
            num_grip_classes=2,       # open or not open
            num_collision_classes=2,  # collisions allowed or not allowed
            input_axis=3,             # 3D tensors have 3 axes
            num_latents=512,          # number of latent vectors
            im_channels=64,           # intermediate channel size
            latent_dim=512,           # dimensions of latent vectors
            cross_heads=1,            # number of cross-attention heads
            latent_heads=8,           # number of latent heads
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
            weight_tie_layers=False,
            pos_encoding_with_lang=True,
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            lang_fusion_type='seq',
            voxel_patch_size=9,
            voxel_patch_stride=8,
            no_skip_connection=False,
            no_perceiver=False,
            no_language=False,
            final_dim=64,
    ):
        super().__init__()
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.pos_encoding_with_lang = pos_encoding_with_lang
        self.lang_fusion_type = lang_fusion_type
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.no_skip_connection = no_skip_connection
        self.no_perceiver = no_perceiver
        self.no_language = no_language

        # patchified input dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # 64 voxel features + 64 proprio features (+ 64 lang goal features if concattenated)
        self.input_dim_before_seq = self.im_channels * 3 if self.lang_fusion_type == 'concat' else self.im_channels * 2

        # CLIP language feature dimensions
        lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, 512, 77

        # learnable positional encoding
        if self.pos_encoding_with_lang:
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         lang_max_seq_len + spatial_size ** 3,
                                                         self.input_dim_before_seq))
        else:
            # assert self.lang_fusion_type == 'concat', 'Only concat is supported for pos encoding without lang.'
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         spatial_size, spatial_size, spatial_size,
                                                         self.input_dim_before_seq))

        # voxel input preprocessing 1x1 conv encoder
        self.input_preprocess = Conv3DBlock(
            self.init_dim, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels, self.im_channels,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation)

        # language preprocess
        if self.lang_fusion_type == 'concat':
            self.lang_preprocess = nn.Linear(lang_feat_dim, self.im_channels)
        elif self.lang_fusion_type == 'seq':
            self.lang_preprocess = nn.Linear(lang_emb_dim, self.im_channels * 2)

        # proprioception
        if self.low_dim_size > 0:
            self.proprio_preprocess = DenseBlock(
                self.low_dim_size, self.im_channels, norm=None, activation=activation,
            )

        # pooling functions
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels)
        flat_size = self.im_channels * 4

        # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # cross attention blocks (remains unchanged)
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim,
                                          self.input_dim_before_seq,
                                          heads=cross_heads,
                                          dim_head=cross_dim_head,
                                          dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        # 기존 self-attention을 AFT로 교체
        self.self_attention = AFTSimple(max_seqlen=1000, dim=latent_dim)  # max_seqlen은 임의로 설정 가능

        # self attention layers
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                self.self_attention,
                PreNorm(latent_dim, FeedForward(latent_dim))
            ]))

        # decoder cross attention (remains unchanged)
        self.decoder_cross_attn = PreNorm(self.input_dim_before_seq, Attention(self.input_dim_before_seq,
                                                                               latent_dim,
                                                                               heads=cross_heads,
                                                                               dim_head=cross_dim_head,
                                                                               dropout=decoder_dropout),
                                          context_dim=latent_dim)

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, self.final_dim,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation,
        )

        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self.input_dim_before_seq)

        flat_size += self.input_dim_before_seq * 4

        # final 3D softmax
        self.final = Conv3DBlock(
            self.im_channels if (self.no_perceiver or self.no_skip_connection) else self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1, norm=None, activation=activation)

        self.trans_decoder = Conv3DBlock(
            self.final_dim, 1, kernel_sizes=3, strides=1,
            norm=None, activation=None,
        )

        # rotation, gripper, and collision MLP layers
        if self.num_rotation_classes > 0:
            self.ss_final = SpatialSoftmax3D(
                self.voxel_size, self.voxel_size, self.voxel_size,
                self.im_channels)

            flat_size += self.im_channels * 4

            self.dense0 =  DenseBlock(flat_size, 256, None, activation)
            self.dense1 = DenseBlock(256, self.final_dim, None, activation)

            self.rot_grip_collision_ff = DenseBlock(self.final_dim,
                                                    self.num_rotation_classes * 3 + \
                                                    self.num_grip_classes + \
                                                    self.num_collision_classes,
                                                    None, None)

    def encode_text(self, x):
        with torch.no_grad():
            text_feat, text_emb = self._clip_rn50.encode_text_with_embeddings(x)

        text_feat = text_feat.detach()
        text_emb = text_emb.detach()
        text_mask = torch.where(x==0, x, 1)  # [1, max_token_len]
        return text_feat, text_emb

    def forward(
            self,
            ins,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            prev_layer_voxel_grid,
            bounds,
            prev_layer_bounds,
            mask=None,
    ):
        # preprocess input
        d0 = self.input_preprocess(ins)                       # [B,10,100,100,100] -> [B,64,100,100,100]

        # aggregated features from 1st softmax and maxpool for MLP decoders
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        # patchify input (5x5x5 patches)
        ins = self.patchify(d0)                               # [B,64,100,100,100] -> [B,64,20,20,20]

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert len(axis) == self.input_axis, 'input must have the same number of axis as input_axis'

        # concat proprio
        if self.low_dim_size > 0:
            p = self.proprio_preprocess(proprio)              # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p], dim=1)                  # [B,128,20,20,20]

        # language ablation
        if self.no_language:
            lang_goal_emb = torch.zeros_like(lang_goal_emb)
            lang_token_embs = torch.zeros_like(lang_token_embs)

        # option 1: tile and concat lang goal to input
        if self.lang_fusion_type == 'concat':
            lang_emb = lang_goal_emb
            lang_emb = lang_emb.to(dtype=ins.dtype)
            l = self.lang_preprocess(lang_emb)
            l = l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, l], dim=1)

        # channel last
        ins = rearrange(ins, 'b d ... -> b ... d')            # [B,20,20,20,128]

        # add pos encoding to grid
        if not self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        # concat to channels of and flatten axis
        queries_orig_shape = ins.shape

        # rearrange input to be channel last
        ins = rearrange(ins, 'b ... d -> b (...) d')          # [B,8000,128]
        ins_wo_prev_layers = ins

        # option 2: add lang token embs as a sequence
        if self.lang_fusion_type == 'seq':
            l = self.lang_preprocess(lang_token_embs)         # [B,77,1024] -> [B,77,128]
            ins = torch.cat((l, ins), dim=1)                  # [B,8077,128]

        # add pos encoding to language + flattened grid (the recommended way)
        if self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=ins, mask=mask) + x
            x = cross_ff(x) + x

            # self-attention layers (AFT 적용)
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(ins, context=x)

        # crop out the language part of the output sequence
        if self.lang_fusion_type == 'seq':
            latents = latents[:, l.shape[1]:]

        # reshape back to voxel grid
        latents = latents.view(b, *queries_orig_shape[1:-1], latents.shape[-1]) # [B,20,20,20,64]
        latents = rearrange(latents, 'b ... d -> b d ...')                      # [B,64,20,20,20]

        # aggregated features from 2nd softmax and maxpool for MLP decoders
        feats.extend([self.ss1(latents.contiguous()), self.global_maxp(latents).view(b, -1)])

        # upsample
        u0 = self.up0(latents)

        # ablations
        if self.no_skip_connection:
            u = self.final(u0)
        elif self.no_perceiver:
            u = self.final(d0)
        else:
            u = self.final(torch.cat([d0, u0], dim=1))

        # translation decoder
        trans = self.trans_decoder(u)

        # rotation, gripper, and collision MLPs
        rot_and_grip_out = None
        if self.num_rotation_classes > 0:
            feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(b, -1)])

            dense0 = self.dense0(torch.cat(feats, dim=1))
            dense1 = self.dense1(dense0)                     # [B,72*3+2+2]

            rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
            rot_and_grip_out = rot_and_grip_collision_out[:, :-self.num_collision_classes]
            collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes:]

        return trans, rot_and_grip_out, collision_out
