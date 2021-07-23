# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
# Written by Pengchuan Zhang, penzhan@microsoft.com
from torch import nn
import torch
from timm.models.layers import trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 rpe=False, wx=14, wy=14, nglo=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Inspired by swin transformer:
        # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L88-L103
        # define parameter tables for local and global relative position bias
        self.rpe = rpe
        if rpe:
            self.wx = wx
            self.wy = wy
            self.nglo = nglo
            self.local_relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * wx - 1) * (2 * wy - 1),
                            num_heads))  # (2*wx-1, 2*wy-1, nH)
            trunc_normal_(self.local_relative_position_bias_table, std=.02)
            if nglo >= 1:
                self.g2l_relative_position_bias = nn.Parameter(
                    torch.zeros(2, num_heads, nglo))  # (2, nH, nglo)
                self.g2g_relative_position_bias = nn.Parameter(
                    torch.zeros(num_heads, nglo, nglo))  # (nH, nglo, nglo)
                trunc_normal_(self.g2l_relative_position_bias, std=.02)
                trunc_normal_(self.g2g_relative_position_bias, std=.02)

            # get pair-wise relative position index
            coords_h = torch.arange(wx)
            coords_w = torch.arange(wy)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wx, wy
            coords_flatten = torch.flatten(coords, 1)  # 2, Wx*Wy
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wx*Wy, Wx*Wy
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wx*Wy, Wx*Wy, 2
            relative_coords[:, :, 0] += wx - 1  # shift to start from 0
            relative_coords[:, :, 1] += wy - 1
            relative_coords[:, :, 0] *= 2 * wy - 1
            relative_position_index = relative_coords.sum(-1)  # Wx*Wy, Wx*Wy
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rpe:
            assert N == self.nglo + self.wx*self.wy, "For relative position, N != self.nglo + self.wx*self.wy!"
            local_relative_position_bias = self.local_relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.wx*self.wy, self.wx*self.wy, -1)  # Wh*Ww, Wh*Ww,nH
            relative_position_bias = local_relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            if self.nglo > 0:
                # relative position embedding of global tokens
                global_relative_position_bias = torch.cat([
                    self.g2g_relative_position_bias,
                    self.g2l_relative_position_bias[0].unsqueeze(-1).expand(-1, -1, self.wx*self.wy)
                ], dim=-1)  # nH, nglo, N
                # relative position embedding of local tokens
                local_relative_position_bias = torch.cat([
                    self.g2l_relative_position_bias[1].unsqueeze(1).expand(-1, self.wx*self.wy, -1),
                    relative_position_bias,
                ], dim=-1)  # nH, Wh*Ww, N
                relative_position_bias = torch.cat([
                    global_relative_position_bias,
                    local_relative_position_bias,
                ], dim=1)  # nH, N, N
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        S = T
        macs = 0
        n_params = 0

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T * S * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T * C * S

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs / 1e8)

        # self attention: T should be equal to S
        assert T == S
        qkv_params = sum([p.numel() for p in module.qkv.parameters()])
        n_params += qkv_params
        # multiply by Seq length
        macs += qkv_params * T
        # print('macs qkv', qkv_params * T / 1e8)

        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T)
        # print('macs proj', proj_params * T / 1e8)

        module.__flops__ += macs
        # return n_params, macs