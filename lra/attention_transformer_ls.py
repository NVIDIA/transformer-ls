# Copyright (c) 2021 NVIDIA CORPORATION. Licensed under the MIT license.
# Written by Chen Zhu during an internship at NVIDIA, zhuchen.eric@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionLS(nn.Module):
    """The long-short term attention for bidirectional language modelling
    """

    def __init__(self, config):
        super().__init__()

        assert not (config.pooling_mode.lower() == 'cls' and config.cls_token)
        self.cls_from_seq = config.pooling_mode.lower() == 'cls'

        self.num_head = config.num_head
        self.head_dim = config.head_dim
        self.num_landmarks = config.num_landmarks
        self.seq_len = config.max_seq_len
        self.dim = config.transformer_dim

        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

        self.window_size = config.window_size

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_o = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.fp32 = config.fp32_attn

        self.dual_ln_s = nn.LayerNorm(self.num_head * self.head_dim)
        self.dual_ln_l = nn.LayerNorm(self.num_head * self.head_dim)

        self.dconv_fc = nn.Linear(self.dim, self.num_head * self.num_landmarks)

        self.use_conv = getattr(config, "conv_kernel_size", -1) > 0
        if self.use_conv:
            self.conv = nn.Conv1d(
                in_channels=self.num_head, out_channels=self.num_head,
                kernel_size=(config.conv_kernel_size, 1), padding=(config.conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_head)
            nn.init.zeros_(self.conv.weight)

    def get_tiles(self, x, transpose=False):
        # x: bsz x n_heads x seqlen x d_head
        bsz, n_heads, seqlen, d_h = x.shape
        out_shape = (bsz, n_heads, seqlen//self.window_size-1, 2 * self.window_size, d_h)
        in_strides = x.stride()
        out_strides = (in_strides[0], in_strides[1], in_strides[2]*self.window_size, in_strides[2], 1)

        x_main = x.as_strided(size=out_shape, stride=out_strides)
        x_last = x[:, :, None, -2*self.window_size:, :]
        x = torch.cat([x_main, x_last], dim=2)
        if transpose:
            return x.transpose(-1, -2)
        else:
            #  bsz x n_heads x seqlen//wlen x 2*wlen x d_h
            return x

    def get_tiled_mask(self, mask):
        bsz, seqlen = mask.shape
        out_shape = (bsz, seqlen//self.window_size-1, 2*self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1]*self.window_size, in_stride[1])
        mask_main = mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, :]
        mask_last = mask[:, None, None, -2*self.window_size:]

        return torch.cat([mask_main, mask_last], dim=2)[:, :, :, None, :]

    def sliding_chunks_matmul_qk(self, Q, K, padding_mask):
        # Q, K: bsz x num_heads x seqlen x d_head
        # padding_mask: bsz x seqlen
        bsz, num_heads, seqlen, d_h = Q.shape
        mask_tiles = self.get_tiled_mask(padding_mask)
        K_tiles = self.get_tiles(K, transpose=True)
        Q_tiles = Q.view(bsz, num_heads, seqlen//self.window_size, self.window_size, d_h)
        # bsz x num_heads x seqlen//winsize x winsize x 2winsize
        qk_scores = Q_tiles.matmul(K_tiles)
        qk_scores.masked_fill_(mask_tiles, float('-inf'))
        return qk_scores.view(bsz, num_heads, seqlen, 2*self.window_size)

    def get_tiles_v2(self, x, transpose=False):
        if self.window_size <= 0:
            return x

        bsz, n_heads, seqlen, d_h = x.shape
        n_groups = seqlen // self.window_size
        ext_len = max(self.window_size//2, 1)
        x = F.pad(x, (0, 0, ext_len, ext_len), value=0)
        strides = x.stride()
        if transpose:
            out_shape = (bsz, n_heads, n_groups, d_h, 2 * ext_len + self.window_size)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[3], strides[2])
        else:
            out_shape = (bsz, n_heads, n_groups, 2 * ext_len + self.window_size, d_h)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def get_tiled_mask_v2(self, mask):
        # only mask along the key dimension
        bsz, seqlen = mask.shape
        ext_len = max(self.window_size//2, 1)
        mask = F.pad(mask, (ext_len, ext_len), value=True)
        out_shape = (bsz, seqlen//self.window_size, 2*ext_len + self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1]*self.window_size, in_stride[1])
        return mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, None, :]

    def sliding_chunks_matmul_qk_v2(self, Q, K, padding_mask):
        bsz, num_heads, seqlen, d_h = Q.shape
        if self.window_size > 0:
            # Q, K: bsz x num_heads x seqlen x d_head
            # padding_mask: bsz x seqlen

            mask_tiles = self.get_tiled_mask_v2(padding_mask)
            K_tiles = self.get_tiles_v2(K, transpose=True)
            Q_tiles = Q.view(bsz, num_heads, seqlen//self.window_size, self.window_size, d_h)
            # bsz x num_heads x seqlen//winsize x winsize x 2winsize
            qk_scores = Q_tiles.matmul(K_tiles)
            qk_scores = qk_scores.masked_fill(mask_tiles, float('-inf'))
            return qk_scores.view(bsz, num_heads, seqlen, -1)
        else:
            qk_scores = torch.sum(Q*K, dim=-1, keepdim=True)
            return qk_scores

    def forward(self, X, mask, cls_embed=None):
        assert not (self.num_landmarks <= 0 and cls_embed is None and self.window_size <= 0)
        if self.cls_from_seq:
            cls_embed = X[:,:1].contiguous()
            X = X[:,1:].contiguous()
            mask = mask[:,1:].contiguous()

        bsz, seqlen, d_model = X.shape
        # bsz x n_head x length x head_dim
        Q = self.split_heads(self.W_q(X)).mul(1./math.sqrt(self.head_dim))

        K = self.split_heads(self.dual_ln_l(self.W_k(X)))
        V = self.split_heads(self.dual_ln_l(self.W_v(X)))
        if self.fp32:
            Q, K, V = Q.float(), K.float(), V.float()

        # bsz x length x num_head*num_lms
        padding_mask = ~mask.bool()

        K_compress = V_compress = None
        if self.num_landmarks > 0:
            head_scores = self.dconv_fc(X).masked_fill(padding_mask[:, :, None], float('-inf'))
            head_scores = F.softmax(head_scores, dim=1, dtype=torch.float32) #.to(X)
            if not self.fp32:
                head_scores = head_scores.to(X)
            # bsz x num_head x num_lms x length
            head_scores = head_scores.view(bsz, seqlen, self.num_head, self.num_landmarks).permute(0, 2, 3, 1)
            K_compress = head_scores.matmul(K)
            V_compress = head_scores.matmul(V)

        if cls_embed is not None:
            Q_cls = self.split_heads(self.W_q(cls_embed)).mul(1. / math.sqrt(self.head_dim))
            K_cls = self.split_heads(self.W_k(cls_embed))
            V_cls = self.split_heads(self.W_v(cls_embed))
            if self.num_landmarks > 0:
                K_compress = torch.cat([K_cls, K_compress], dim=2)
                V_compress = torch.cat([V_cls, V_compress], dim=2)
            else:
                K_compress = K_cls
                V_compress = V_cls

        if self.dual_ln_s is not None and K_compress is not None:
            K_compress = self.dual_ln_s(K_compress.transpose(1, 2).contiguous().view(bsz, -1, d_model))
            K_compress = self.split_heads(K_compress)
            V_compress = self.dual_ln_s(V_compress.transpose(1, 2).contiguous().view(bsz, -1, d_model))
            V_compress = self.split_heads(V_compress)

        if self.num_landmarks > 0 or (cls_embed is not None):
            # bsz x num_head x length x num_lms
            attn_compress = Q.matmul(K_compress.transpose(-1, -2))
        else:
            attn_compress = None

        if self.window_size > 0 or self.num_landmarks == 0:
            # First, compute the compressed part, or the attentions on the landmarks
            # First use window attention to attend to the diagonals
            # V: bsize, self.seq_len, self.num_head, self.head_dim
            # win_attn_weights = self.sliding_chunks_matmul_qk(Q, K, padding_mask)
            win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K, padding_mask)
        else:
            win_attn_weights = None

        if attn_compress is None:
            all_attn_ = win_attn_weights
        elif win_attn_weights is None:
            all_attn_ = attn_compress
        else:
            all_attn_ = torch.cat([attn_compress, win_attn_weights], dim=-1)

        all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)
        # If one of the rows are all -inf, then it will be NaN!
        all_attn = all_attn.masked_fill(padding_mask[:,None,:,None], 0)
        if not self.fp32:
            all_attn = all_attn.to(X)
        all_attn = self.drop_attn(all_attn)

        C = 0
        if attn_compress is not None:
            C += all_attn[:,:,:,:K_compress.shape[2]].matmul(V_compress)

        if win_attn_weights is not None:
            win_attn_probs = all_attn[:,:,:,-win_attn_weights.shape[-1]:]
            if self.window_size > 0:
                win_attn_probs = win_attn_probs.view(bsz, self.num_head, seqlen // self.window_size, self.window_size,-1)
                V_tiles = self.get_tiles_v2(V, transpose=False)
                C += win_attn_probs.matmul(V_tiles).view(bsz, self.num_head, seqlen, self.head_dim)
            else:
                C += win_attn_probs * V

        if cls_embed is not None:
            if self.fp32:
                Q_cls, K_cls, V_cls = Q_cls.float(), K_cls.float(), V_cls.float()
            # bsz x n_heads x 1 x (1+seqlen)
            cls_scores = torch.cat([Q_cls.matmul(K_cls.transpose(-1, -2)),
                                    Q_cls.matmul(C.transpose(-1, -2)).masked_fill(padding_mask[:,None,None,:], float('-inf'))],
                                   dim=-1)
            cls_probs = torch.softmax(cls_scores, dim=-1, dtype=torch.float32)#.to(X)
            if not self.fp32:
                cls_probs = cls_probs.to(X)
            out_cls_embed = V_cls * cls_probs[:,:,:,:1] + cls_probs[:,:,:,1:].matmul(C)

        if self.use_conv:
            V = V.masked_fill(padding_mask[:, None, :, None], 0)
            C = C + self.conv(V)

        if cls_embed is not None:
            C = torch.cat([out_cls_embed, C], dim=2)

        if self.fp32:
            # Finally convert it back, same as Nystromformer
            C = C.to(X)
        out = self.W_o(self.combine_heads(C))
        return out

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, window_size={self.window_size}'

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
