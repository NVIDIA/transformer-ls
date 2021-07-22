"""
Adapted from https://github.com/mlpen/Nystromformer
Add dynamic convolution which is not included in Linformer.
"""

import torch
import torch.nn as nn
import math
import pdb

class LinformerAttention(nn.Module):
    projection_matrix = None

    def __init__(self, config):
        super().__init__()

        self.num_head = config.num_head
        self.head_dim = config.head_dim
        self.linformer_k = config.linformer_k
        self.seq_len = config.max_seq_len

        self.n_sparse = getattr(config, "n_sparse", 0)

        self.dynamic_conv = getattr(config, "dynamic_conv", False)
        if not self.dynamic_conv:
            if LinformerAttention.projection_matrix is not None:
                self.E = LinformerAttention.projection_matrix
            else:
                LinformerAttention.projection_matrix = nn.Parameter(torch.Tensor(self.num_head, self.linformer_k, self.seq_len))
                torch.nn.init.normal_(LinformerAttention.projection_matrix, std = 0.02)
                self.E = LinformerAttention.projection_matrix

        self.use_conv = config.conv_kernel_size > 0
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.num_head, out_channels=self.num_head,
                kernel_size=(config.conv_kernel_size, 1), padding=(config.conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_head)

    def forward(self, Q, K, V, mask, dconv_weight=None):
        if self.dynamic_conv:
            # V: bsize, self.num_head, self.seq_len, self.head_dim
            # dconv_weight: bsize x num_head x k x seqlen
            E = dconv_weight
        else:
            E = self.E

        V_orig = V
        if self.n_sparse > 0:
            # sample the sparse tokens to attend to
            sample_probs = mask / torch.sum(mask, dim=1, keepdim=True)
            sample_probs = sample_probs.unsqueeze(1).expand(-1, self.num_head, -1).reshape(-1, self.seq_len)
            sample_idxes = torch.multinomial(sample_probs, self.n_sparse, replacement=False).reshape(
                -1, self.num_head, self.n_sparse)
            sparse_mask = torch.zeros((Q.shape[0], self.num_head, self.seq_len), dtype=torch.bool).to(Q.device)
            # sparse_mask: bsize x self.num_head x seqlen
            sparse_mask.scatter_(2, sample_idxes, True)
            K_samples = K.masked_select(sparse_mask.unsqueeze(-1)).reshape(
                -1, self.num_head, self.n_sparse, self.head_dim)
            V_samples = V.masked_select(sparse_mask.unsqueeze(-1)).reshape(
                -1, self.num_head, self.n_sparse, self.head_dim)

            K = torch.cat([E.matmul(K * mask[:, None, :, None]), K_samples], dim=-2)
            V = torch.cat([E.matmul(V * mask[:, None, :, None]), V_samples], dim=-2)

        else:
            K = torch.matmul(E, K * mask[:, None, :, None])
            V = torch.matmul(E, V * mask[:, None, :, None])

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        attn = nn.functional.softmax(dot, dim = -1)

        X = torch.matmul(attn, V)
        if self.use_conv:
            X += self.conv(V_orig * mask[:, None, :, None])

        return X

    def extra_repr(self):
        return f'linformer_k={self.linformer_k}'


