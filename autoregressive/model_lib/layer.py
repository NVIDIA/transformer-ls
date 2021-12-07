# Copyright (c) 2021 NVIDIA CORPORATION. Licensed under the MIT license.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from fairseq.modules.layer_norm import LayerNorm

import pdb


class ChunkedLSAttention(nn.Module):
    def __init__(self, d_model, n_head, chunk_size, chunk_rank, window_len, dropout,
                 grad_chk=False, use_bias=False, dp_attn=0,
                 probing=False):
        nn.Module.__init__(self)

        self.dropout = nn.Dropout(dropout)
        self.dp_attn = nn.Dropout(dp_attn)

        assert d_model % n_head == 0
        assert chunk_size > 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window_len = window_len

        self.chunk_rank = chunk_rank
        self.chunk_size = chunk_size
        self.n_head = n_head
        self.d_h = d_model // n_head
        self.d_model = d_model

        self.dconv_1 = nn.Linear(d_model, n_head * chunk_rank)

        self.r_net = nn.Linear(d_model, d_model, bias=False)
        self.r_net_chunk = nn.Linear(d_model, d_model)
        self.d_head = d_model // self.n_head
        # Positional bias as in Transformer-XL.
        self.r_r_bias = nn.Parameter(torch.FloatTensor(1, self.n_head, 1, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(1, self.n_head, 1, 1, self.d_head))

        self.grad_chk = grad_chk

        self.proj_query = nn.Linear(d_model, d_model, bias=use_bias)
        nn.init.xavier_normal_(self.proj_query.weight)
        self.proj_out = nn.Linear(d_model, d_model, bias=use_bias)
        nn.init.xavier_normal_(self.proj_out.weight)
        self.proj_val = nn.Linear(d_model, d_model, bias=use_bias)
        nn.init.xavier_normal_(self.proj_val.weight)
        self.proj_key = nn.Linear(d_model, d_model, bias=use_bias)
        nn.init.xavier_normal_(self.proj_key.weight)

        self.dual_ln_dproj = LayerNorm(d_model, export=probing)
        self.dual_ln_win = LayerNorm(d_model, export=probing)

        nn.init.zeros_(self.r_r_bias)
        nn.init.zeros_(self.r_w_bias)
        if use_bias:
            nn.init.zeros_(self.proj_query.bias)
            nn.init.zeros_(self.proj_out.bias)
            nn.init.zeros_(self.proj_val.bias)
            nn.init.zeros_(self.proj_key.bias)

    def head_reshape(self, x):
        K = self.n_head
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        return x

    def compute_scores(self, h_vecs):
        # h_vecs: B x L x H
        bsz = h_vecs.shape[0]
        n_chunks = h_vecs.shape[1] // self.chunk_size
        h_scores = self.dconv_1(h_vecs).view(bsz, n_chunks, self.chunk_size, self.n_head, self.chunk_rank)
        # bsz x num_heads x n_chunks x chunk_rank x chunk_size
        h_scores = h_scores.permute(0, 3, 1, 4, 2)
        h_scores = F.softmax(h_scores.float(), dim=-1).type_as(h_scores)
        return h_scores

    def compress_chunks(self, h_vecs, h_scores):
        # Reshape hvecs to be compatible with the weights
        # h_vecs: B x L x H
        bsz = h_vecs.shape[0]
        n_chunks = h_vecs.shape[1] // self.chunk_size
        # bsz x n_heads x n_chunks x chunk_size x d_h
        h_vecs = h_vecs.view(-1, n_chunks, self.chunk_size, self.n_head, self.d_h).permute(0, 3, 1, 2, 4)
        # bsz x n_heads x n_chunks x chunk_rank x d_h
        h_vecs = h_scores.matmul(h_vecs).view(bsz, self.n_head, n_chunks * self.chunk_rank, self.d_h)
        return h_vecs

    def get_tiles(self, x, n_queries, transpose=False):
        # input: bsz x win_bp_len x d
        bsz, win_bp_len, d = x.shape
        in_strides = x.stride()
        out_strides = (in_strides[0], self.window_len*in_strides[1], in_strides[1], d//self.n_head, 1)
        out_shape = (bsz, n_queries//self.window_len, 2*self.window_len, self.n_head, d//self.n_head)
        x = x.as_strided(size=out_shape, stride=out_strides)
        if transpose:
            # shape: bsz x n_heads x n_queries//wlen x d//n_heads x 2*wlen
            return x.permute(0, 3, 1, 4, 2)
        else:
            # shape: bsz x n_heads x n_queries//wlen x 2*wlen x d//n_heads
            return x.permute(0, 3, 1, 2, 4)

    def put_tiles(self, x):
        # input: bsz x n_heads x bp_len x self.window_len
        bsz, n_heads, bp_len, window_len = x.shape
        if bp_len > window_len:
            x = x.view(bsz, n_heads, bp_len//window_len, window_len, window_len)
            out_size = (bsz, n_heads, bp_len//window_len, window_len, 2*window_len)
            x = F.pad(x, (1, window_len))
        else:
            x = x.view(bsz, n_heads, 1, bp_len, window_len)
            out_size = (bsz, n_heads, 1, bp_len, window_len + bp_len)
            x = F.pad(x, (1, bp_len))

        stride = x.stride()
        out_stride = (stride[0], stride[1], stride[2], stride[3]-1, stride[4])
        return x.as_strided(size=out_size, stride=out_stride)

    def compute_pv(self, attn, val):
        # attn: bsz x n_head x seqlen//wlen x wlen x 2*wlen
        # val:  bsz x n_head x seqlen//wlen x 2*wlen x d_h
        bsz, n_head, chunks, wlen, _ = attn.shape
        out = attn.matmul(val)
        return out.view(bsz, n_head, int(chunks*wlen), -1)

    def get_diagonals(self, attn):
        # attn:  bsz x n_heads x bp_len//self.window_len x self.window_len x 2*self.window_len
        # takes the upper diagonal with length self.window_len from attn, ignoring the diagonal
        bsz, n_heads, n_tiles, n_query, _ = attn.shape
        out_size = (bsz, n_heads, n_tiles, n_query, self.window_len)
        in_stride = attn.stride()
        out_stride = (in_stride[0], in_stride[1], in_stride[2], in_stride[3]+1, 1)
        return attn.as_strided(size=out_size, stride=out_stride, storage_offset=1).contiguous().view(
            bsz, n_heads, -1, self.window_len)

    def _rel_shift_chunked(self, x, chunk_size, chunk_rank):
        # x: bsz x n_head x n_query x (n_chunks * chunk_rank)
        # out: same size but shifted to the left, relative position encoding
        bsz, n_head, n_query, n_c_vecs = x.shape
        n_q_chunks = n_query // chunk_size
        x = x.view(bsz, n_head, n_q_chunks, chunk_size, n_c_vecs).transpose(2, 3).contiguous()
        x = F.pad(x, [0, chunk_rank])
        p_stride = x.stride()
        out_shape = list(x.shape)
        out_shape[-1] -= chunk_rank
        out_strides = (p_stride[0], p_stride[1], p_stride[2], p_stride[3]-chunk_rank, p_stride[4])

        x = x.as_strided(size=out_shape, stride=out_strides, storage_offset=n_q_chunks*chunk_rank)
        return x.transpose(2, 3).contiguous().view(bsz, n_head, n_query, n_c_vecs)

    def attn(self, query, key_window, val_window, key_compressed, value_compressed,
             pos_embed_chunks, pos_embed_window, chunk_attn_mask=None):
        # query size = bsz x n_heads x M x H
        # key, value sizes = bsz x (seq_len + cache_len) x (n_heads * H)
        # key_compressed: bsz x n_heads x (M+L)//chunk_size*chunk_rank x H
        bsz, n_heads, seq_len, d_h = query.shape
        assert (self.window_len > 0 or self.chunk_size > 1)

        query = query / math.sqrt(self.d_model // self.n_head)

        # get the keys, values for the local window attention
        if seq_len > self.window_len:
            query_tile = query.view(bsz, n_heads, seq_len // self.window_len, self.window_len, d_h)
            key_window = self.get_tiles(key_window, seq_len, transpose=True)
            val_window = self.get_tiles(val_window, seq_len,
                                        transpose=False)  # bsz x n_heads x n_queries//wlen x 2*wlen x d//n_heads
        else:
            query_tile = query.view(bsz, n_heads, 1, seq_len, d_h)
            key_window = key_window.view(bsz, -1, self.n_head, d_h).permute(0, 2, 3, 1)[:, :, None, :, :]
            val_window = val_window.view(bsz, -1, self.n_head, d_h).permute(0, 2, 1, 3)[:, :, None, :, :]
        # bsz x n_heads x bp_len//self.window_len x self.window_len x 2*self.window_len
        attn_window = (query_tile+self.r_w_bias).matmul(key_window)
        attn_window = self.get_diagonals(attn_window)

        pos_trans = self.r_net(pos_embed_window).view(1, self.window_len, self.n_head, self.d_head).permute(0, 2, 3, 1)
        attn_window_pos = (query+self.r_r_bias).matmul(pos_trans)
        attn_window = attn_window + attn_window_pos

        # Compute the long-range attention.
        n_chunks = key_compressed.shape[2]
        # compute attention from context
        # bsz x n_heads x seq_len x (n_chunks*chunk_rank)
        attn_cont = torch.matmul(query, key_compressed.transpose(-1, -2))
        pos_chunks = self.r_net_chunk(pos_embed_chunks).view(1, n_chunks, self.n_head, self.d_head).permute(0, 2, 3, 1)

        attn_pos = torch.matmul(query, pos_chunks)  # B x H x M x L_pos
        attn_pos = self._rel_shift_chunked(attn_pos, self.chunk_size, self.chunk_rank)

        attn_compress = attn_cont + attn_pos
        if chunk_attn_mask is not None:
            attn_compress = attn_compress.view(
                bsz, n_heads, seq_len//self.chunk_size, self.chunk_size, -1)
            attn_compress = attn_compress.masked_fill(chunk_attn_mask, float('-inf'))
            attn_compress = attn_compress.view(bsz, n_heads, seq_len, -1)

        # Get the softmax score of both short-term and long-range attentions.
        full_attn = torch.cat([attn_compress, attn_window], dim=3)
        full_attn = F.softmax(full_attn.float(), dim=-1).type_as(full_attn)
        full_attn = self.dp_attn(full_attn)

        attn_compress = full_attn[:, :, :, :attn_compress.shape[3]]
        attn_window = full_attn[:, :, :, attn_compress.shape[3]:]

        attn_window = self.put_tiles(attn_window)
        out = torch.matmul(attn_compress, value_compressed) \
              + self.compute_pv(attn_window, val_window)

        return out

    def forward(self,  h, h_cache, key_pe, pos_embed_window, chunked_attn_mask=None):
        if self.grad_chk:
            out = cp.checkpoint(self.forward_, *[
                h, h_cache, key_pe, pos_embed_window, chunked_attn_mask
            ])
        else:
            out = self.forward_(h, h_cache, key_pe, pos_embed_window, chunked_attn_mask)
        return out

    def forward_(self, h, h_cache, key_pe, pos_embed_window, chunked_attn_mask=None):
        # h = bsz x seq_len x H
        # h_cache = bsz x cache_len x H
        bsz = h.size(0)
        seqlen = h.size(1)

        query = self.proj_query(h)
        query = self.head_reshape(query)

        # sequence length and cache length should be divisible by the chunk size
        assert seqlen % self.chunk_size == 0 and h_cache.shape[1] % self.chunk_size == 0

        cache_scores = self.compute_scores(h_cache)
        h_cache_compressed = self.compress_chunks(h_cache, cache_scores)

        # The projection for the cache can be compressed using dynamic projection
        h_cache_merge = h_cache_compressed.view(
            bsz, self.n_head, -1, self.d_h).transpose(1, 2).contiguous().view(
            bsz, -1, self.d_model)
        # Apply projections to the compressed sequence.
        val_cache = self.proj_val(h_cache_merge)
        key_cache = self.proj_key(h_cache_merge)
        # DualLN (dproj)
        key_cache = self.dual_ln_dproj(key_cache)
        val_cache = self.dual_ln_dproj(val_cache)
        val_cache = self.head_reshape(val_cache)
        key_cache = self.head_reshape(key_cache)

        # Apply window attention
        val_window_bp = self.proj_val(h)
        key_window_bp = self.proj_key(h)

        # better using multipliers of 8
        h_cache_win = h_cache[:, -self.window_len:]
        key_cache_win = self.proj_key(h_cache_win)
        val_cache_win = self.proj_val(h_cache_win)
        key_window = torch.cat([key_cache_win, key_window_bp], dim=1)
        val_window = torch.cat([val_cache_win, val_window_bp], dim=1)

        # DualLN (window)
        key_window = self.dual_ln_win(key_window)
        val_window = self.dual_ln_win(val_window)

        bp_scores = self.compute_scores(h)
        # Compress the projeced keys and values.
        val_bp_compressed = self.compress_chunks(val_window_bp, bp_scores)
        key_bp_compressed = self.compress_chunks(key_window_bp, bp_scores)

        # DualLN (dproj)
        val_bp_compressed = self.dual_ln_dproj(
            val_bp_compressed.transpose(1, 2).contiguous().view(bsz, -1, self.d_model))
        key_bp_compressed = self.dual_ln_dproj(
            key_bp_compressed.transpose(1, 2).contiguous().view(bsz, -1, self.d_model))
        val_bp_compressed = self.head_reshape(val_bp_compressed)
        key_bp_compressed = self.head_reshape(key_bp_compressed)

        val_compressed = torch.cat([val_cache, val_bp_compressed], dim=2)
        key_compressed = torch.cat([key_cache, key_bp_compressed], dim=2)

        out = self.attn(query, key_window, val_window, key_compressed, val_compressed, key_pe, pos_embed_window, chunked_attn_mask)  # B_K x M x D

        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(bsz, seqlen, -1)  # B x M x K_D
        out = self.proj_out(out)
        out = self.dropout(out)
        return out
