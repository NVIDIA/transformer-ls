# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.gelu import gelu
from .layer import ChunkedLSAttention


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_inner, dropout, use_gelu):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout = nn.Dropout(dropout)
        self.use_gelu = use_gelu

    def forward(self, h):
        if self.use_gelu:
            h1 = gelu(self.fc1(h))
        else:
            h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        h2 = self.dropout(h2)
        return h2


class TransformerLSLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, chunk_size, chunk_rank, window_len,
                 dropout, grad_chk, use_bias, pre_ln, use_gelu, probing):
        nn.Module.__init__(self)
        self.pre_ln = pre_ln

        self.attn = ChunkedLSAttention(
            d_model=d_model, n_head=n_head,
            chunk_size=chunk_size, chunk_rank=chunk_rank, window_len=window_len,
            dropout=dropout, grad_chk=grad_chk, use_bias=use_bias, probing=probing)
        self.norm1 = LayerNorm(d_model, export=probing)
        self.ff = FeedForwardLayer(d_model=d_model, d_inner=d_inner, dropout=dropout, use_gelu=use_gelu)
        self.norm2 = LayerNorm(d_model, export=probing)

    def forward(self, h, h_cache, key_pe, pos_embed_window, attn_mask=None, chunk_attn_mask=None):
        # h = B x M x H
        # h_cache = B x L x H

        if self.pre_ln:
            h = self.norm1(h)
            h_cache = self.norm1(h_cache)

        attn_out = self.attn(h, h_cache, key_pe, pos_embed_window, chunk_attn_mask)

        if self.pre_ln:
            h = h + attn_out
        else:
            h = self.norm1(h + attn_out)  # B x M x H

        if self.ff is not None:
            if self.pre_ln:
                h = self.norm2(h)
            ff_out = self.ff(h)

            if self.pre_ln:
                out = h + ff_out  # B x M x H
            else:
                out = self.norm2(h + ff_out)  # B x M x H
        else:
            out = h
        return out


class TransformerLSModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        d_inner,
        n_head,
        n_layer,
        mem_len,
        emb_dropout,
        chunk_rank,
        chunk_size,
        window_len,
        dropout,
        use_bias,
        pre_ln,
        use_gelu,
        grad_chk,
        clamp_len,
        cpos_clamp_len=-1,
        probing=False,
    ):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.in_emb.weight, mean=0, std=d_model ** -0.5)
        self.pos_emb = PositionalEmbedding(d_model)
        # nn.init.uniform_(self.in_emb.weight, -0.01, 0.01)
        self.out_emb = nn.Linear(d_model, vocab_size)
        self.out_emb.weight = self.in_emb.weight
        self.window_len = window_len

        # Some knobs copied from Transformer XL
        self.init = 'normal'
        self.init_range = 0.01
        self.proj_init_std = 0.01
        self.init_std = 0.02

        self.cpos_clamp_len = cpos_clamp_len
        self.d_model = d_model
        if emb_dropout > 0:
            self.emb_dropout = nn.Dropout(emb_dropout)
        else:
            self.emb_dropout = None
        self.chunk_size = chunk_size
        self.chunk_rank = chunk_rank

        self.layers = nn.ModuleList()

        self.layers.extend(
            TransformerLSLayer(
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_head,
                chunk_rank=chunk_rank,
                chunk_size=chunk_size,
                window_len=window_len,
                dropout=dropout,
                use_bias=use_bias,
                pre_ln=pre_ln,
                use_gelu=use_gelu,
                grad_chk=grad_chk,
                probing=probing,
            )
            for _ in range(n_layer)
        )
        self.mem_len = mem_len

        self.clamp_len = clamp_len

        self.apply(self._init_weights)

    def forward(self, x, h_cache, target=None):
        # x size = B x M
        padded = False
        if self.chunk_size > 0 and (x.shape[1] % self.chunk_size ):
            # or x.shape[1] % self.window_len
            # usually happens at the end
            # ad-hoc solution for the chunking issue during evaluation
            orig_seqlen = x.shape[1]
            pad_multip = abs(self.chunk_size * self.window_len) // math.gcd(self.chunk_size, self.window_len)
            n_pad = pad_multip - x.shape[1] % pad_multip
            x = F.pad(x, (0, n_pad))
            padded = True

        block_size = x.size(1)
        h = self.in_emb(x) #.mul_(self.d_model ** 0.5)  # B x M x H
        h.mul_(self.d_model ** 0.5)

        mlen = h_cache[0].shape[1]
        klen = h.shape[1] + mlen

        dec_attn_mask = None
        pos_seq = torch.arange(self.window_len - 1, -1, -1.0, device=h.device, dtype=h.dtype)

        n_chunk_vecs = klen // self.chunk_size * self.chunk_rank
        n_chunks = klen // self.chunk_size
        n_mem_chunks = mlen // self.chunk_size
        chunk_attn_mask = torch.triu(h.new_ones((x.shape[1]//self.chunk_size, n_chunks), dtype=torch.bool), diagonal=n_mem_chunks)[
                        None, None, :, None, :, None]
        chunk_attn_mask = chunk_attn_mask.expand(-1, -1, -1, -1, -1, self.chunk_rank).contiguous().view(1, 1, -1, 1, n_chunks*self.chunk_rank)
        pos_chunk_ids = torch.arange(n_chunk_vecs - 1, -1, -1.0, device=h.device, dtype=h.dtype)
        if self.cpos_clamp_len > 0:
            pos_chunk_ids.clamp_(max=self.cpos_clamp_len)
        pos_chunks = self.pos_emb(pos_chunk_ids)

        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        if self.emb_dropout is not None:
            h = self.emb_dropout(h)
            pos_emb = self.emb_dropout(pos_emb)

        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = self.mem_len
            if cache_size > block_size:
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + block_size :, :], h], dim=1
                ).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], pos_chunks, pos_emb, dec_attn_mask, chunk_attn_mask)  # B x M x H

        if self.emb_dropout is not None:
            h = self.emb_dropout(h)

        out = F.log_softmax(self.out_emb(h).float(), dim=-1).type_as(h)
        dummy_loss = None

        if padded:
            out = out[:, :orig_seqlen]

        return out, h_cache_next, dummy_loss

    def get_aux_loss(self):
        loss = 0.0
        for layer in self.layers:
            loss += layer.attn.attn.adaptive_span.get_loss()
        return self.aux_loss_scaler * loss

    def get_current_max_span(self):
        max_span = 0.0
        for layer in self.layers:
            max_span = max(
                max_span, layer.attn.attn.adaptive_span.get_current_max_span()
            )
        return max_span

    def get_current_avg_span(self):
        avg_span = 0.0
        for layer in self.layers:
            avg_span += layer.attn.attn.adaptive_span.get_current_avg_span()
        return avg_span / len(self.layers)

    def _init_weight(self, weight):
        if self.init == "uniform":
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == "normal":
            nn.init.normal_(weight, 0.0, self.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        """Initialize the weights."""
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            hit = False
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
                hit = True
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
                hit = True
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
                hit = True
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)
                hit = True
            if not hit:
                print("Missing {}".format(classname))
