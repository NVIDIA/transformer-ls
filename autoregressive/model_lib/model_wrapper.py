# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
)
from .model import TransformerLSModel


logger = logging.getLogger(__name__)


@dataclass
class TransformerLSConfig(FairseqDataclass):
    # defaults come from https://github.com/facebookresearch/adaptive-span/blob/master/experiments/enwik8_small.sh
    vocab_size: int = 50
    d_model: int = 256
    n_head: int = 4
    d_inner: int = 1024
    n_layer: int = 8
    dropout: float = 0.0
    emb_dropout: float = 0.0
    chunk_rank: int = 1
    chunk_size: int = 32
    mem_len: int = 4096
    window_len: int = 256
    grad_chk: bool = False
    pre_ln: bool = False
    use_gelu: bool = False
    use_bias: bool = False
    clamp_len: int = -1
    cpos_clamp_len: int = -1
    probing: bool = False


@register_model("transformer-ls", dataclass=TransformerLSConfig)
class TransformerLS(FairseqLanguageModel):
    @classmethod
    def build_model(cls, cfg: TransformerLSConfig, task):
        return cls(TransformerLSDecoder(cfg, task))

    def get_aux_loss(self):
        return self.decoder.get_aux_loss()

    def get_current_max_span(self):
        return self.decoder.get_current_max_span()

    def get_current_avg_span(self):
        return self.decoder.get_current_avg_span()


class TransformerLSDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, task):

        super().__init__(task.target_dictionary)

        self.config = cfg

        config = TransformerLSConfig(
            vocab_size=len(task.target_dictionary),
            d_model=cfg.d_model,
            n_head=cfg.n_head,
            d_inner=cfg.d_inner,
            n_layer=cfg.n_layer,
            dropout=cfg.dropout,
            emb_dropout=cfg.emb_dropout,
            mem_len=cfg.mem_len,
            chunk_rank=cfg.chunk_rank,
            chunk_size=cfg.chunk_size,
            window_len=cfg.window_len,
            grad_chk=cfg.grad_chk,
            pre_ln=cfg.pre_ln,
            use_gelu=cfg.use_gelu,
            use_bias=cfg.use_bias,
            clamp_len=cfg.clamp_len,
            cpos_clamp_len=cfg.cpos_clamp_len,
            probing=cfg.probing,
        )
        logger.info(config)
        del config.__dict__['_name']
        self.model = TransformerLSModel(**config.__dict__)
        self.cache_size = cfg.mem_len

        self._mems = None

    def forward(
        self,
        src_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        bsz = src_tokens.size(0)
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state("mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        if mems is None:
            # first time init
            mems = self.init_hid_cache(bsz)
        output = self.model(x=src_tokens, h_cache=mems,)
        if incremental_state is not None:
            self.set_incremental_state(incremental_state, "mems", output[1])
        else:
            self._mems = output[1]
        return (output[0],)

    def init_hid_cache(self, batch_sz):
        hid = []
        for layer in self.model.layers:
            param = next(self.model.parameters())
            h = torch.zeros(
                batch_sz,
                self.cache_size,
                self.config.d_model,
                dtype=param.dtype,
                device=param.device,
            )
            hid.append(h)
        return hid

    def get_aux_loss(self):
        return self.model.get_aux_loss()

    def get_current_max_span(self):
        return self.model.get_current_max_span()

    def get_current_avg_span(self):
        return self.model.get_current_avg_span()

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        raise NotImplementedError("This is required for generation/beam search")
        # mems = self.get_incremental_state(incremental_state, "mems")
        # if mems is not None:
        #     new_mems = [mems_i.index_select(1, new_order) for mems_i in mems]
        #     self.set_incremental_state(incremental_state, "mems", new_mems)
