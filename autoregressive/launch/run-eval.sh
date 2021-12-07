#!/usr/bin/env bash

chk=/path/to/checkpoint

dataset=enwik8 # or text8
memlen=32768
override_str="{'mem_len':${memlen}}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-eval-lm datasets/${dataset}/data-bin --path ${chk} \
 --user-dir model_lib \
 --task truncated_bptt_lm --batch-size 1 \
 --tokens-per-sample 256 --gen-subset test \
 --model-overrides ${override_str}
