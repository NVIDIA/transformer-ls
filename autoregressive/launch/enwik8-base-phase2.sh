#!/usr/bin/env bash

lr=1.25e-4
warmup=5000
iters=50000
csize=16
crank=1
bsize=4
memlen=4096
tps=4096
wlen=512

wd=0.01
dp=0.2

update_freq=1

task=enwik8

expname=${task}-base-phase2

export OMP_NUM_THREADS=8

# It is important to set --num-workers to larger values for the speed on text8.
# A good practice is to set it to (num_cpu_cores / num_gpus)
# Note: we need 8 V100 GPUs with 32GB memory
fairseq-train \
    datasets/${task}/data-bin/ \
    --user-dir ./model_lib \
    --fp16 --fp16-no-flatten-grads \
    --task truncated_bptt_lm --arch transformer-ls \
    --n-layer 12 --d-model 512 --n-head 8 --d-inner 2048 --dropout ${dp} --emb-dropout ${dp} \
    --tokens-per-sample ${tps} --mem-len ${memlen} --window-len ${wlen} \
    --keep-last-epochs 1 --validate-interval-updates 1000 \
    --optimizer adam --weight-decay ${wd} \
    --lr-scheduler fixed --warmup-updates ${warmup}  --max-update ${iters} --batch-size-valid 32 \
    --lr ${lr}  --batch-size ${bsize} \
    --save-dir ./chks/${expname} \
    --chunk-size ${csize} --chunk-rank ${crank} \
    --update-freq ${update_freq} \
    --criterion char_level_lm_loss  --pre-ln --use-gelu \
    --num-workers 8 \
    --seed 2  --log-interval 25 \
    --finetune-from-model ./chks/${task}-base-phase1/checkpoint_last.pt


