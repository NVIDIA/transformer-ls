#!/usr/bin/env bash

lr=1.25e-4
warmup=5000
iters=50000
csize=16
crank=1
bsize=2
memlen=4096
tps=4096
wlen=512

wd=0.1
dp=0.4

update_freq=4 # final batch size will be (update_freq * bsize * n_gpus)

task=enwik8

expname=${task}-large-phase2

export OMP_NUM_THREADS=8

fairseq-train \
    datasets/${task}/data-bin/ \
    --user-dir ./model_lib \
    --fp16 --fp16-no-flatten-grads \
    --task truncated_bptt_lm --arch transformer-ls \
    --n-layer 30 --d-model 512 --n-head 8 --d-inner 2048 --dropout ${dp} --emb-dropout ${dp} \
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
    --finetune-from-model chks/${task}-large-phase1/checkpoint_best.pt \
