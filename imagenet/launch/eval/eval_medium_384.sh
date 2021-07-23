n_gpus=8
nnodes=1
bsz=$((64*n_gpus*nnodes))

res=384
expname=LS_medium_384

datapath=/path/to/imagenet
outpath=checkpoints

export OMP_NUM_THREADS=8

DISTRIBUTED_ARGS="--nproc_per_node ${n_gpus} \
                  --nnodes ${nnodes} \
                  --node_rank ${NODE_RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

# f: window size, p: patch size, n: number of blocks per stage, g: number of global tokens, r: number of dynamic projections
python -m torch.distributed.launch ${DISTRIBUTED_ARGS}  --use_env \
    run_experiment_distributed.py --config-file \
    "config/msvit.yaml" --data ${datapath} \
    --expname ${expname} \
    --output_dir ${outpath}/${expname}  \
    --resume-path ${outpath}/${expname}/model_best.pth \
    DATA.TRAIN "'imagenet-draco'," DATA.TEST "'imagenet-draco'," OPTIM.OPT adamw \
    DATALOADER.BSZ ${bsz} \
    MODEL.VIT.MSVIT.ATTN_TYPE ls  \
    OPTIM.EPOCHS 300 SOLVER.LR_POLICY cosine INPUT.IMAGE_SIZE ${res} MODEL.VIT.MSVIT.ARCH \
    "l1,h3,d96,n1,s1,g1,p4,f8,a0,r8_l2,h3,d192,n4,s1,g1,p2,f8,a0,r8_l3,h6,d384,n16,s0,g1,p2,f8,a0,r8_l4,h12,d768,n1,s0,g0,p2,f8,a0,r8" \
    AUG.REPEATED_AUG False EVALUATE True \


# INFO:root:ACCURACY: 84.42599487304688%
# INFO:root:iter: 0  max mem: 3989
#     accuracy_metrics - top1: 84.4260 (84.4260)  top5: 96.8100 (96.8100)
#     epoch_metrics    - total_cnt: 50000.0000 (50000.0000)  loss: 0.0222 (0.0222)  time: 0.0106 (0.0106)