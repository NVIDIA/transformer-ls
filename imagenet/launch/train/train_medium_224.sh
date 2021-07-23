# Each GPU has 32GB memory
n_gpus=8
nnodes=1
bsz=$((128*n_gpus*nnodes))

lr=8e-4
dpp=0.3
res=224
expname=LS_medium_bsz_${bsz}_300epochs_r8_lr${lr}_dpp${dpp}

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
    DATA.TRAIN "'imagenet-draco'," DATA.TEST "'imagenet-draco'," OPTIM.OPT adamw \
    OPTIM.LR ${lr} OPTIM.WD 0.1 DATALOADER.BSZ ${bsz} \
    MODEL.VIT.MSVIT.ATTN_TYPE ls MODEL.VIT.DROP_PATH ${dpp} \
    OPTIM.EPOCHS 300 SOLVER.LR_POLICY cosine INPUT.IMAGE_SIZE ${res} MODEL.VIT.MSVIT.ARCH \
    "l1,h3,d96,n1,s1,g1,p4,f7,a0,r8_l2,h3,d192,n4,s1,g1,p2,f7,a0,r8_l3,h6,d384,n16,s0,g1,p2,f7,a0,r8_l4,h12,d768,n1,s0,g0,p2,f7,a0,r8" \
    AUG.REPEATED_AUG False \
