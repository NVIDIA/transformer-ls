function runexp {

gpu=${1}
task=${2}
model=${3}
layers=${4}
lms=${5}         # lms (landmarks) is r in paper
k_conv=${6}
wsize=${7}       # wsize is w in paper
lr=${8}
wd=${9}
seed=${10}
flags=${11}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=${task}-${model}-l_${layers}-lr_${lr}-wd_${wd}-lms_${lms}-winsize_${wsize}-k_conv_${k_conv}-seed_${seed}${flags_print}

cmd="
CUDA_VISIBLE_DEVICES=${gpu} python run_tasks.py --model ${model} --task ${task}
    --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
    --learning_rate ${lr} --weight_decay ${wd} --num_layers ${layers} --cls_last_layer --max_seq_len 2048
    --dropout_prob 0 --attention_dropout 0
    --num_train_steps 5000 --num_eval_steps 62 --eval_frequency 50 --batch_size 32 --warmup 1000
    --n_train_samples 96000 --n_dev_samples 2000 --n_test_samples 2000 --num_classes 10
    --seed ${seed}
"

debug=1
if [ ${debug} -eq 0 ]; then
cmd="${cmd} --logging --expname ${expname}  > logs/${expname}.log 2>&1 &"
else
cmd="${cmd} "
fi

echo logs/${expname}.log

eval ${cmd}

}

# The following hyperparameters correspond to Transformer-LS (w,r = 8,32) in the paper.
# One can change it to Transformer-LS (best) with lms=2, win_size=16
# runexp  gpu   task    model    layers  lms  k_conv  win_size lr   wd   seed   flags
runexp    0    listops   lsta      2     32    -1      8     1e-4   0.0 1234

