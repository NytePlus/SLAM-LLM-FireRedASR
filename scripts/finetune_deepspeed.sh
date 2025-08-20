#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
# export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
# export HCCL_CONNECT_TIMEOUT=7200
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0

# run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM-FireRedASR
# cd $run_dir
code_dir=.
dataset=mandarin_long_merge_20-30
task=
train_scp_file_path=./data/${dataset}/${task}/train/
dev_scp_file_path=./data/${dataset}/${task}/dev/
train_max_frame_length=600
eval_max_frame_length=2000
multitask_prompt_path=conf/multiprompt.jsonl
ckpt_path=
# prompt_style="\{\}\\<speech\\>" # "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n" | "USER: {}\n ASSISTANT:"
projector=linear

use_peft=true # For llm
use_fp16=true
freeze_encoder=false
freeze_llm=true
# use absolute path
deepspeed_config=conf/ds_config.json


# Choose LLM
llm_name=Qwen2-7B-Instruct
llm_path=/aistor/aispeech/hpc_stor01/group/asr/model/${llm_name}
llm_dim=3584

output_dir=${code_dir}/exp/$(date +"%Y%m%d-%H%M")-$dataset-lora${use_peft}_${task}_instruct
hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++dataset_config.train_max_frame_length=$train_max_frame_length \
++dataset_config.eval_max_frame_length=$eval_max_frame_length \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++train_config.model_name=aispeech_asr \
++train_config.num_epochs=50 \
++train_config.use_peft=$use_peft \
++train_config.freeze_llm=$freeze_llm \
++train_config.freeze_encoder=$freeze_encoder \
++train_config.batching_strategy=dynamic \
++train_config.validation_interval=10000 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++metric=acc \
"
if [[ $use_peft == "true" && -n "$ckpt_path" ]];then
    hydra_args+=" ++ckpt_path=$ckpt_path/pytorch_model.bin"
fi

# debug
# deepspeed \
#      $code_dir/finetune_deepspeed.py \
#      ++train_config.enable_fsdp=false \
#      ++train_config.enable_ddp=true \
#      ++train_config.use_fp16=$use_fp16 \
#      ++deepspeed_config=$deepspeed_config \
#      ${hydra_args}

# exit 0

HOST_FILE="/tmp/"${JobID}                        #生成的hostfile的完整文件名，$JobID调度系统会自动生成
 
echo "${VC_MASTER_HOSTS} slots=${GPU_PER_TASK}" > ${HOST_FILE}
echo "${VC_WORKER_HOSTS}" | awk -F ',' -v gpu_num=$GPU_PER_TASK '{for (i=1; i<=NF; i++) print $i" slots="gpu_num}' >> ${HOST_FILE}

deepspeed \
    --node_rank=$RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --hostfile $HOST_FILE \
    --no_ssh \
    $code_dir/finetune_deepspeed.py \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=$use_fp16 \
    ++deepspeed_config=$deepspeed_config \
    ${hydra_args}


