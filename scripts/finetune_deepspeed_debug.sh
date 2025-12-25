#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0

code_dir=.
dataset=slidespeech
task=asr
train_scp_file_path=/data/${dataset}/train_95/
dev_scp_file_path=/data/${dataset}/dev_oracle_v1/
train_max_frame_length=50000
eval_max_frame_length=50000
multitask_prompt_path=conf/multiprompt.jsonl
ckpt_path=

use_peft=false # For llm
use_fp16=true
freeze_encoder=true
freeze_projector=false
freeze_llm=true

firered_path=
# use absolute path
deepspeed_config=conf/ds_config.json

# Choose Encoder
encoder_name=wavlm
if [[ $encoder_name == "whisper" ]]
then
    encoder_ckpt_path=/aistor/sjtu/hpc_stor01/home/xiyu/models/Whisper/medium.pt
    mel_size=80 
    encoder_dim=1024
    file=dataset/speech_dataset_large_whisper.py:get_speech_dataset
    
elif [[ $encoder_name == "wavlm" ]]
then
    encoder_ckpt_path=/aistor/sjtu/hpc_stor01/home/guoyiwei/remote/code/AudioFeatExtraction/wavlm/pretrained/WavLM-Large.pt
    encoder_dim=1024
    file=dataset/speech_dataset_large_wavlm.py:get_speech_dataset

elif [[ $encoder_name == "conformer" ]]
then
    firered_path=
    encoder_dim=1280
    file=dataset/speech_dataset_large.py:get_speech_dataset
else
    exit 1
fi


# Choose Projector
projector=linear

# Choose LLM
llm_name=vicuna-7b-v1.5
if [[ $llm_name == "vicuna-7b-v1.5" ]]
then
    llm_path=/aistor/sjtu/hpc_stor01/home/xiyu/models/vicuna-7b-v1.5
    llm_dim=4096
elif [[ $llm_name == "Qwen2.5-7B-Instruct" ]]
then
    llm_path=/aistor/sjtu/hpc_stor01/home/yangyi/model/Qwen2.5-7B-Instruct
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=
    llm_dim=3584 
elif [[ $llm_name == "Qwen2.5-1.5B-Instruct" ]]
then
    llm_path=/aistor/sjtu/hpc_stor01/home/yangyi/model/Qwen2.5-1.5B-Instruct
    llm_dim=3584 
else
    exit 1
fi

output_dir=${code_dir}/exp/$(date +"%Y%m%d-%H%M")-$dataset-lora${use_peft}_${task}_instruct
hydra_args="
hydra.run.dir=$output_dir \
++model_config.encoder_name=$encoder_name \
++model_config.encoder_path=$encoder_ckpt_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=$projector \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.firered_path=$firered_path \
++dataset_config.file=$file \
++dataset_config.train_max_frame_length=$train_max_frame_length \
++dataset_config.eval_max_frame_length=$eval_max_frame_length \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++dataset_config.spec_aug=false \
++dataset_config.wav_reverb=false \
++dataset_config.add_noise=false \
++train_config.model_name=aispeech_asr \
++train_config.num_epochs=50 \
++train_config.use_peft=$use_peft \
++train_config.freeze_llm=$freeze_llm \
++train_config.freeze_encoder=$freeze_encoder \
++train_config.freeze_projector=$freeze_projector \
++train_config.batching_strategy=dynamic \
++train_config.validation_interval=10000 \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

if [[ $use_peft == "true" && -n "$ckpt_path" ]];then
    hydra_args+=" ++ckpt_path=$ckpt_path/pytorch_model.bin"
fi

# 单机Debug
torchrun --nproc_per_node=1 \
    $code_dir/finetune_deepspeed.py \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=$use_fp16 \
    ++deepspeed_config=$deepspeed_config \
    ${hydra_args}

exit 0

# 调试机多卡训练
# deepspeed \
#     --num_nodes 1 \
#     --num_gpus 8 \
#     $code_dir/finetune_deepspeed.py \
#     ++train_config.enable_fsdp=false \
#     ++train_config.enable_ddp=true \
#     ++train_config.use_fp16=$use_fp16 \
#     ++deepspeed_config=$deepspeed_config \
#     ${hydra_args}

# exit 0
