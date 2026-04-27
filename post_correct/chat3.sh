#!/bin/bash

export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export ASCEND_RT_VISIBLE_DEVICES=0

code_dir=.
use_peft=false
use_fp16=true
freeze_encoder=true
freeze_projector=true
freeze_llm=true
eval_max_frame_length=15000
ckpt_path=
dataset=slidespeech
task=asr
sub_test=test
test_scp_file_path=/data/${dataset}/${sub_test}_oracle_v1/

deepspeed_config=conf/inference_config.json

llm_name=vicuna-7b-v1.5
if [[ $llm_name == "vicuna-7b-v1.5" ]]
then
    llm_path=/models/AI-ModelScope/vicuna-7b-v1.5
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

projector=linear


python $code_dir/post_correct/chat3.py \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.encoder_path=$encoder_ckpt_path \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.encoder_projector=$projector \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.attn_implementation=flash_attention_2 \
    ++train_config.use_fp16=$use_fp16 \
    # ++ckpt_path=$ckpt_path/pytorch_model.bin \