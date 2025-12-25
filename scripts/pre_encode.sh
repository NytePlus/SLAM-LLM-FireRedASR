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
multitask_prompt_path=conf/multiprompt.jsonl
ckpt_path=


firered_path=

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
"

python \
     $code_dir/dataset/pre_encode.py \
     ${hydra_args}

exit 0
