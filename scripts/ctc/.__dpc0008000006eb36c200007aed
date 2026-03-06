#!/bin/bash

code_dir=.
use_peft=false
use_fp16=true
freeze_encoder=true
freeze_projector=true
freeze_llm=true
eval_max_frame_length=15000
ckpt_path=exp/20260216-1515-slidespeech-ctc/aispeech_asr_epoch_13_total_step_80000
dataset=slidespeech
task=asr
sub_test=test
test_scp_file_path=/data/${dataset}/${sub_test}_oracle_v1/

deepspeed_config=conf/inference_config.json

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

decode_log=$ckpt_path/decode_${dataset}_${task}_${sub_test}
deepspeed \
    $code_dir/inference_batch_deepspeed.py \
    hydra.run.dir=$ckpt_path \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.encoder_path=$encoder_ckpt_path \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.encoder_projector=$projector \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.firered_path=$firered_path \
    ++dataset_config.file=$file \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.inference_mode=true \
    ++dataset_config.eval_max_frame_length=$eval_max_frame_length \
    ++dataset_config.max_audio_length=30 \
    ++train_config.model_name=aispeech_asr \
    ++train_config.use_peft=$use_peft \
    ++train_config.freeze_llm=$freeze_llm \
    ++train_config.freeze_encoder=$freeze_encoder \
    ++train_config.freeze_projector=$freeze_projector \
    ++train_config.batching_strategy=fixed \
    ++train_config.val_batch_size=1 \
    ++train_config.num_epochs=1 \
    ++train_config.num_workers_dataloader=0 \
    ++train_config.output_dir=$output_dir \
    ++train_config.use_fp16=$use_fp16 \
    ++train_config.repetition_penalty=1.0 \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/pytorch_model.bin \
    ++deepspeed_config=$deepspeed_config \
|| exit 1


python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer