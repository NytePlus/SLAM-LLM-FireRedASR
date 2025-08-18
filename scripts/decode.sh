#!/bin/bash
run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM-FireRedASR
cd $run_dir
code_dir=.
projector=linear
encoder_name=whisper
use_peft=true
use_fp16=false
eval_max_frame_length=3000
# ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM-FireRedASRSLAM-LLM-ASR/exp/20250509-1623-aishell-1-loratrue_hotword_instruct/aispeech_asr_epoch_1_step_10
dataset=aishell-2
task=asr
target=test
multitask_prompt_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multiprompt.jsonl"
test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/${target}/
# Choose Encoder


llm_name="Qwen2-7B-Instruct"
llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-7B-Instruct
llm_dim=3584 

ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM-FireRedASR/exp/20250709-2008-aishell-2-loratrue_hotword_instruct/aispeech_asr_epoch_1_step_21000

decode_log=$ckpt_path/decode_${dataset}_${task}_${target}
python \
    $code_dir/inference_batch.py \
    hydra.run.dir=$ckpt_path \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++dataset_config.dataset=$dataset \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.inference_mode=true \
    ++dataset_config.eval_max_frame_length=$eval_max_frame_length \
    ++train_config.model_name=aispeech_asr \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=dynamic \
    ++train_config.num_epochs=1 \
    ++train_config.num_workers_dataloader=0 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/pytorch_model.bin


python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer
python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer