#!/bin/bash

code_dir=.
use_peft=true
eval_max_frame_length=3000
ckpt_path=exp/20250829-0037-mandarin_long_merge_20-30-loratrue__instruct/aispeech_asr_epoch_1_step_30000
dataset=test
task=test_bgb_A3_far_July18

test_scp_file_path=/aistor/aispeech/hpc_stor01/group/asr/${dataset}/${task}

llm_name="Qwen2-7B-Instruct"
llm_path=/aistor/aispeech/hpc_stor01/group/asr/model/${llm_name}
llm_dim=3584


decode_log=$ckpt_path/decode_${dataset}_${task}
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
    ++ckpt_path=$ckpt_path/pytorch_model.bin \
|| exit 1


python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer