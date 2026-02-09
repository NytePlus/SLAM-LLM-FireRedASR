ckpt_path=exp/20251110-1416-slidespeech-hotenc_asr/aispeech_asr_epoch_32_total_step_400000
dataset=slidespeech
task=asr
sub_test=test
test_scp_file_path=/data/${dataset}/${sub_test}_oracle_v1/

decode_log=$ckpt_path/decode_${dataset}_${task}_${sub_test}
context=/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech/test_oracle_v1/test_meta_info.json

export ASCEND_RT_VISIBLE_DEVICES=3
python score/ppl.py --pred ${decode_log}_pred --multitask ${test_scp_file_path}/multitask.jsonl --context $context --norm --lenient 
#  > ${decode_log}_norm_pplcmg10