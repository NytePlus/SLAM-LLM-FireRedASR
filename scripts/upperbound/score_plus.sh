ckpt_path=exp/20251110-1416-slidespeech-lorafalse_asr_instruct/aispeech_asr_epoch_32_total_step_400000
dataset=slidespeech
task=asr
sub_test=test
test_scp_file_path=/data/${dataset}/${sub_test}_oracle_v1/

decode_log=$ckpt_path/decode_${dataset}_nokw_${task}_${sub_test}

python score/score_plus.py --pred ${decode_log}_pred --multitask ${test_scp_file_path}/multitask.jsonl --norm --lenient \
    --multitask2 ${test_scp_file_path}/slides/multitask.jsonl > ${decode_log}_norm_nnpright_wer