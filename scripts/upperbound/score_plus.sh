ckpt_path=exp/20260121-1702-slidespeech-linear/aispeech_asr_epoch_24_total_step_100000
dataset=slidespeech
task=asr
sub_test=test
test_scp_file_path=/data/${dataset}/${sub_test}_oracle_v1/

decode_log=$ckpt_path/decode_${dataset}_${task}_${sub_test}

python score/score_except_hw.py --pred ${decode_log}_pred --multitask ${test_scp_file_path}/multitask.jsonl --norm --lenient \
    --multitask2 ${test_scp_file_path}/slides/multitask.jsonl > ${decode_log}_norm_nnpright_wer