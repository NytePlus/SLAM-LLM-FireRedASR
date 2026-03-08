ckpt_path=${EXP_DIR}/${CKPT_NAME}
dataset=slidespeech
task=asr
sub_test=test
test_scp_file_path=${DATA_DIR}/${dataset}/${sub_test}_oracle_v1/

decode_log=$ckpt_path/decode_${dataset}_${task}_${sub_test}

python score/normalize.py ${decode_log}_pred ${decode_log}_norm_pred
python score/normalize.py ${decode_log}_gt ${decode_log}_norm_gt

python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_norm_gt ${decode_log}_norm_pred > ${decode_log}_norm_cer
python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer

# python score/score.py --pred ${decode_log}_pred --multitask ${test_scp_file_path}/multitask.jsonl --lenient > ${decode_log}_wer
python score/score.py --pred ${decode_log}_pred --multitask ${test_scp_file_path}/multitask.jsonl --norm --lenient > ${decode_log}_norm_wer
