ckpt_path=exp/20260121-1702-slidespeech-linear/aispeech_asr_epoch_24_total_step_100000
dataset=slidespeech
task=asr
sub_test=test
test_scp_file_path=/data/${dataset}/${sub_test}_oracle_v1/

decode_log=$ckpt_path/decode_${dataset}_${task}_${sub_test}
context=/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech/test_oracle_v1/test_meta_info.json

python score/normalize.py ${decode_log}_corr ${decode_log}_norm_corr
python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_norm_gt ${decode_log}_norm_corr > ${decode_log}_corr_cer
python score/score.py --pred ${decode_log}_norm_corr --multitask ${test_scp_file_path}/multitask.jsonl --norm --lenient > ${decode_log}_corr_wer