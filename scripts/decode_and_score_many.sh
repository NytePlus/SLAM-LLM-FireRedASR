EXP_NAME=exp/history-20260328-0046-slidespeech
EPOCH_LIST="exp/history-20260328-0046-slidespeech/aispeech_asr_epoch_4_total_step_10000 exp/history-20260328-0046-slidespeech/aispeech_asr_epoch_7_total_step_20000 exp/history-20260328-0046-slidespeech/aispeech_asr_epoch_13_total_step_40000"
DECODE_SCRIPT=decode.sh
epochs_array=($EPOCH_LIST)

for epoch in "${epochs_array[@]}"; do
    echo "------------------------------------------------"
    ckpt_name=$EXP_NAME/$epoch
    echo "当前处理ckpt名称: $ckpt_name"
    
    CKPT_NAME="$ckpt_name" bash scripts/${DECODE_SCRIPT}
    CKPT_NAME="$ckpt_name" bash scripts/score.sh
    
    if [ $? -eq 0 ]; then
        echo "Successfully finished: $epoch"
    else
        echo "Error occurred in: $epoch"
    fi
done