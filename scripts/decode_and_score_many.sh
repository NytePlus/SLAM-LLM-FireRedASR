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