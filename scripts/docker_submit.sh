vc submit -p pdgpu-sjtu-ai -i hub.szaic.com/hpc/ai_asr-jingpeng-ps-slm:v2.0 \
    -c 20 -m 120G -g 1 -n 1 \
    -v "/aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/SLAM-LLM-FireRedASR:/workspace,/aistor/sjtu/hpc_stor01/home/wangchencheng/models:/models,/aistor/sjtu/hpc_stor01/home/wangchencheng/data:/data" \
    --cmd "bash scripts/finetune_deepspeed.sh"

docker run -it --rm \
    -v /aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/SLAM-LLM-FireRedASR:/workspace \
    -v /aistor/sjtu/hpc_stor01/home/wangchencheng/models:/models \
    -v /aistor/sjtu/hpc_stor01/home/wangchencheng/data:/data \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    hub.szaic.com/hpc/ai_asr-jingpeng-ps-slm:v2.0 \
    /bin/bash

vc list -p pdgpu-sjtu-ai --pri