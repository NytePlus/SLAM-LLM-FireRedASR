vc submit -p pdgpu-sjtu-ai -i hub.szaic.com/sjtu/sjtu_yukai-wangchencheng_slm:v2.3 \
    -c 20 -m 360G -g 3 -n 1 \
    -v "/aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/SLAM-LLM-FireRedASR:/workspace,/aistor/sjtu/hpc_stor01/home/wangchencheng/models:/models,/aistor/sjtu/hpc_stor01/home/wangchencheng/data:/data" \
    --cmd "bash scripts/finetune_deepspeed.sh"
# 单GPU申请的CPU核数不能超过20个，且单GPU申请的MEM大小不能超过120G

vc submit -p pdgpu-sjtu-ai -i hub.szaic.com/sjtu/sjtu_yukai-wangchencheng_slm:v2.3 -c 80 -m 480G -g 4 -j "mala2" -v "/aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/SLAM-LLM-FireRedASR:/workspace,/aistor/sjtu/hpc_stor01/home/wangchencheng/models:/models,/aistor/sjtu/hpc_stor01/home/wangchencheng/data:/data" \
    -d $PWD JOB=1:1 logs/qwen.JOB.log --cmd "sleep 10000000"

docker run -it --rm --net=host\
    -v /aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/SLAM-LLM-FireRedASR:/workspace \
    -v /aistor/sjtu/hpc_stor01/home/wangchencheng/models:/models \
    -v /aistor/sjtu/hpc_stor01/home/wangchencheng/data:/data \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    hub.szaic.com/sjtu/sjtu_yukai-wangchencheng_slm:v2.3 \
    /bin/bash

vc list -p pdgpu-sjtu-ai --pri