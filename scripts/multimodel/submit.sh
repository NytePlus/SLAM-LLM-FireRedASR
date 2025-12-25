vc submit -p pdgpu-sjtu-ai -i hub.szaic.com/hpc/ai_asr-jingpeng-ps-slm:v2.0 \
    -c 20 -m 120G -g 2 -n 1 \
    -v "/aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/SLAM-LLM-FireRedASR:/workspace,/aistor/sjtu/hpc_stor01/home/wangchencheng/models:/models,/aistor/sjtu/hpc_stor01/home/wangchencheng/data:/data" \
    --cmd "bash scripts/multimodel/finetune_deepspeed.sh"
