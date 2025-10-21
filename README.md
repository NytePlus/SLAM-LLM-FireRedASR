# What it is

We extract code from [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) and [FireredASR](https://github.com/FireRedTeam/FireRedASR) to support the fine-tuning of the FireredASR-LLM model. 

# Finetuning

```bash
./scripts/finetune_deepspeed.sh
```

# Inference

```bash
./scripts/decode_deepspeed.sh
```

# Features

**Large-scale industrial dataset**: adopt an iterative dataset and dynamic batching strategy.

**On-the-fly data augmentation**: support spectral augmentaton, reverberation and noise adding.

**Training on NPU**: multi-machine multi-NPU training using deepspeed. If GPU training is required, simply replace all `.npu` with `.cuda` in the code.
