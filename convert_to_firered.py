import os
import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', help='source checkpoint dir')
    parser.add_argument('save_dir', help='output dir')
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir
    ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    save_dir = args.save_dir
    save_full_path = os.path.join(save_dir, "model.pth.tar")
    # 构造 args 结构
    args = argparse.Namespace(
        encoder_path=save_dir,
        llm_dir=save_dir,
        freeze_encoder=False,
        freeze_llm=0,
        use_flash_attn=0,
        use_fp16=0,
        use_lora=1,
        encoder_downsample_rate=2,
    )
    # 构造 checkpoint
    checkpoint = {
        "model_state_dict": ckpt_dict,  # 模型权重
        "args": args  # 额外的训练参数
    }

    # 保存到 .pth.tar
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    torch.save(checkpoint, save_full_path)
    print(f"Completed.You can find the chekcpoint in {save_dir}")

if __name__ == "__main__":
    main()
