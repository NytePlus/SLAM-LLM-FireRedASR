import torch
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 环境变量与模型加载
model_name = os.environ.get('MODEL_PATH', "/models/vicuna-7b-v1.5")
device = "npu:0" if torch.npu.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
model.eval()

class Color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

print(f"{Color.BOLD}--- 交互式 PPL 测试与生成工具 ---{Color.END}")
context = ""

while True:
    user_input = input(f"{Color.BLUE}|{Color.END} ").strip()

    if user_input.lower() == "exit": break
    if user_input.lower() == "c":
        context = ""; print(f"{Color.GREEN}Context cleared.{Color.END}"); continue

    # 情况 A: 用户提供了输入，计算该输入的 PPL 并存入 Context
    if user_input:
        full_text = context + user_input
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        # 计算前缀长度以屏蔽 Loss
        prefix_len = tokenizer(context, return_tensors="pt").input_ids.shape[1] if context else 0
        labels = inputs.input_ids.clone()
        labels[:, :prefix_len] = -100 

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            ppl = math.exp(loss.item()) if not torch.isnan(loss) else float('inf')
            print(f"{Color.RED}(PPL: {ppl:.2f}){Color.END}")
        
        context += user_input # 更新上下文

    # 情况 B: 输入为空，模型基于当前 Context 开始生成
    else:
        if not context:
            print(f"{Color.YELLOW}Context is empty, please input something first.{Color.END}")
            continue
            
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.no_grad():
            # 生成新内容
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                streamer=streamer,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 将生成的文本同步回 Context
            new_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            context += new_text