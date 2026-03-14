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
    LIGHT_GRAY = '\033[90m' # 添加浅灰色

print(f"{Color.BOLD}--- 交互式 PPL 测试与 Top-N 分析工具 ---{Color.END}")
print(f"指令: {Color.BLUE}直接回车{Color.END}让模型生成 | {Color.BLUE}'c'{Color.END}清空上下文 | {Color.BLUE}'exit'{Color.END}退出")
context = ""

while True:
    user_input = input(f"\n{Color.BLUE}Input/Prefix >>{Color.END} ").strip()

    if user_input.lower() == "exit": break
    if user_input.lower() == "c":
        context = ""; print(f"{Color.GREEN}Context cleared.{Color.END}"); continue

    # 情况 A: 计算输入部分的 PPL 和 Top-N
    if user_input:
        full_text = context + user_input
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        # 计算前缀长度
        prefix_ids = tokenizer(context, return_tensors="pt").input_ids if context else torch.tensor([[]])
        prefix_len = prefix_ids.shape[1]
        
        labels = inputs.input_ids.clone()
        labels[:, :prefix_len] = -100 

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            ppl = math.exp(loss.item()) if not torch.isnan(loss) else float('inf')
            
            # --- Top-N 逻辑 ---
            logits = outputs.logits  # [1, seq_len, vocab_size]
            # Shift 对齐: logits 的 t 位置预测的是 inputs.input_ids 的 t+1 位置
            # 我们只关心 user_input 对应的部分
            shift_logits = logits[..., max(0, prefix_len - 1) : -1, :].contiguous()
            shift_labels = labels[..., prefix_len : ].contiguous()
            
            target_ids = shift_labels[0]
            target_logits = shift_logits[0]
            
            top_k = 5
            probs = torch.softmax(target_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            top_n_results = []
            for i in range(1, target_ids.size(0)):
                actual_token = tokenizer.decode([target_ids[i]])
                step_top_n = []
                for j in range(top_k):
                    p_token = tokenizer.decode([top_indices[i - 1, j]])
                    p_val = top_probs[i - 1, j].item()
                    step_top_n.append((p_token, p_val))
                
                top_n_results.append({
                    "target": actual_token,
                    "top_n": step_top_n
                })

            # --- 打印结果 ---
            print(f"{Color.RED}(PPL: {ppl:.2f}){Color.END}")
            
            # 按照你要求的格式显示
            top_n_display = '\n'.join(
                f"{res['target']}: {' '.join(f'({i[0]} {i[1] : 0.4f})' for i in res['top_n'])}" 
                for res in top_n_results
            ) 
            print(f"{Color.LIGHT_GRAY}TOP_N:\n{top_n_display}{Color.END}")
        
        context += user_input 

    # 情况 B: 自由生成
    else:
        if not context:
            print(f"{Color.YELLOW}Context is empty, please input something first.{Color.END}")
            continue
            
        print(f"{Color.YELLOW}Assistant: {Color.END}", end='')
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                streamer=streamer,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            new_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            context += new_text
            print() # 换行