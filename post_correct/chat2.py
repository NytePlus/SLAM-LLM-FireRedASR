from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import os

model_name = os.environ.get('MODEL_PATH', "/models/vicuna-7b-v1.5")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="npu:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

print("--- Chatbot 已启动 (输入 'exit' 或 'quit' 退出) ---")

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    LIGHT_GRAY = "\033[90m"
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

while True:
    role_description = input(f"{Color.BLUE}{Color.BOLD}System:{Color.END} ")
    user_input = input(f"{Color.GREEN}{Color.BOLD}User:{Color.END} ")
    
    if user_input.lower() in ["exit", "quit"]:
        break
    print(f"{Color.YELLOW}{Color.BOLD}Assistant: {Color.END}[", end='')
    forced_prefix = input("") # 强制引导句
    eval_sentence = input(f"{Color.RED}sentence to eval PPL: {Color.END}")

    def format_vicuna_with_prefix(messages, prefix):
        prompt = f"{messages[0]['content']} "
        for msg in messages[1:]:
            if msg["role"] == "user":
                prompt += f"USER: {msg['content']} "
        prompt += f"ASSISTANT: {prefix}" 
        return prompt

    system_content = role_description if role_description else "You are a helpful assistant."
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_input}]
    
    try:
        messages.append({"role": "assistant", "content": forced_prefix})
        base_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ).strip()[:-10]
    except:
        base_text = format_vicuna_with_prefix(messages, forced_prefix)
    
    # --- PPL 计算逻辑 ---
    ppl_display = ""
    top_n_output = ""
    if eval_sentence.strip():
        # 1. 编码全文本与前缀
        full_eval_text = base_text + eval_sentence
        inputs = tokenizer(full_eval_text, return_tensors="pt").to(model.device)
        prefix_ids = tokenizer(base_text, return_tensors="pt").input_ids
        prefix_len = prefix_ids.shape[1]
        
        # 2. 构造 Labels (屏蔽前缀)
        labels = inputs.input_ids.clone()
        labels[:, :prefix_len] = -100
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            ppl = math.exp(loss.item())
            ppl_display = f"{Color.RED}(PPL: {ppl:.2f}){Color.END}"
            
            # 3. 提取 Top-N (针对 eval_sentence 部分)
            logits = outputs.logits # [1, seq_len, vocab_size]
            # 对齐 Shift: logits[i] 预测的是 input_ids[i+1]
            # 我们只需要 eval_sentence 对应的部分
            shift_logits = logits[..., prefix_len-1:-1, :].contiguous()
            shift_labels = labels[..., prefix_len:].contiguous()
            
            # 找到非 mask 的位置进行 topk
            target_logits = shift_logits[0] # [eval_len, vocab_size]
            target_ids = shift_labels[0]    # [eval_len]
            
            top_k = 3 # 你可以修改展示前几个
            probs = torch.softmax(target_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            # 格式化 Top-N 字符串
            lines = []
            for i in range(target_ids.size(0)):
                actual_token = tokenizer.decode([target_ids[i]])
                preds = []
                for j in range(top_k):
                    p_token = tokenizer.decode([top_indices[i, j]])
                    p_val = top_probs[i, j].item()
                    preds.append(f"'{p_token}'({p_val:.1%})")
                
                # 拼接：目标词 -> 候选1 / 候选2
                lines.append(f"  {actual_token} -> {' / '.join(preds)}")
            
            top_n_output = f"{Color.LIGHT_GRAY}Top-N Analysis:\n" + "\n".join(lines) + f"{Color.END}"

    # --- 生成逻辑 ---
    model_inputs = tokenizer([base_text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    
    input_length = model_inputs.input_ids.shape[1]
    new_tokens = generated_ids[0][input_length:]
    content_continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 最终输出显示
    print(f"{ppl_display}]{Color.BOLD}{content_continuation}{Color.END}\n")
    if top_n_output:
        print(top_n_output + "\n")

"""
And thank you very much good evening everybody and a warm welcome to our next presentation. My name is Katharina Morlang and together with my colleagues hiker hoods and patrick young, please give me your hands.
Yes and we represent uh the german sports youth and um, together we drive the project I coach kids forward germany.
The german sports youth represents all youth sports organizations in germany and we are very happy to be a partner of the follow-up project I coach kids class.
For this reason, we are also participating in this huge conference and we are happy to be part of it.
The german sports youth developed together with the (~) university of erlangan and bavaria, a concept for developing the personality and the team in sports.
For the past five years, um, we also have been supporting coaches in germany, especially together with the german olympic sports federation.
We want that um, coaches in germany um receive a good education their children and young people receive the best possible support of their coaches and that they can grow up healthy and are made strong for life.
Martin Muchem, research assistant at the university of erlangen, an important key partner for us.
Um, will now explain to us the concept of personality in team development in sport which clearly uses the sport itself and targets methods in sport to develop the team, make the team and the athlete more successful.
Martin um, if you want and if you are ready you can start now and we will discuss later.
            

Wir wollen, dass, äh, Trainer in Deutschland, äh, eine gute Ausbildung erhalten, ihre Kinder und Jugendlichen die bestmögliche Unterstützung durch ihre Trainer bekommen und dass sie gesund aufwachsen und stark fürs Leben gemacht werden.
我们希望德国的教练能接受良好的教育，他们的孩子和年轻人能得到教练最好的支持，他们能健康成长，终身强壮。
我们想要，那个，嗯，在德国的教练，嗯，接受良好的教育，他们的孩子和年轻人得到他们教练最好的可能支持，并且他们能健康成长，并为生活变得强壮。

KIDS KATHARINA, KATHARINA MORLANG, DROPOUT KIDS, COACH DSJ, DSJ DAY, KATHARINA, MORLANG, DROPOUT, MORLANG INTERNATIONAL, PREVENTING DROPOUT, DSJ, KIDS, DAY PREVENTING, DAY, COACH, CONFERENCE COACH, PREVENTING, CONFERENCE, INTERNATIONAL CONFERENCE, INTERNATIONAL
"""

"""
System: Transcript the sentence to English
User:  我们想要，那个，嗯，在德国的教练，嗯，接受良好的教育，他们的孩子和年轻人得到他们教练最好的可能支持，并且他们能健康成长，并为生活变得强壮。
"""