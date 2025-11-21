import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/aistor/sjtu/hpc_stor01/home/xiyu/models/vicuna-7b-v1.5"   # 本地模型路径

"""
You: 你会说中文吗
Assistant: 是的，我可以说中文。有什么问题需要我回答吗？
"""

# ------------------------------------------------------------
# 1. 加载 tokenizer + model（本地）
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"   # 自动放到 GPU/CPU
)

# Vicuna 使用的 Chat 模板关键 token
def build_prompt(user_input):
    # Vicuna v1.5 官方对话格式（适配 ChatML）
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant.\n"
        "The assistant gives helpful, detailed, and polite answers.\n\n"
        f"USER: {user_input}\n"
        "ASSISTANT:"
    )
    return prompt

# ------------------------------------------------------------
# 2. 简单交互循环
# ------------------------------------------------------------
print("Vicuna-7B-v1.5 Chat (Ctrl+C 退出)")
while True:
    try:
        user_input = input("\nYou: ")
        if user_input.strip() == "":
            continue

        prompt = build_prompt(user_input)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        # 截掉 prompt 部分，只打印模型回答
        print("\nAssistant:", answer.split("ASSISTANT:")[-1].strip())

    except KeyboardInterrupt:
        print("\n退出")
        break
