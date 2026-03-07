from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/models/vicuna-7b-v1.5"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="npu:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

print("--- Chatbot 已启动 (输入 'exit' 或 'quit' 退出) ---")

while True:
    user_input = input("\nUser: ")
    
    if user_input.lower() in ["exit", "quit"]:
        break
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    messages.append({"role": "user", "content": user_input})

    # 3. 构建模板输入
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        def format_vicuna_prompt(messages):
            prompt = "A chat between a curious user and an artificial intelligence assistant. " \
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"USER: {msg['content']} "
                elif msg["role"] == "assistant":
                    prompt += f"ASSISTANT: {msg['content']}</s>"
            
            prompt += "ASSISTANT:"
            return prompt

        text = format_vicuna_prompt(messages)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 4. 生成回答
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=300,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.9
    )
    
    # 5. 切片获取生成的回复部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 6. 打印结果并存入历史
    print(f"Assistant: {response}")
    # messages.append({"role": "assistant", "content": response})

"""
### Context:
And we found that in some of these very specific diseases as well, people seem to restrict their physical activity. So this is one study that we did here at Wilmer, where we found that as compared to people with normal sight, those with age related macular degeneration did thirty-five percent less physical activity and there was a dose response. So for every line of visual acuity, worse they were, they did about twelve percent less, they spend twelve percent less time in moderate or vigorous physical activity. Um, we also saw kind of the same thing with diabetic retinopathy, although we actually flipped the analysis in this way, so we saw that that people who were diabetic were more physically active were actually less likely to have diabetic retinopathy. Uh and even, uh, in females, even just an extra ten minutes of moderate or vigorous activity a day really decrease the chance that people would be uh would have moderate or severe retinopathy by three quarters. And these effects were independent of those things that, you know, affect diabetese as well, such as disease duration, blood pressure or (~). So of course not everything affects vision through visual acuity, so visual acuity is what you go to and see when you go to the doctor, so you read down the chart with increasingly small letters. And they give you a number, so twenty-fifty means that you see at twenty feet what a normal person would see at fifty feet. In other words, you have to be closer to it to see it and a normal person would be twenty-twenty, so they see a twenty feet what a normal person to see at twenty feet. And there is other types of vision loss as well, and I'll talk to you about one of those in (~) in particular glaucoma, which actually affects your field of vision, so it's actually tested with uh something called a perimeter, so it measures the your the perimeter of your, the perimetry of your vision. And it sees how light, ah, how bright they have to show you lights for you to see them, so areas that are shown in black over here means that the person had to be shown a light brighter in that area for, for them to see it, meaning that they had damage to that portion of their visual field, as sure, and it's see graded on a decibel scale.
### Requirements:
1. **Word-for-Word Mapping**: Only fix misrecognized words (homophones, technical terms). 
2. **No Paraphrasing**: Do not replace phrases with synonyms (e.g., do not change "at this time" to "currently").
3. **Preserve Disfluencies**: DO NOT remove repeated words (e.g., if you see "for for for", keep "for for for").
4. **Maintain Structure**: The sentence structure, word order, and total meaning must remain identical to the Raw Transcription.
5. **JSON Output**: Strictly output JSON format.
### Raw Transcription to Correct:
from 0 to -30
### Desired JSON Format:
{{
  "Corrected Transcription": "only corrected Raw Transcription here, do not include Context",
}}
"""
