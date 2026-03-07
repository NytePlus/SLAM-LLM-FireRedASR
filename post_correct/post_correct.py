import torch
import math
import argparse
import logging
import sys
import json
import re
import os
import time
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from whisper_normalizer.english import EnglishTextNormalizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

english_normalizer = EnglishTextNormalizer()

MAPPING_FILE = "/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech/test_oracle_v1/id2id"
id_map = {}
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        a, b = line.split()
        id_map[a] = b

class Paragraph():
    def __init__(self, video):
        utt = id_map[video["youtube_channel"]]
        self.idx = {}
        self.segments = []
        for s in video["segments"]:
            self.segments.append(s["txt_raw"])
            idx = int(s["uttid"].split('-')[-1])

            uttid = f'{utt}-{idx :04d}'
            self.idx[uttid] = len(self.segments) - 1

    def part(self, uttid, range = 40):
        end = self.idx[uttid]
        begin = max(0, end - range)
        return " ".join(self.segments[begin : end]), self.segments[end]

paragraph_dict = {}

class BasePostCorrector():
    def build_prompt(self, context, hyp):
        return f"""
### Context:
{context}
### Requirements:
1. **Word-for-Word Mapping**: Only fix misrecognized words (homophones, technical terms). 
2. **No Paraphrasing**: Do not replace phrases with synonyms (e.g., do not change "at this time" to "currently").
3. **Preserve Disfluencies**: DO NOT remove repeated words (e.g., if you see "for for for", keep "for for for").
4. **Maintain Structure**: The sentence structure, word order, and total meaning must remain identical to the Raw Transcription.
5. **JSON Output**: Strictly output JSON format.
### Raw Transcription to Correct:
{hyp}
### Desired JSON Format:
{{
  "Corrected Transcription": "only corrected Raw Transcription here, do not include Context",
}}
"""

    def extract_from_pred(self, pred_text):
        try:
            match = re.search(r'\{[\s\S]*?\}', pred_text)
            if not match:
                raise ValueError("No JSON object found")

            json_str = match.group(0).replace('\n', ' ')
            data = json.loads(json_str)
            return data["Corrected Transcription"].replace('\n', ' ')
        except Exception as e:
            raise ValueError(f"json error: {e}")

class LocalPostCorrector(BasePostCorrector):
    def __init__(self, model_name = "/models/Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left", trust_remote_code=True)
        self.device="npu:0"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

    def correct_batch(self, contexts, hyps):
        """
        contexts: List[str]
        hyps: List[str]
        """
        prompts = []
        for ctx, h in zip(contexts, hyps):
            formatted_prompt = self.build_prompt(ctx, h)
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(e)
                text = f"A chat between a curious user and an artificial intelligence assistant. USER: {formatted_prompt} ASSISTANT:"
            prompts.append(text)

        model_inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        results = []
        for res in decoded_outputs:
            try:
                results.append(self.extract_from_pred(res))
            except:
                results.append(res.strip().replace('\n', ' '))
        return results

class APIPostCorrector(BasePostCorrector):
    def __init__(self, model_name="gpt-4.1"):
        # API 配置
        self.api_key = "sk-EI5L8iNGqSlmj4NaCdF522419b8f4c8683DeFc15E2C4Db60"
        self.api_base = "https://api.xi-ai.cn/v1"
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=360)

    def run_llm(self, prompt, temperature=0.2):
        """核心调用逻辑，包含重试机制"""
        messages = [
            {"role": "system", "content": "You are a professional ASR post-processor."},
            {"role": "user", "content": prompt}
        ]
        
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024,
                    response_format={"type": "json_object"} # 强制要求 JSON 格式输出
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries:
                    print(f"API Error: {e}, Retrying {attempt}/{max_retries}...")
                    time.sleep(2)
                else:
                    print("Max retries reached. Failing.")
                    return None

    def correct_batch(self, contexts, hyps):
        """
        contexts: List[str] - 领域上下文信息
        hyps: List[str] - ASR 原始转录文本
        """
        results = []

        for context, hyp in zip(contexts, hyps):
            formatted_prompt = self.build_prompt(context, hyp)
            
            # 执行 API 调用
            response_json = self.run_llm(formatted_prompt)
            
            try:
                results.append(self.extract_from_pred(response_json))
            except:
                results.append('[ERROR]' + hyp)
                
        return results


def main(args):
    if args.context is not None:
        with open(args.context, "r") as f:
            data = json.load(f)
            for v in data["videos"]:
                utt = id_map[v["youtube_channel"]]
                paragraph_dict[utt] = Paragraph(v)

    refs = {}
    with open(args.multitask, "r") as f:
        for line in f:
            data = json.loads(line)
            uttid, ref = data['key'], data['target'].upper()
            biasing_words = set(item.strip() for item in data['hotword'].strip().split(","))
            if args.norm:
                ref = english_normalizer(ref).upper()
            refs[uttid] = {"text": ref, "biasing_words": biasing_words}

    logger.info("Loaded %d reference utts from %s", len(refs), args.multitask)

    hyps = {}
    with open(args.pred, "r") as f:
        for line in f:
            ary = line.strip().split(" ", maxsplit=1)
            # May have empty hypo
            if len(ary) >= 2:
                uttid, hyp = ary[0], ary[1].upper()
            else:
                uttid, hyp = ary[0], ""
            
            if args.norm:
                hyp = english_normalizer(hyp).upper()
            hyps[uttid] = hyp
    logger.info("Loaded %d hypothesis utts from %s", len(hyps), args.pred)

    if not args.lenient:
        for uttid in refs:
            if uttid in hyps:
                continue
            raise ValueError(
                f"{uttid} missing in pred! Set `--lenient` flag to ignore this error."
            )
    
    processed_ids = set()
    if os.path.exists(args.corr):
        with open(args.corr, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", maxsplit=1)
                if parts:
                    processed_ids.add(parts[0])
    logger.info("Found %d already processed utts in %s", len(processed_ids), args.corr)

    corrector = APIPostCorrector()
    
    # 2. 收集任务时过滤已处理的 ID
    tasks = []
    for uttid in refs:
        if uttid in processed_ids: continue
        if uttid not in hyps: continue
        
        hyp_text = hyps[uttid].lower()
        utt = '-'.join(uttid.split('-')[:-1])

        if hyp_text == '':
            with open(args.corr, "a", encoding="utf-8") as fout:
                fout.write(f"{uttid} {hyp_text}\n")
            continue
        
        if args.context is not None:
            p = paragraph_dict[utt]
            context, _ = p.part(uttid, range=10)
            # 计算 prompt 长度用于动态 batch
            prompt_len = len(corrector.build_prompt(context, hyp_text))
            tasks.append({
                "uttid": uttid, 
                "context": context, 
                "hyp": hyp_text, 
                "length": prompt_len
            })
            if uttid == '1236-27507-0043':
                print(f'context: {context}\nhyp_text: {hyp_text}')
                return 0
    
    # 按照长度升序排序
    tasks.sort(key=lambda x: x["length"], reverse=False)

    THRESHOLD = 1
    current_idx = 0
    total_tasks = len(tasks)
    pbar = tqdm(total=total_tasks, desc="Dynamic Batch Correcting")

    with open(args.corr, "a", encoding="utf-8") as fout:
        while current_idx < total_tasks:
            batch = []
            current_batch_length = 0
            
            while current_idx < total_tasks:
                task = tasks[current_idx]
                if current_batch_length + task["length"] > THRESHOLD and len(batch) > 0:
                    break
                
                batch.append(task)
                current_batch_length += task["length"]
                current_idx += 1

            batch_contexts = [t["context"] for t in batch]
            batch_hyps = [t["hyp"] for t in batch]
            batch_uttids = [t["uttid"] for t in batch]

            # 执行批量推理
            corrected_texts = corrector.correct_batch(batch_contexts, batch_hyps)

            for uttid, corr_hyp in zip(batch_uttids, corrected_texts):
                fout.write(f"{uttid} {corr_hyp}\n")
            
            pbar.update(len(batch))

    pbar.close()
        

if __name__ == '__main__':
    desc = "Compute ppl with qwen2-7b. Results are output to stdout."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to space-separated _pred file. First column is utterance ID. "
        "Second column is reference text.",
    )
    parser.add_argument(
        "--corr",
        required=True,
        help="Path to space-separated _corr file. First column is utterance ID. "
        "Second column is reference text.",
    )
    parser.add_argument(
        "--multitask",
        required=True,
        help="Path to multitask files. It is a jsonl file, each line a train/test data."
        "Has key, task, target, path, hotword. Use key, target and hotword.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="If set, hyps doesn't have to cover all of refs.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="If set, use whisper_normalizer to normalize refs and hyps",
    )
    parser.add_argument(
        "--context",
    )
    args = parser.parse_args()

    main(args)