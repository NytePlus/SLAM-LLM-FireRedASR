import re
import json
import time
from openai import OpenAI
from tqdm import tqdm

from score import EditDistance, Code

class ASRErrorAgent:
    def __init__(self, client):
        self.client = client
        self.error_taxonomy = {}
        self.known_types = ["无错误", "专有名词/生僻词错误",]
        
        # 统计数据
        self.total_ref_words = 0  # 全局总词数 (分母)
        self.total_err_count = 0
        self.type_metrics = {}    # 各类型错误数: { "类型": 错误总数 }

    def parse_file(self, file_path):
        """解析原始文本文件，提取 uttid, lab, rec"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则匹配每个数据块
        pattern = re.compile(
            r"utt: (?P<uttid>[\w-]+).*?"
            r"lab: (?P<lab>.*?)\n"
            r"rec: (?P<rec>.*?)\n", 
            re.DOTALL
        )
        return [m.groupdict() for m in pattern.finditer(content)]

    def get_api_response(self, batch, big_retries=3):
        original_map = {item['uttid']: item for item in batch}
        batch_text = ""
        for item in batch:
            batch_text += f"ID: {item['uttid']}\nLab: {item['lab']}\nRec: {item['rec']}\n---\n"

        prompt = f"""
        你是一个语音识别(ASR)分析专家。请对比提供的10条数据，总结具体的错误类型。不要分类为替换、删除、插入错误，而是总结出具体的错误类型，比如“专有名词/生僻词错误”、“同音/近音词混淆”，然后给出相应的数据，以及该数据具体为什么属于此类错误
        
        已知类型库(优先从中选择): {self.known_types}
        
        要求以 JSON 格式返回结果，结构严格为: {{"结果": [ {{"uttid": "...", "type": "...", "explanation": "..."}}, ... ]}}
        """

        attempt = 0
        while True:
            try:
                response_str = self.client.chat(prompt + "\n数据：\n" + batch_text)
                
                json_str_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if not json_str_match:
                    raise ValueError("未找到 JSON 结构")
                
                data = json.loads(json_str_match.group())

                if "结果" not in data or not isinstance(data["结果"], list):
                    raise ValueError("JSON 缺少 '结果' 字段或格式非列表")

                validated_results = []
                for entry in data["结果"]:
                    if not (isinstance(entry, dict) and "uttid" in entry and "type" in entry and "explanation" in entry):
                        raise ValueError(f"条目格式缺失: {entry}")
                
                    uttid = entry["uttid"]
                    
                    if uttid in original_map:
                        entry["lab"] = original_map[uttid]["lab"]
                        entry["rec"] = original_map[uttid]["rec"]
                        validated_results.append(entry)
                        del original_map[uttid]
                    else:
                        print(f"警告：API 返回了未知的 uttid: {uttid}，可能产生了幻觉。")

                if len(original_map) > 0:
                    raise ValueError(f"数据不完整，以下 ID 缺失: {list(original_map.keys())}")
                return {"结果": validated_results}

            except Exception as e:
                attempt += 1
                if attempt < big_retries:
                    time.sleep(2)
                else:
                    print(f"已达到很大重试次数({batch[0]['uttid']}...)")
                    time.sleep(60)
        
    def calculate_errors(self, lab, rec):
        """利用 EditDistance 计算单条数据的错误总数 (S+D+I)"""
        ref_tokens = lab.split()
        hyp_tokens = rec.split()
        
        ed = EditDistance()
        try:
            result = ed.align(ref_tokens, hyp_tokens)
            error_count = sum(1 for code in result.codes if code != Code.match)
            return len(ref_tokens), error_count
        except Exception as e:
            print(f"对齐失败: {e}")
            return len(ref_tokens), 0

    def process(self, file_path, batch_size=10):
        all_data = self.parse_file(file_path)
        total = len(all_data)
        print(f"共加载 {total} 条数据，开始处理...")

        for i in tqdm(range(0, total, batch_size)):
            batch = all_data[i:i + batch_size]
            
            result = self.get_api_response(batch)
            if result and "结果" in result:
                for entry in result["结果"]:
                    err_type = entry["type"]
                    uttid = entry["uttid"]
                    explanation = entry["explanation"]

                    ref_count, err_count = self.calculate_errors(entry['lab'], entry['rec'])
                    self.total_ref_words += ref_count
                    self.total_err_count += err_count
                    
                    if err_type not in self.error_taxonomy:
                        self.type_metrics[err_type] = 0
                        self.error_taxonomy[err_type] = []
                        self.known_types.append(err_type)

                    self.type_metrics[err_type] += err_count
                        
                    self.error_taxonomy[err_type].append({
                        "uttid: ": uttid,
                        "rec: ": entry['rec'],
                        "lab: ": entry['lab'],
                        "explanation: ": explanation,
                        "error_count": err_count
                    })

        self.save_report()

    def save_report(self):
        """
        保存包含错误贡献统计和详细案例的完整报告
        """
        # 1. 构建统计概览 (Summary)
        report_data = {
            "overall_stats": {
                "total_ref_words": self.total_ref_words,
                "total_errors": sum(self.type_metrics.values()),
                "total_wer": (sum(self.type_metrics.values()) / self.total_ref_words * 100) if self.total_ref_words > 0 else 0
            },
            "type_contributions": {},
            "details": self.error_taxonomy
        }

        # 2. 计算每种类型的具体百分比贡献
        for err_type, count in self.type_metrics.items():
            contribution_pct = (count / self.total_err_count * 100) if self.total_err_count > 0 else 0
            report_data["type_contributions"][err_type] = f'{round(contribution_pct, 4)}%'

        # 3. 写入文件
        with open('asr_error_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[Done] 报告已更新！当前总 WER: {report_data['overall_stats']['total_wer']:.2f}%")

class ApiClient():
    def __init__(self, model_name="gpt-4.1"):
        # API 配置
        self.api_key = "sk-EI5L8iNGqSlmj4NaCdF522419b8f4c8683DeFc15E2C4Db60"
        self.api_base = "https://api.xi-ai.cn/v1"
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=360)

    def chat(self, prompt, temperature=0.2):
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


agent = ASRErrorAgent(ApiClient())
agent.process("exp/20260121-1702-slidespeech-linear/aispeech_asr_epoch_24_total_step_100000/decode_slidespeech_asr_test_norm_cer")