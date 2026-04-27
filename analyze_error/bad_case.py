import re
import json
import time
import yaml
import random
import os
from openai import OpenAI
from tqdm import tqdm
from collections import Counter, defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from score.score import EditDistance, Code

class ApiClient():
    def __init__(self, model_name="gpt-4.1"):
        # API 配置
        self.api_key = os.environ.get('OPENAI_API_KEY', '')
        self.api_base = "https://api.xi-ai.cn/v1"
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=360)

    def chat(self, prompt, temperature=0.2):
        """核心调用逻辑，包含重试机制"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
        
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2048,
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

class ErrorStatic:
    def __init__(self):
        self.block_substitutions = Counter()
        self.total_error_blocks = 0
        self.block_substitution_err = Counter()
        self.total_error_count = 0

    def update(self, lab, rec):
        ref_tokens = lab.split()
        hyp_tokens = rec.split()
        
        ed = EditDistance()
        try:
            result = ed.align(ref_tokens, hyp_tokens)
            
            # 状态机：记录当前连续错误的块
            current_ref_block = []
            current_hyp_block = []
            
            for i, code in enumerate(result.codes):
                if code == Code.match:
                    self._commit_block(current_ref_block, current_hyp_block)
                    current_ref_block = []
                    current_hyp_block = []
                elif code == Code.insertion:
                    hyp_idx = result.hyps[i]
                    current_hyp_block.append(hyp_tokens[hyp_idx])
                elif code == Code.deletion:
                    ref_idx = result.refs[i]
                    current_ref_block.append(ref_tokens[ref_idx])
                else:
                    hyp_idx = result.hyps[i]
                    current_hyp_block.append(hyp_tokens[hyp_idx])
                    ref_idx = result.refs[i]
                    current_ref_block.append(ref_tokens[ref_idx])
            
            # 循环结束后，处理最后一组可能存在的错误块
            self._commit_block(current_ref_block, current_hyp_block)
                    
        except Exception as e:
            print(f"[ErrorStatic] 块对齐分析失败: {e}")

    def _commit_block(self, ref_list, hyp_list):
        """将收集到的错误块合并并存入统计"""
        if not ref_list and not hyp_list:
            return
            
        ref_words = [w.lower() for w in ref_list]
        hyp_words = [w.lower() for w in hyp_list]
        
        best_match = None # (ref_start, ref_end, hyp_start, hyp_end)

        # 1. 滑动窗口寻找去空格匹配的最长子段
        # 遍历 ref 的所有子区间
        for r_start in range(len(ref_list)):
            for r_end in range(r_start + 1, len(ref_list) + 1):
                ref_sub_str = "".join(ref_words[r_start:r_end])
                if not ref_sub_str: continue
                
                # 遍历 hyp 的所有子区间
                for h_start in range(len(hyp_list)):
                    for h_end in range(h_start + 1, len(hyp_list) + 1):
                        hyp_sub_str = "".join(hyp_words[h_start:h_end])
                        
                        # 如果去空格匹配且确实有空格差异（或者至少不是完全一样的字符串）记录最长的匹配（或者第一个发现的匹配）
                        if ref_sub_str == hyp_sub_str:
                            if best_match is None or (r_end - r_start) > (best_match[1] - best_match[0]):
                                best_match = (r_start, r_end, h_start, h_end)

        # 2. 根据锚点进行三段式提交
        if best_match:
            r_s, r_e, h_s, h_e = best_match
            
            prefix_ref = ref_list[:r_s]
            prefix_hyp = hyp_list[:h_s]
            if prefix_ref or prefix_hyp:
                self._save(" ".join(prefix_ref), " ".join(prefix_hyp))
            
            self._save(" ".join(ref_list[r_s:r_e]), " ".join(hyp_list[h_s:h_e]))
            
            suffix_ref = ref_list[r_e:]
            suffix_hyp = hyp_list[h_e:]
            if suffix_ref or suffix_hyp:
                self._commit_block(suffix_ref, suffix_hyp)
        else:
            self._save(" ".join(ref_list), " ".join(hyp_list))

    def _save(self, r_str, h_str):
        r_out = r_str if r_str.strip() else ""
        h_out = h_str if h_str.strip() else ""
        
        ref_tokens = r_out.split()
        hyp_tokens = h_out.split()
        
        ed = EditDistance()
        try:
            res = ed.align(ref_tokens, hyp_tokens)
            actual_error_weight = sum(1 for code in res.codes if code != Code.match)
        except:
            actual_error_weight = max(len(ref_tokens), len(hyp_tokens))

        pair = f"{r_out} -> {h_out}"
        self.block_substitutions[pair] += 1
        self.total_error_blocks += 1
        self.block_substitution_err[pair] += actual_error_weight
        self.total_error_count += actual_error_weight

    def get_summary(self, top_n=None):
        """
        获取统计摘要。
        :param top_n: 如果指定数字（如 500），则只取频率最高的 N 个错误对。
                      如果不指定，则取全部数据。
        """
        # 获取排序后的错误对
        if top_n:
            raise NotImplementedError()
            selected_examples = dict(self.block_substitution_err.most_common(top_n))
            subset_count = sum(selected_examples.values())
        else:
            count_dict = dict(self.block_substitutions.most_common())
            err_count_dict = dict(self.block_substitution_err.most_common())
            subset_count = self.total_error_blocks
            err_count = self.total_error_count

        return [{
            "type": "ALL",
            "count": subset_count,
            "rate": "100.00%",
            "err_count": err_count,
            "err_rate": "100.00%",
            "count_dict": count_dict,
            "err_count_dict": err_count_dict
        }]

    def save_to_file(self, filename):
        """保存为标准的 initial_data 格式"""
        data = self.get_summary()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Done] 统计数据已按 initial_data 格式保存至 {filename}")

    @staticmethod
    def load_from_file(filename):
        """
        从文件中加载 initial_data，直接用于 RecursiveErrorAgent
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Load] 已从 {filename} 加载数据，包含 {data[0].get('count', 0)} 个错误块")
        return data

class BlockErrorProcessor:
    def __init__(self):
        self.word_stats = ErrorStatic()

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

    def process(self, file_path):
        all_data = self.parse_file(file_path)
        total = len(all_data)
        print(f"共加载 {total} 条数据，开始处理并进行词级统计...")

        for entry in tqdm(all_data):
            self.word_stats.update(entry['lab'], entry['rec'])

        self.word_stats.save_to_file("analyze_error/initial_error_data.json")

class HardCaseStatic:
    def __init__(self):
        self.hard_case = {}
        self._lock = Lock()

    def add(self, p):
        with self._lock:
            self.hard_case[p] = self.hard_case.get(p, 0) + 1

    def clear(self, p):
        with self._lock:
            self.hard_case.pop(p, None)

    def reset(self):
        with self._lock:
            self.hard_case = {}

    def get_all(self):
        with self._lock:
            return self.hard_case.copy()

class RecursiveErrorAgent:
    def __init__(self, client, config_path="config.yaml"):
        self.client = client
        self.results_lock = Lock()
        self.hard_case = HardCaseStatic()
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
            self.root_key = list(raw_config.keys())[0]
            self.structure = raw_config[self.root_key]

    def _parse_label(self, label_str):
        """
        扩展解析逻辑：支持提取规则函数名
        返回: (名称, 说明, 函数名)
        """
        match = re.match(r'^(.*?)\((.*?)\)$', label_str)
        if match:
            name = match.group(1).strip()
            content = match.group(2).strip()
            # 检查是否有 f: 标记
            if content.startswith("f:"):
                return name, "", content[2:].strip() # 只有规则，无描述
            return name, content, None # 有描述，无规则
        return label_str.strip(), "", None

    def _is_alphabet_error(self, pair):
        ref, hyp = pair.split(" -> ")
        r_clean = ref.replace(" ", "").lower()
        h_clean = hyp.replace(" ", "").lower()
        return r_clean == h_clean and r_clean != ""
    
    def _is_nonexist_words(self, pair):
        ref, hyp = pair.split(" -> ")

    def process_node(self, node_data, config_list):
        if not config_list or not node_data["count_dict"]:
            return [node_data]

        count_dict = node_data["count_dict"].copy()
        error_count_dict = node_data["err_count_dict"].copy()
        current_rate_scalar = float(node_data["rate"].strip("%")) / 100
        current_rate_scalar_err = float(node_data["err_rate"].strip("%")) / 100
        parent_total = node_data["count"]
        parent_total_err = node_data['err_count']
        
        labels = []
        descriptions = []
        sub_configs = {}
        rule_map = {}

        for item in config_list:
            has_child = isinstance(item, dict)
            full_key = list(item.keys())[0] if has_child else item
            name, desc, func_name = self._parse_label(full_key)
            
            labels.append(name)
            sub_configs[name] = item[full_key] if has_child else None
            if func_name:
                rule_map[name] = func_name
            if desc:
                descriptions.append(f"- {name}: {desc}")

        split_results = {l: {"count": 0, "err_count": 0, "count_dict": {}} for l in labels}
        remaining_examples = {}

        for pair, count in count_dict.items():
            matched = False
            for label in labels:
                if label in rule_map:
                    func = getattr(self, rule_map[label], None)
                    if func and func(pair):
                        split_results[label]["count"] += count
                        split_results[label]["err_count"] += error_count_dict[pair]
                        split_results[label]["count_dict"][pair] = count
                        matched = True
                        break # 匹配到第一个规则即停止
            
            if not matched:
                remaining_examples[pair] = count

        # 3. 剩余数据交给 LLM 分类
        if remaining_examples:
            llm_labels = [l for l in labels if l not in rule_map or not rule_map[l]]
            # 如果 YAML 里全是规则且都没匹配上，兜底给最后一个标签或“其他”
            target_labels = llm_labels if llm_labels else labels 
            
            llm_res = self._llm_classify_batch_parallel(remaining_examples, error_count_dict, target_labels, descriptions)
            
            # 合并 LLM 结果到 split_results
            for l, res in llm_res.items():
                split_results[l]["count"] += res["count"]
                split_results[l]["err_count"] += res["err_count"]
                split_results[l]["count_dict"].update(res["examples"])

        # 4. 递归处理子节点
        final_sub_nodes = []
        for label in labels:
            res = split_results[label]
            if res["count"] > 0:
                child_node = {
                    "type": label,
                    "count": res["count"],
                    "rate": f"{(res['count'] / parent_total * 100 * current_rate_scalar):.2f}%",
                    "err_count": res['err_count'],
                    "err_rate": f"{(res['err_count'] / parent_total_err * 100 * current_rate_scalar_err):.2f}%",
                    "count_dict": res["count_dict"],
                    "err_count_dict": error_count_dict,
                }
                if label in sub_configs and sub_configs[label]:
                    final_sub_nodes.extend(self.process_node(child_node, sub_configs[label]))
                else:
                    del child_node['err_count_dict']
                    final_sub_nodes.append(child_node)

        return final_sub_nodes

    def _llm_classify_batch(self, examples, labels, descriptions, batch_size=10):
        """分批请求 LLM，强制要求分类到指定标签，失败则无限重试"""
        raise NotImplementedError()
        results = {l: {"count": 0, "examples": {}} for l in labels}
        
        desc_text = "\n".join(descriptions)
        items = list(examples.items())

        for i in tqdm(range(0, len(items), batch_size)):
            batch = items[i:i + batch_size]
            
            temp_storage = self._process_single_batch(batch, labels, desc_text)
            for p, t in temp_storage:
                results[t]["count"] += examples[p]
                results[t]["examples"][p] = examples[p]

        return results
    
    def _process_single_batch(self, batch, labels, desc_text):
        """处理单个批次的逻辑，包含无限重试"""
        batch_success = False
        retry_count = 0

        def fix_llm_pair(pair_str):
            if not pair_str or "->" not in pair_str:
                return pair_str
            match = re.match(r'^(.*?)\s*->\s*(.*)$', pair_str.strip())
            if match:
                ref = match.group(1).strip()
                hyp = match.group(2).strip()
                return f"{ref} -> {hyp}"
            
            return pair_str
        
        while not batch_success:
            try:
                random.shuffle(batch)
                batch_text = "\n".join([f"{p}" for p, c in batch])
                retry_warning = "" if retry_count == 0 else f"\n请确保返回的 'type' 必须严格属于：{labels}"
                # 随着重试次数增加，稍微提高随机性
                temp = 0.4 if retry_count < 3 else 0.8
                
                prompt = f"""你是一个 ASR 错误分析专家。请将错误对归类。{retry_warning}
分类标准说明：
{desc_text}
【强制要求】可选分类名称必须且只能是：{labels}
输出 JSON 格式要求：{{"结果": [{{"pair": "Ref -> Hyp", "type": "分类名称"}}]}}
严格保留其完整 Pair，严禁修改 Pair 中的内容或标点。
待处理数据：
{batch_text}
"""
                response = self.client.chat(prompt, temperature=temp)
                
                # 解析 JSON
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if not match:
                    raise ValueError("未检测到有效 JSON")
                
                res_json = json.loads(match.group())
                batch_results = res_json.get("结果", [])
                
                # 校验
                temp_storage = []
                for item in batch_results:
                    p = fix_llm_pair(item.get("pair"))
                    t = item.get("type")
                    if len(p) > 150: # 大模型对无限重复的句子，复制能力很差
                        mp = None
                        for b in batch:
                            if p[:150] == b[0][:150]:
                                mp = b[0]
                                break
                        if mp is None:
                            self.hard_case.add(p)
                            raise ValueError(f"非法Pair '{p}'")
                        p = mp
                    else:
                        if p not in [b[0] for b in batch]:
                            self.hard_case[p] += 1
                            self.hard_case.add(p)
                            raise ValueError(f"非法Pair '{p}'")
                    if t not in labels:
                        self.hard_case.add(p)
                        raise ValueError(f"标签 '{t}' 不在可选列表")
                    temp_storage.append((p, t))

                # 检查漏选
                processed_pairs = {x[0] for x in temp_storage}
                for p, _ in batch:
                    if p not in processed_pairs:
                        self.hard_case.add(p)
                        raise ValueError(f"漏分类: {p}")

                return temp_storage  # 返回成功的分类结果
                
            except Exception as e:
                retry_count += 1
                # 打印信息时带上线程 ID 方便调试
                print(f"失败: {e}, 第 {retry_count} 次尝试...")
                time.sleep(0.5)

    def _llm_classify_batch_parallel(self, examples, error_count_dict, labels, descriptions, batch_size=10, max_workers=20):
        # 初始化结果字典
        final_results = {l: {"count": 0, "err_count": 0, "examples": {}} for l in labels}
        desc_text = "\n".join(descriptions)
        items = list(examples.items())
        
        # 将数据按 batch_size 切分
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        print(f"开始并行处理: 共 {len(batches)} 个批次, 线程数: {max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(self._process_single_batch, b, labels, desc_text): b 
                for b in batches
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="分类进度"):
                batch_data = future.result()
                if batch_data:
                    # 获取锁并写入最终结果
                    with self.results_lock:
                        for p, t in batch_data:
                            final_results[t]["count"] += examples[p]
                            final_results[t]["err_count"] += error_count_dict[p]
                            final_results[t]["examples"][p] = examples[p]
                            
        return final_results

processor = BlockErrorProcessor()
processor.process('exp/distill+dropout-Qwen2.5-7B-Instruct-linear-20260412-1644-slidespeech/aispeech_asr_epoch_19_total_step_60000/decode_slidespeech_asr_test_norm_cer')

initial_data = processor.word_stats.get_summary()

agent = RecursiveErrorAgent(ApiClient(), "analyze_error/type.yaml")
final_report = agent.process_node(initial_data[0], agent.structure)

with open("analyze_error/refined_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2)