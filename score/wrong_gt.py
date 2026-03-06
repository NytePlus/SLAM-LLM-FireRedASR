import json
import os
import re
from tqdm import tqdm

# 假设 EditDistance 和 Code 已在 score 中定义
from score import EditDistance, Code 

def load_id_mapping(id2id_file):
    """加载 id2id 映射文件"""
    mapping = {}
    with open(id2id_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

def convert_id(original_id, mapping):
    """
    转换逻辑: YTB+15-b1NyRxtM+00004 -> 1475-91655-0004
    """
    # 使用正则匹配中间的 ytbid 和 最后的 5位序号
    # 格式：YTB + (任何非加号内容) + (5位数字)
    match = re.match(r"YTB\+(.+)\+(\d{5})", original_id)
    if match:
        ytbid, num_str = match.groups()
        if ytbid in mapping:
            new_prefix = mapping[ytbid]
            # 5位变4位：取后四位 00004 -> 0004
            new_num = num_str[-4:] 
            return f"{new_prefix}-{new_num}"
    return None

def analyze_deletion_errors(jsonl_file, pred_txt, id2id_file, report_file, ids_output_file=None):
    ed = EditDistance()
    bad_cases = []
    del_rates = []
    wrong_id_list = []
    
    # 0. 加载映射
    id_map = load_id_mapping(id2id_file)

    stats_summary = {
        "total_jsonl": 0,
        "total_pred": 0,
        "matched_utt": 0,
        "total_del": 0,
        "total_ref_words": 0
    }
    
    # 1. 加载 Ground Truth
    references = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                references[data['key']] = data['target'].split()
                stats_summary["total_jsonl"] += 1

    total_lines = sum(1 for _ in open(pred_txt, 'r', encoding='utf-8'))

    # 2. 读取预测并进行对齐
    with open(pred_txt, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="计算对齐", unit="utt"):
            line = line.strip()
            if not line: continue
            stats_summary["total_pred"] += 1
            
            parts = line.split(maxsplit=1)
            original_uttid = parts[0]
            hyp_tokens = parts[1].split() if len(parts) > 1 else []

            if original_uttid not in references:
                continue
            new_uttid = convert_id(original_uttid, id_map)

            stats_summary["matched_utt"] += 1
            ref_tokens = references[original_uttid]
            
            try:
                result = ed.align(ref_tokens, hyp_tokens)
                
                stats = {"match": 0, "sub": 0, "del": 0, "ins": 0}
                for code, _, _ in zip(result.codes, result.refs, result.hyps):
                    if code == Code.match: stats["match"] += 1
                    elif code == Code.substitution: stats["sub"] += 1
                    elif code == Code.deletion: stats["del"] += 1
                    elif code == Code.insertion: stats["ins"] += 1
                
                ref_len = len(ref_tokens)
                stats_summary["total_del"] += stats["del"]
                stats_summary["total_ref_words"] += ref_len
                del_rate = stats["del"] / ref_len if ref_len > 0 else 0

                if (stats["del"] == 3 and del_rate > 0.3) or (stats["del"] == 2 and del_rate > 0.5) or (stats["del"] >= 4 and del_rate > 0.2):
                    wrong_id_list.append(new_uttid)

                del_rates.append(del_rate)
                bad_cases.append({
                    "original_id": original_uttid,
                    "new_id": new_uttid if new_uttid else "N/A",
                    "gt": " ".join(ref_tokens),
                    "pred": " ".join(hyp_tokens),
                    "del_count": stats["del"],
                    "del_rate": del_rate,
                })
                
            except ValueError as e:
                continue

    # 3. 统计分布
    bad_cases.sort(key=lambda x: x['del_count'], reverse=True)
    bins = [0] * 11
    for r in del_rates:
        idx = int(r * 10)
        if idx > 10: idx = 10
        bins[idx] += 1

    # 4. 写入分析报告
    if ids_output_file:
        with open(ids_output_file, 'w', encoding='utf-8') as f_ids:
            for wid in wrong_id_list:
                f_ids.write(f"{wid}\n")

    avg_del_rate = (stats_summary["total_del"] / stats_summary["total_ref_words"]) if stats_summary["total_ref_words"] > 0 else 0
    with open(report_file, 'w', encoding='utf-8') as rf:
        rf.write(f"总体 Deletion Rate: {avg_del_rate:.2%}\n")
        rf.write("-" * 90 + "\n")
        rf.write(f"{'New ID':<20} | {'Old ID':<30} | {'Del Cnt':<8} | {'Del%':<8}\n")
        rf.write("-" * 90 + "\n")
        
        for case in bad_cases:
            rf.write(f"{case['new_id']:<20} | {case['original_id']:<30} | {case['del_count']:>7} | {case['del_rate']:>7.1%}\n")
            rf.write(f"  REF: {case['gt']}\n")
            rf.write(f"  HYP: {case['pred']}\n")
            rf.write("-" * 90 + "\n")

    return bad_cases

cases = analyze_deletion_errors(
    '/data/slidespeech/train_95/slides/multitask.jsonl', 
    '/aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/AudioOmniTest/score/qwen_asr_train',
    '/data/slidespeech/train_95/id2id',
    'train_wrong_gt.txt',
    ids_output_file='wrong_ids.txt'
)