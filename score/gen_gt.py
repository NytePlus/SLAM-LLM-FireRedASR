import json
import re
from tqdm import tqdm

def generate_gt_file(jsonl_file, output_gt_file):
    count = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f_in, \
         open(output_gt_file, 'w', encoding='utf-8') as f_out:
        
        # 使用 tqdm 监控进度
        for line in tqdm(f_in, desc="生成 GT 文件"):
            if not line.strip():
                continue
            
            data = json.loads(line)
            id = data.get('key', '')
            target_text = data.get('target', '')
            
            clean_text = target_text.upper().strip()
            
            # 写入格式: ID TEXT
            f_out.write(f"{id} {clean_text}\n")
            count += 1

    print(f"\n✅ 成功生成 GT 文件: {output_gt_file}")
    print(f"📊 总计写入: {count} 条数据")

# 执行转换
generate_gt_file(
    '/data/slidespeech/test_oracle_v1/multitask.jsonl',
    '/aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/SLAM-LLM-FireRedASR/exp/20260121-1702-slidespeech-linear/aispeech_asr_epoch_24_total_step_100000/correct_gt/decode_slidespeech_asr_test_gt'
)