import json
import os

def save_context_to_json(context_file, id_map_file, output_json):
    # 1. 加载 id_map (假设是行分隔的 id2id 文件)
    id_map = {}
    with open(id_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_map[parts[0]] = parts[1]

    paragraph_data = {}

    # 2. 读取并解析 context
    with open(context_file, "r", encoding='utf-8') as f:
        data = json.load(f)
        for v in data["videos"]:
            channel_id = v["youtube_channel"]
            if channel_id not in id_map:
                continue
                
            utt_prefix = id_map[channel_id]
            
            # 初始化该文段的数据结构
            segments_list = []
            id_to_idx = {}
            
            for s in v["segments"]:
                segments_list.append(s["txt_raw"])
                # 提取序号并格式化为 4 位，例如 0004
                try:
                    seq_num = int(s["uttid"].split('-')[-1])
                    full_uttid = f"{utt_prefix}-{seq_num:04d}"
                    id_to_idx[full_uttid] = len(segments_list) - 1
                except (ValueError, IndexError):
                    continue

            # 以转换后的 utt_prefix 作为 Key
            paragraph_data[utt_prefix] = {
                "utt": utt_prefix,
                "ytb": channel_id,
                "segments": segments_list,
            }

    # 3. 保存为 JSON
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(paragraph_data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 文段数据已保存至: {output_json}")

# --- 执行 ---
save_context_to_json(
    context_file="/data/slidespeech/test_oracle_v1/test_meta_info.json", 
    id_map_file="/data/slidespeech/test_oracle_v1/id2id", 
    output_json="paragraph_segments.json"
)