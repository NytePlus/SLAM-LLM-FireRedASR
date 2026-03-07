import argparse
import os

def clean_corr_file(pred_path, corr_path, ratio_threshold=1.5, min_diff=10):
    """
    pred_path: 原始识别结果文件
    corr_path: 纠错后的结果文件
    ratio_threshold: 长度比例阈值。如果 len(corr)/len(pred) > ratio 或 < 1/ratio，则视为异常。
    min_diff: 绝对字符长度差异。如果长度差异小于这个值，即使比例超标也不剔除（针对超短句）。
    """
    
    # 1. 加载 pred 原始长度
    preds = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ", maxsplit=1)
            if len(parts) == 2:
                preds[parts[0]] = parts[1]
            else:
                preds[parts[0]] = ""

    # 2. 读取并过滤 corr
    valid_lines = []
    removed_count = 0
    
    if not os.path.exists(corr_path):
        print(f"Error: {corr_path} 不存在")
        return

    with open(corr_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split(" ", maxsplit=1)
            uttid = parts[0]
            corr_text = parts[1] if len(parts) == 2 else ""
            
            if uttid not in preds:
                # 如果 pred 里没这个 ID，说明数据不匹配，保留或剔除视你而定
                valid_lines.append(line)
                continue
                
            pred_text = preds[uttid]
            len_p = len(pred_text)
            len_c = len(corr_text)
            
            # 计算长度比例
            # 避免除以 0
            safe_len_p = max(len_p, 1)
            ratio = len_c / safe_len_p
            
            # 判断逻辑：
            # 1. 长度比例超过阈值 (如 1.5 倍或低于 0.6 倍)
            # 2. 且绝对差异大于 min_diff (防止误删像 "OK" 变 "Okay" 这种短句)
            is_abnormal = (ratio > ratio_threshold or ratio < (1/ratio_threshold)) and abs(len_c - len_p) > min_diff
            
            # 特殊情况：如果 corr 输出了 JSON 标签或者大量重复，通常长度会暴增
            if is_abnormal:
                print(f"剔除异常行 [{uttid}]: pred_len={len_p}, corr_len={len_c}, ratio={ratio:.2f}")
                removed_count += 1
            else:
                valid_lines.append(line)

    # 3. 写回文件 (覆盖原文件，建议先备份)
    backup_path = corr_path + ".bak"
    os.rename(corr_path, backup_path)
    
    with open(corr_path, 'w', encoding='utf-8') as f:
        for line in valid_lines:
            f.write(line + "\n")
            
    print(f"\n处理完成!")
    print(f"原始备份至: {backup_path}")
    print(f"已从 {corr_path} 中剔除 {removed_count} 条异常记录。")
    print("现在你可以重新运行主推理脚本，它会自动补全这些缺失的行。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="原始 pred 文件路径")
    parser.add_argument("--corr", required=True, help="待清理的 corr 文件路径")
    parser.add_argument("--ratio", type=float, default=1.5, help="长度差异比例阈值")
    args = parser.parse_args()

    clean_corr_file(args.pred, args.corr, ratio_threshold=args.ratio)