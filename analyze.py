import re
import sys
import argparse

def filter_high_wer_simple(file_path):
    """简化版本，假设每4行一个完整条目"""
    high_wer_entries = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return []
    except Exception as e:
        print(f"错误: 读取文件时出错 - {e}")
        return []
    
    # 每4行处理一个条目
    for i in range(0, len(lines), 4):
        if i + 3 < len(lines):
            utt_line = lines[i].strip()
            wer_line = lines[i + 1].strip()
            lab_line = lines[i + 2].strip()
            rec_line = lines[i + 3].strip()
            
            # 提取WER值
            wer_match = re.search(r'WER:\s*([\d.]+)\s*%', wer_line)
            if wer_match:
                wer_value = float(wer_match.group(1))
                
                if wer_value > 10.0:
                    entry = {
                        'utt': utt_line.replace('utt: ', ''),
                        'wer_line': wer_line,
                        'wer_value': wer_value,
                        'lab': lab_line.replace('lab: ', ''),
                        'rec': rec_line.replace('rec: ', '')
                    }
                    high_wer_entries.append(entry)
    
    return high_wer_entries

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='筛选WER大于10%的ASR结果条目')
    parser.add_argument('input_file', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（默认：high_wer_results.txt）', 
                       default='high_wer_results.txt')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 处理输入文件
    high_wer_entries = filter_high_wer_simple(args.input_file)
    
    if not high_wer_entries:
        print("未找到WER大于10%的条目")
        return
    
    # 写入输出文件
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"# WER大于10%的条目统计 (共{len(high_wer_entries)}个)\n")
            f.write(f"# 源文件: {args.input_file}\n\n")
            
            for i, entry in enumerate(high_wer_entries, 1):
                f.write(f"条目 {i} (WER: {entry['wer_value']}%):\n")
                f.write(f"utt: {entry['utt']}\n")
                f.write(f"{entry['wer_line']}\n")
                f.write(f"lab: {entry['lab']}\n")
                f.write(f"rec: {entry['rec']}\n")
                f.write("\n")
        
        print(f"找到 {len(high_wer_entries)} 个WER大于10%的条目")
        print(f"结果已保存到: {args.output}")
        
        # 如果启用详细模式，在控制台也显示
        if args.verbose:
            print("\n详细结果:")
            for i, entry in enumerate(high_wer_entries, 1):
                print(f"条目 {i} (WER: {entry['wer_value']}%):")
                print(f"  utt: {entry['utt']}")
                print(f"  lab: {entry['lab']}")
                print(f"  rec: {entry['rec']}")
                print()
                
    except Exception as e:
        print(f"错误: 写入输出文件时出错 - {e}")

if __name__ == "__main__":
    main()