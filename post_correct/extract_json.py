import json
import re

def extract_transcription(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            ary = line.strip().split(" ", maxsplit=1)
            if len(ary) >= 2:
                uttid, hyp = ary[0], ary[1]
            else:
                uttid, hyp = ary[0], ""
                    
            # try:
            #     data = json.loads(hyp)
            #     corrected_text = data.get("Corrected Transcription", "").replace('\n', ' ')
            
            #     # 3. 写入新文件，保持 "ID 内容" 的格式
            #     f_out.write(f"{uttid} {corrected_text}\n")
            # except Exception as e:
            #     f_out.write(f"{uttid} {hyp}\n")
            #     print(f"处理行时出错: {uttid}: {e}")
            if 'Corrected Transcription' in hyp:
                continue
            
            f_out.write(f"{uttid} {hyp}\n")

if __name__ == "__main__":
    # 修改为你实际的文件名
    input_path = "exp/20260121-1702-slidespeech-linear/aispeech_asr_epoch_24_total_step_100000/decode_slidespeech_asr_test_corr"
    output_path = "exp/20260121-1702-slidespeech-linear/aispeech_asr_epoch_24_total_step_100000/decode_slidespeech_asr_test_corr2"
    extract_transcription(input_path, output_path)
    print(f"转换完成！结果已保存至: {output_path}")