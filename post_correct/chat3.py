import os
import re
import torch
import torch_npu
import torch.nn.functional as F
import soundfile
import logging
import random
import numpy as np
import hydra
import kaldiio
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field

from aispeech_asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig
from utils.model_utils import get_custom_model_factory

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class Color:
    LIGHT_GRAY = "\033[90m"
    END = '\033[0m'

@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    ckpt_path: str = field(default="", metadata={"help": "Path to the model checkpoint"})

def compute_ppl_and_topk(model, input_ids, attention_mask, input_features, input_feature_length, labels, n=5):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            input_feature_length=input_feature_length,
            labels=labels
        )
        output_text = model.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        print(output_text)

    with torch.no_grad():
        outputs, _ = model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            input_feature_length=input_feature_length,
            labels=labels
        )

        loss = outputs.loss
        ppl = torch.exp(loss)
        logits = outputs.logits  # 假设维度 [1, L_total, Vocab]
        seq_len_labels = (labels != -100).int().sum().item()
        
        # 确保 relevant_logits 和 mask 维度完全一致
        # target_logits 形状变为 [num_valid_tokens, vocab_size]
        target_logits = logits[0][-seq_len_labels:] 
        target_ids = labels[0][-seq_len_labels:]

        # 3. 计算 Top-N
        probs = torch.softmax(target_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=n, dim=-1)

        top_n_results = []
        for i in range(target_ids.size(0)):
            step_results = []
            for j in range(n):
                # 针对 Qwen2.5 或其他模型，确保使用正确的 tokenizer
                token = model.tokenizer.decode([top_indices[i, j]])
                prob = top_probs[i, j].item()
                step_results.append((token, prob))
            
            actual_token = model.tokenizer.decode([target_ids[i]])
            top_n_results.append({
                "target": actual_token,
                "top_n": step_results
            })

    return loss.item(), ppl.item(), top_n_results

def process_audio(audio_path, target_sr=16000):
    if re.search(r'\.ark:\d+', audio_path):
        sample_rate, wav_np = kaldiio.load_mat(audio_path)
        audio_raw = wav_np.astype(np.float32)
    elif audio_path.endswith('wav'):
        audio_raw, sample_rate = soundfile.read(audio_path)
        if len(audio_raw.shape) > 1:
            audio_raw = audio_raw[:, 0]
        audio_raw = audio_raw.astype(np.float32)
    
    input_features = torch.from_numpy(audio_raw) 
    with torch.no_grad():
        input_features = torch.nn.functional.layer_norm(input_features, input_features.shape)
    input_feature_length = input_features.shape[0]
    return input_features, input_feature_length

@hydra.main(config_name=None, version_base=None)
def main(cfg: DictConfig):
    # 1. 合并默认配置与命令行参数
    base_cfg = RunConfig()
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    train_config = cfg.train_config
    model_config = cfg.model_config
    torch_npu.npu.manual_seed(train_config.seed)
    
    # 2. 设备准备
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. 加载模型与分词器 (复用 factory)
    logger.info("Initializing model factory...")
    model_factory = get_custom_model_factory(model_config, logger)
    
    # 这里通过转换后的 cfg 传参，确保 kwargs 包含所有必要信息
    model, tokenizer = model_factory(train_config, model_config)
    
    # 如果指定了 checkpoint，加载权重
    if cfg.ckpt_path:
        logger.info(f"Loading checkpoint from: {cfg.ckpt_path}")
        state_dict = torch.load(cfg.ckpt_path)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    ignore_index = getattr(tokenizer, "default_ignore_token", -100)

    print("  Speech-LLM Interactive PPL Tool(Type 'exit' to quit)")

    while True:
        try:
            print("\n" + "-"*20)
            prompt_text = input("Prompt: ").strip().replace('\\n', '\n')
            if prompt_text.lower() == 'exit': break
            
            audio_path = input("Audio File for <speech>: ").strip()
            if audio_path.lower() == 'exit': break

            target_text = input("Guide Text (for PPL): ").strip().replace('\\n', '\n')
            if target_text.lower() == 'exit': break

            # 4. 数据处理
            feat, feat_length = process_audio(audio_path)
            feat = feat.to(device).unsqueeze(0)
            feat_length = torch.tensor([feat_length]).to(device)
            
            p_ids = tokenizer.encode(prompt_text)
            t_ids = tokenizer.encode(target_text)
            if hasattr(tokenizer, 'eos_token_id'):
                t_ids.append(tokenizer.eos_token_id)
            
            input_ids = torch.tensor(p_ids + t_ids).unsqueeze(0).to(device)
            attention_mask = input_ids.ge(-1).to(device)
            
            # 构造 Labels: Prompt 设为 ignore_index，Target 设为原 ID
            labels = torch.full(input_ids.shape, ignore_index).to(device)
            labels[0, len(p_ids):] = torch.tensor(t_ids)

            # 5. 计算
            loss, ppl, top_n_results = compute_ppl_and_topk(model, input_ids, attention_mask, feat, feat_length, labels)
            print(f">> PPL:    {ppl:.4f}")
            top_n_display = '\n'.join(f"{res['target']}: {' '.join(f'({i[0]} {i[1] : 0.4f})' for i in res['top_n'])}" for res in top_n_results) 
            print(f"{Color.LIGHT_GRAY}TOP_N:\n{top_n_display}{Color.END}")

        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)

if __name__ == "__main__":
    main()

# <|im_start|>user\n<speech>{}<|im_end|>\n<|im_start|>assistant\n
# <|im_start|>user\n<speech>Transcribe speech to text. Use hotwords in ppt to improve speech recognition accuracy. But if the hotwords are irrelevant, just ignore them. The hotwords are SYSTEMIC DESIGN, INNOVATION SYSTEMIC, SERVICE INNOVATION, SYSTEMIC CHANGE, INNOVATION, DANISH DESIGN, ORGANISATIONS DESIGN, DESIGN CENTRE, CIVIC ORGANISATIONS, DESIGN, PRODUCT THINKING, DESIGN CHRISTIAN, PUBLICS SECTOR, SOCIETY CENTRE, BASO METHODS, CHANGE, SECTOR ACTORS, METHODS SERVICE, VALUE CITIZENS, SERVICE, BEHAVIOUR SOCIETY, METHODS PRODUCT, CITIZENS BEHAVIOUR, CIVIC, EXPERIENCE, EMERGING PROBLEMS, CHRISTIAN BASO, INTERACTION EMERGING, EMERGING, CHANGE SOLVE, PRODUCT, CHALLENGES ADDRESS, BEHAVIOUR, COMPLEX, FOCUS DESIGN, DANISH, PROBLEMS, THINKING, VALUE, CONCRETE LIMITED, COPRODUCES VALUE, ANDR METHODS, CENTRE ZOOM, ACTORS PUBLIC, PRIVATE, LIMITED, WICKED ARISING, USER, COMPLEX TURBULENT, ADDRESS.<|im_end|>\n<|im_start|>assistant\n
# /aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech/test_oracle_v1/data/format.1/data_wav.ark:12795606
# BUT I AM SAYING THAT THIS DESIGN FOCUS ON USER EXPERIENCE AND BEHAVIOR AT A VERY DISCRETE LEVEL -> DISCREET/DESCRIPTIVE

"""
And there's, a network now civil servants working with the sign thinking in the Danish government, and we hear that the center also work on on a cross cutting networks, for example in social innovation.
Leadership is essential and I'll get back to that, but it's a really, really important piece to embrace new ways of innovating uh, to tap into the role of managers and and the role of leadership.
So we all I think most of us will recognized that design thinking is about exploring problem spaces getting close to users, co-creating new solutions and prototyping testing iserating to make the future concrete.
And that also goes for for for designing within government settings.
So we would at the dangerous design center still using these methods with working with government dive into how citizens and this in this case enterprises experience interacting with government.
Go out and visit farms to explore the problem in practice and understand what it takes to ah to make it easier and they're more smooth to to run. Ah run a farming company.
And to co-create with citizens and other stakeholders in collaborative ways, um, all of which I think leads to the question of what more is needed.
And again, I'll address this as early said in an upcoming book, I won't say too much about it today that will be next time, maybe next year, but I think we need to move towards more systemic design.
And by that I mean that we need to shift away from thinking we can just and only use design thinking to solve concrete, quite limited challenges.
That Arise in your interaction between the citizens and the public sector. So you could say a narrow view of service design I'm not saying it's not important and can and can make a difference.

"""