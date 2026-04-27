import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class ConformerConfig:
    idim: int = 80
    n_layers: int = 16
    n_head: int = 20
    d_model: int = 1280
    residual_dropout: float = 0.1
    dropout_rate: float = 0.1
    kernel_size: int = 33
    pe_maxlen: int = 5000

@dataclass
class ModelConfig:
    file: str = "model/aispeech_asr.py:model_factory"
    llm_name: str = "Qwen2-7B-Instruct"
    llm_path: str = "PATH/to/LLAMA/7B"
    llm_type: str = "decoder_only"
    llm_dim: int = 4096
    attn_implementation: str = "eager"
    encoder_name: str = 'conformer'
    encoder_config: ConformerConfig = field(default_factory=ConformerConfig)  #
    encoder_path: Optional[str] = None
    encoder_dim: int = 768
    encoder_projector: str = "linear"
    firered_path :str = ""
    encoder_projector_ds_rate: int = 2
    vl_name: str = "Qwen2-vl-7B-Instruct"
    vl_path: Optional[str] = None
    vl_dim: int = 3584
    cif_loss_weight: Optional[int] = None
    ctc_loss_weight: Optional[float] = None
    attn_distill_weight: float = 0.0
    attn_distill_layers: Optional[List[int]] = None
    attn_distill_context_ratio: float = 0.5
    attn_distill_ignore: str = os.environ.get('DISTILL_IGNORE', "")
    align_part: str = ['tc']
    attn_dropout: float = 0.0
    label_smoothing: float = 0.0

@dataclass
class PeftConfig:
    peft_method: str = "lora" # None , llama_adapter, prefix
    r: int = 64
    lora_alpha: int = 16
    target_modules: List = field(default_factory=lambda: [ "q_proj","k_proj", "v_proj", "o_proj", "up_proj","gate_proj","down_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class FbankConfig:
    num_mel_bins: int = 128  # 梅尔频率滤波器组的滤波器数量为80
    frame_length: int = 25  # 音频帧的长度为25毫秒
    frame_shift: int = 10  # 帧移为10毫秒
    dither: float = 0.001  # 抖动系数为0.001
    window_type: str = "hamming"  # 使用hamming窗口类型
    use_energy: bool = False  # 不使用能量特征
    low_freq:int = 0  # 低频截止频率为0Hz
    high_freq: int = 8000  # 高频截止频率为8000Hz
    htk_compat: bool = True  # 尝试使其与HTK兼容


@dataclass
class TrainConfig:
    model_name:str = "asr_model"
    enable_ddp:bool = False
    enable_deepspeed:bool = False
    enable_fsdp:bool = False
    low_cpu_fsdp:bool = False
    run_validation:bool = True
    batch_size_training: Optional[int] = None
    batching_strategy:str = field(default="packing", metadata={
        "help":"alternative: padding"
    }) #
    context_length:int = 4096
    gradient_accumulation_steps:int = 1
    num_epochs:int = 3
    num_workers_dataloader:int = 4
    warmup_steps:int = 1000
    total_steps:int = 100000
    validation_interval:int = 1000
    weight_decay:float = 0.0
    gamma:float = 0.85
    seed:int = 42
    use_fp16:bool = False
    mixed_precision:bool = True
    val_batch_size:Optional[int] = None
    use_peft:bool = False
    peft_config:PeftConfig = field(default_factory=PeftConfig)
    output_dir:str = "PATH/to/save/PEFT/model"
    run_test_during_validation:bool = False
    freeze_layers:bool = False
    num_freeze_layers:int = 1
    quantization:bool = False
    one_gpu:bool = False
    save_model:bool = True
    freeze_llm:bool = field(default=False, metadata={
        "help": "whether to freeze llm when finetuning, should be true when use peft finetuning"
    })
    freeze_encoder:bool = False
    freeze_projector:bool = False
    exp_name: str = ""
    repetition_penalty: float = 3.0


@dataclass
class DataConfig:
    dataset: str = "multitask_dataset"
    max_audio_length : int = 30
    train_max_frame_length: int = 1500
    ds_rate: int = 8
    eval_max_frame_length: int = 2000
    append_info_tasks : List = field(default_factory=lambda: ["hotword", "cot", "history"])
    multitask_prompt_path: str = "conf/multiprompt.jsonl"
    prompt_style: str = os.environ.get('PROMPT_STYLE', "<|im_start|>user\n<speech>{}<|im_end|>\n<|im_start|>assistant\n")
    cmvn_file: str = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/FireRedASR-LLM/cmvn.ark"
    file: str = "dataset/speech_dataset_large.py:get_speech_dataset"
    train_scp_file_path: str = ""
    dev_scp_file_path: str = ""
    test_scp_file_path: str = ""
    train_split: str = "train"
    dev_split: str = "val"
    test_split:str = "test"
    pad_or_trim: bool = True
    prompt: Optional[str] = None
    inference_mode: bool = False
    lower: bool = False
    fix_length_audio: int = -1
    fbankConfig: FbankConfig = field(default_factory=FbankConfig)
    inference_mode:bool = False
    input_type: str = field(default="raw", metadata={
                                "help":"Use raw when input is wav, mel when for whisper"
                            })
    mel_size: int = field(default=80, metadata={
        "help": "80 for whisper large v1 and v2, 128 for v3"
    })
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })
    spec_aug: bool = False
    spec_aug_conf: Dict[str, int] = field(default_factory=lambda: {"num_t_mask": 4, "num_f_mask": 4, "max_t": 50, "max_f": 10})
    wav_reverb: bool = False
    reverb_prob: float = 0.3
    rirs_path: str = "./data/rirs.wavlist"
    add_noise: bool = False
    noise_prob: float = 0.3
    noises_path: str = "./data/noises.wavlist"
    max_pixels: int = 256 * 256
    image_processor_path: Optional[str] = None
    include_transcript: bool = False

@dataclass
class LogConfig:
    use_wandb: bool = False
    wandb_dir: str = "tmp/test_wandb"
    wandb_entity_name: str = "project_name"
    wandb_project_name: str = "project_name"
    wandb_exp_name: str = "exp_name"
    log_file: str = "tmp/test.log"
    log_interval: int = 5
