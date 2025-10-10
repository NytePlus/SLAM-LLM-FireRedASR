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
    encoder_name: str = 'conformer'
    encoder_config: ConformerConfig = field(default_factory=ConformerConfig)  #
    firered_path :str = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/FireRedASR-LLM/model.pth.tar"
    encoder_projector_ds_rate: int = 2


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
    lr:float = 1e-4
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


@dataclass
class DataConfig:
    dataset: str = "multitask_dataset"
    max_audio_length : int = 30
    train_max_frame_length: int = 1500
    ds_rate: int = 8
    eval_max_frame_length: int = 2000
    append_info_tasks : List = field(default_factory=lambda: ["hotword"])
    multitask_prompt_path: str = "conf/multiprompt.jsonl"
    prompt_style: str = "<|im_start|>user\n<speech>{}<|im_end|>\n<|im_start|>assistant\n"
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

@dataclass
class LogConfig:
    use_wandb: bool = False
    wandb_dir: str = "tmp/test_wandb"
    wandb_entity_name: str = "project_name"
    wandb_project_name: str = "project_name"
    wandb_exp_name: str = "exp_name"
    log_file: str = "tmp/test.log"
    log_interval: int = 5
