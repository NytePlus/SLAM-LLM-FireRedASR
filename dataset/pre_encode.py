from model.aispeech_asr import setup_encoder
from omegaconf import DictConfig, OmegaConf
from utils.deepspeed_utils import deepspeed_main_wrapper, clear_gpu_cache, setup_environ_flags, train
from dataset.speech_dataset_large_wavlm import MultiTaskDataset
from dataclasses import dataclass, field

def get_dataset(dataset_config, tokenizer, split):
    ds_config = copy.deepcopy(dataset_config)
    if split != "train":
        ds_config.spec_aug = False
        ds_config.wav_reverb = False
        ds_config.add_noise = False
    dataset = MultiTaskDataset(ds_config, tokenizer, split)
    return dataset

@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to projector checkpoint"}
    )
    encoder_ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to wavlm-encoder checkpoint"}
    )

@deepspeed_main_wrapper(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)

    kwargs = cfg
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
    
    logging.basicConfig(level=log_level)
    
    if kwargs.get("debug", False):
        import pdb;
        pdb.set_trace()
        
    main(kwargs)

def main(kwargs: DictConfig):
    model_config, log_config, dataset_config = kwargs.model_config, \
                                                kwargs.log_config, \
                                                kwargs.dataset_config

    encoder = setup_encoder(train_config, model_config)

    dataset_train = get_dataset(tokenizer, dataset_config, split="train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=1,
        pin_memory=True
    )

    for step, batch in enumerate(train_dataloader):
        for key in batch.keys():
            batch[key] = (
                batch[key].to(device).half()
                if isinstance(batch[key], torch.Tensor)
                and batch[key].dtype in [torch.float32, torch.float64]
                else (
                    batch[key].to(device)
                    if isinstance(batch[key], torch.Tensor)
                    else batch[key]
                )
            )
        with autocast(dtype=torch.bfloat16):
            outputs = encoder.extract_features(batch['input_features'])
            print(batch['input_features'].shape, outputs.shape)

if __name__ == "__main__":
    main_hydra()