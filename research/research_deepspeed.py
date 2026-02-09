import os
import random
from typing import Optional
import logging
from contextlib import nullcontext

import hydra
import logging
from dataclasses import dataclass, field
import torch
import torch_npu
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
import deepspeed
from typing import Optional
from aispeech_asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig
from dataset.speech_dataset_large_wavlm import MultiTaskDataset, FixedBatchDataset
from utils.deepspeed_utils import deepspeed_main_wrapper

from research.model_factory import model_factory
from research.plot import save_heatmap, save_scalar, save_scalar_withstr
from research.res_utils import find_generated_spans, decode_replace_audios

@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to projector checkpoint"}
    )
    deepspeed_config : str =""



@deepspeed_main_wrapper(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    # kwargs = to_plain_list(cfg)
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if cfg.get("debug", False):
        import pdb

        pdb.set_trace()

    main(cfg)

def collate_keys(samples):
    result = {}
    result["keys"] = [s['key'] for s in samples]
    return result

def compute_coarse_similarity(H, audio_start, audio_length, gen_start, gen_length):
    """
    H: [1, seq_len, hidden_dim] tensor
    audio_start, audio_length: int
    gen_start, gen_length: int

    Returns:
        h1_mean: [hidden_dim] tensor
        h2_mean: [hidden_dim] tensor
        cosine_sim: float
        euclidean_dist: float
    """
    # 取 slice
    h1 = H[:, audio_start:audio_start+audio_length, :]   # [1, audio_length, hidden_dim]
    h2 = H[:, gen_start:gen_start+gen_length, :]         # [1, gen_length, hidden_dim]

    # 沿 seq_len 维求均值
    h1_mean = h1.mean(dim=1).squeeze(0)  # [hidden_dim]
    h2_mean = h2.mean(dim=1).squeeze(0)  # [hidden_dim]

    # 余弦相似度
    cosine_sim = F.cosine_similarity(h1_mean.unsqueeze(0), h2_mean.unsqueeze(0)).item()

    # 欧几里得距离
    euclidean_dist = torch.norm(h1_mean - h2_mean, p=2).item()

    return h1_mean, h2_mean, cosine_sim, euclidean_dist


def main(kwargs: DictConfig):
    # Update the configuration for the training and sharding process
    # train_config, fsdp_config, model_config, log_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG(), LOG_CONFIG()
    # update_config((train_config, fsdp_config, model_config, log_config), **kwargs)

    train_config, model_config, log_config, dataset_config, deepspeed_config = kwargs.train_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config, \
                                                                          kwargs.deepspeed_config
    del kwargs.train_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config
    
    # Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 

    logger.addHandler(file_handler)


    # Set the seeds for reproducibility
    torch_npu.npu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    deepspeed.init_distributed(
        dist_backend='hccl',    # 使用NCCL后端（GPU场景）
    )

    if rank == 0:
        logger.info("train_config: {}".format(train_config))
        logger.info("model_config: {}".format(model_config))
        logger.info("log_config: {}".format(log_config))

    # Set wandb
    if rank == 0:
        if log_config.use_wandb:
            if not os.path.exists(log_config.wandb_dir):
                os.makedirs(log_config.wandb_dir, exist_ok=True)
            wandb_config={"train_config": train_config, "model_config": model_config, "log_config": log_config}
            wandb.init(dir=log_config.wandb_dir, entity=log_config.wandb_entity_name, project=log_config.wandb_project_name,name=log_config.wandb_exp_name ,config=wandb_config)

    model, tokenizer = model_factory(train_config, model_config, **kwargs)

    device = torch.device(f"npu:{local_rank}" if torch.npu.is_available() else "cpu")
    model.to(device)
    model.eval()
    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=None, config=deepspeed_config
    )
    logger.info("dataset_config: {}".format(dataset_config))
    dataset_train = MultiTaskDataset(dataset_config, tokenizer, split="train")
    batch_dataset = FixedBatchDataset(dataset_train, 1)
    train_dataloader = torch.utils.data.DataLoader(
            batch_dataset,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            shuffle=False,
            batch_size=train_config.val_batch_size,
            drop_last=False,
            collate_fn=dataset_train.collator,
        )
    key_dataloader = torch.utils.data.DataLoader(
            FixedBatchDataset(MultiTaskDataset(dataset_config, tokenizer, split="train"), 1),
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            shuffle=False,
            batch_size=train_config.val_batch_size,
            drop_last=False,
            collate_fn=collate_keys,
        )
    
    autocast = torch.npu.amp.autocast if train_config.use_fp16 else nullcontext

    output_dir = kwargs.get('decode_log')
    attn_output = os.path.join(output_dir, 'attention_maps')
    norm_output = os.path.join(output_dir, 'hidden_norm')
    sim_output = os.path.join(output_dir, 'coarse sim')
    wavlmout_output = os.path.join(output_dir, 'wavlm_output')
    os.makedirs(attn_output, exist_ok=True)
    os.makedirs(norm_output, exist_ok=True)
    os.makedirs(sim_output, exist_ok=True)
    os.makedirs(wavlmout_output, exist_ok=True)
    n, i = 10, 0
    with torch.no_grad():
        for step, (batch_key, batch) in tqdm(enumerate(zip(key_dataloader, train_dataloader))):
            i += 1
            if i > n: break
            keys = batch_key['keys']
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
                model_outputs, _ = model(**batch)
                encoder_outs, _ = model.encoder.extract_features(batch['input_features'])
                encode_feature_length = model.encoder.compute_feature_length(batch['input_feature_length']) // model.encoder_projector.ds

            # --- wavlm output ---
            for b, key in enumerate(keys):
                seq_len = encode_feature_length[b]
                wavlm_l2_norms = torch.norm(encoder_outs[b, :seq_len], p=2, dim=-1).detach().cpu().tolist()
                save_path = os.path.join(wavlmout_output, f"norm_{key}.png")
                if os.path.exists(save_path):
                    continue
                save_scalar_withstr(
                    x_labels=[''] * seq_len,
                    y_values=wavlm_l2_norms,
                    save_path=save_path,
                    xlabel="Token",
                    ylabel="L2 Norm",
                    title=f"Key {key} | Wavlm ouput L2 Norm"
                )

            # --- attention map ---
            audio_lengths = encode_feature_length.detach().cpu().numpy().tolist()
            all_token_ids = batch['input_ids']
            token_texts = [
                decode_replace_audios(token_ids, model.tokenizer, audio_length)
                for token_ids, audio_length in zip(all_token_ids, audio_lengths)
            ]
            attn_map_list = model_outputs.attentions # torch.Size([1, 32, 357, 357])
            for layer, attn_maps in enumerate(attn_map_list):
                if attn_maps is None:
                    continue
                for i, (key, attn_map, token_text) in enumerate(zip(keys, attn_maps, token_texts)):
                    save_path=os.path.join(attn_output, f'attn_{key}_{layer}.png')
                    if os.path.exists(save_path):
                        continue
                    save_heatmap(
                        matrix=attn_map[0].cpu().detach().to(torch.float32).numpy(), 
                        save_path=save_path,
                        x_labels=token_text,
                        y_labels=token_text
                    )
                    print(f'save to attn_{key}_{layer}.png')
                    
            # --- 粗粒度跨模态相似度 ---
            speech_token_id = tokenizer.default_speech_token
            batch_size = batch['input_ids'].shape[0]
            batch_idx, audio_starts = (batch['input_ids'] == speech_token_id).nonzero(as_tuple=True)

            gen_starts, gen_lengths = find_generated_spans(batch['input_ids'], batch['labels'])

            hiddens = model_outputs.hidden_states # torch.Size([1, 357, 4096])
            cos_sims, enc_dist = {}, {}
            for layer, hidden in enumerate(hiddens):
                for key, audio_start, audio_length, gen_start, gen_length in zip(keys, audio_starts, audio_lengths, gen_starts, gen_lengths):
                    h1_mean, h2_mean, cos_sim, euc_dist = compute_coarse_similarity(hidden, audio_start, audio_length, gen_start, gen_length)
                    if cos_sims.get(key) is None:
                        cos_sims[key] = []
                    cos_sims[key].append(cos_sim)
                    if enc_dist.get(key) is None:
                        enc_dist[key] = []
                    enc_dist[key].append(euc_dist)
            for key in keys:
                save_path = os.path.join(sim_output, key)
                if not os.path.exists(save_path):
                    save_scalar(save_path, cos_sims[key], "cosine similarity")
                    save_scalar(save_path, enc_dist[key], "enc distance")

            # --- 嵌入空间大小 ---
            for layer, hidden in enumerate(hiddens):
                l2_norms = torch.norm(hidden[0], p=2, dim=-1).detach().cpu().tolist()

                for b, key in enumerate(keys):
                    x_labels = token_texts[b]

                    assert len(x_labels) == len(l2_norms), \
                        f"Length mismatch: tokens={len(x_labels)}, norms={len(l2_norms)}"

                    save_path = os.path.join(norm_output, f"norm_{key}_{layer}.png")
                    if os.path.exists(save_path):
                        continue
                    save_scalar_withstr(
                        x_labels=x_labels,
                        y_values=l2_norms,
                        save_path=save_path,
                        xlabel="Token",
                        ylabel="L2 Norm",
                        title=f"Layer {layer} | Key {key} | Hidden L2 Norm"
                    )

if __name__ == "__main__":
    main_hydra()
