import torch
from torch.utils.data import Dataset,IterableDataset
import whisper
import kaldiio
import types
from functools import partial
import torch.distributed as dist
import string
import copy
import numpy as np
import copy
from tqdm import tqdm
import os
import json
import random
import logging
import subprocess
import torchaudio
import torchaudio.functional as F
import torchaudio.compliance.kaldi as kaldi
from model.asr_feat import ASRFeatExtractor
import logging


logger = logging.getLogger(__name__)


class MultiTaskDataset(IterableDataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()
        cmvn_path = dataset_config.cmvn_file
        self.feature_extractor = ASRFeatExtractor(cmvn_path)
        self.multitask_prompt_list = {}
        self.append_info_tasks = dataset_config.append_info_tasks
        with open(dataset_config.multitask_prompt_path) as f_prompt:
            for line in f_prompt:
                item = json.loads(line.strip())
                if item["task"] in self.multitask_prompt_list:
                    self.multitask_prompt_list[item["task"]].append(item["prompt"])
                else:
                    self.multitask_prompt_list[item["task"]] = [item["prompt"]]
        print(f"[Prompt] {self.multitask_prompt_list}")
        if split == "train":
            self.data_path = dataset_config.train_scp_file_path
            if dataset_config.wav_reverb:
                self.rirs_list = []
                with open(dataset_config.rirs_path, encoding='utf-8') as fin:
                    for line in fin:
                        self.rirs_list.append(line.strip())
        elif split == "val":
            self.data_path = dataset_config.dev_scp_file_path
        elif split == "test":
            self.data_path = dataset_config.test_scp_file_path
        else:
            raise ValueError("Split must be train val test")
        
        self.prompt_template = dataset_config.get("prompt_style", "{}")
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.max_audio_length = dataset_config.get("max_audio_length", 30)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.sample_rate = 16000

    def __iter__(self):
        multitask_task_path = os.path.join(self.data_path,"multitask.jsonl")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        total_num_workers = num_workers * world_size
        worker_rank = rank * num_workers + worker_id 
        with open(multitask_task_path) as f_task:
            for data_index,line in enumerate(f_task):
                if (data_index % total_num_workers) == worker_rank:
                    # try:
                    item = json.loads(line.strip())
                    ark_path = item["path"]
                    key = item["key"]
                    target = item["target"].lower()
                    task = item["task"]
                    sample_rate, wav_np = kaldiio.load_mat(ark_path)
                    audio_raw = wav_np.astype(np.float32) / 32768
                    if len(audio_raw) / self.sample_rate > self.max_audio_length or len(audio_raw) / self.sample_rate < 0.1: 
                        continue

                    if self.dataset_config.wav_reverb:
                        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
                        wav_tensor = self.wav_reverb(wav_tensor, self.dataset_config.reverb_prob)
                        wav_np = wav_tensor.squeeze(0).numpy()

                    input_features, input_feature_length = self.feature_extractor((sample_rate, wav_np))

                    # feature postprocessing
                    if self.dataset_config.spec_aug:
                        spec_aug_conf = self.dataset_config.spec_aug_conf
                        input_features = self.spec_aug(input_features, **spec_aug_conf)

                    prompt = random.choice(self.multitask_prompt_list[task])
                    prompt = self.prompt_template.format(prompt)
                    if task in self.append_info_tasks:
                        prompt = prompt.format(item[task])
                    prompt_ids = self.tokenizer.encode(prompt)
                    prompt_length = len(prompt_ids)
                    prompt_ids = torch.tensor(prompt_ids)

                    if  not self.inference_mode:
                        target_ids = self.tokenizer.encode(target)
                        target_ids.append(self.tokenizer.eos_token_id)
                        target_ids = torch.tensor(target_ids)
                        input_ids = torch.cat([prompt_ids,target_ids])
                    else:
                        input_ids = prompt_ids
                    attention_mask = input_ids.ge(-1)  
                    result = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask ,
                            "input_features": input_features ,
                            "input_feature_length":input_feature_length,
                            'key': key,
                            'target': target,
                    }

                    if  not self.inference_mode:
                        labels = copy.deepcopy(input_ids)
                        labels[:prompt_length] = self.tokenizer.default_ignore_token
                        result["labels"] = labels
                    yield result
                    # except:
                    #     logger.error(f"{data_index},{item}")
            
    def pad(self, sequence, max_length, padding_idx=0,padding_style = "right"):
            if isinstance(sequence, (int, list, tuple)):
                if len(sequence) < max_length:
                    if padding_style == "right":
                        sequence = sequence + [padding_idx] * (max_length - len(sequence))
                    else:
                        sequence =  [padding_idx] * (max_length - len(sequence)) + sequence 
                else:
                    sequence = sequence[:max_length]
            elif isinstance(sequence, torch.Tensor):
                if len(sequence) < max_length:
                    if padding_style == "right":
                        sequence = torch.cat(
                            (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
                    else:
                        sequence = torch.cat(
                            (torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx),sequence))
                else:
                    sequence = sequence[:max_length]
            elif isinstance(sequence, np.ndarray):
                if len(sequence) < max_length:
                    if padding_style == "right":
                        sequence = np.concatenate(
                            (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
                    else:
                        sequence = np.concatenate(
                            (np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx),sequence))
                else:
                    sequence = sequence[:max_length]
            else:
                raise Exception("Type mismatch during padding!")
            return sequence
    
    def extract_fbank(self,waveform):
        fbank_features = kaldi.fbank(
            waveform,
            num_mel_bins=self.dataset_config["fbankConfig"]["num_mel_bins"],  # 梅尔频率滤波器组的滤波器数量
            frame_length=self.dataset_config["fbankConfig"]["frame_length"],  # 音频帧的长度（毫秒）
            frame_shift=self.dataset_config["fbankConfig"]["frame_shift"],  # 帧移（毫秒）
            dither=self.dataset_config["fbankConfig"]["dither"] if self.split == "train" else 0,  # 抖动系数
            window_type=self.dataset_config["fbankConfig"]["window_type"],  # 窗口类型
            use_energy=self.dataset_config["fbankConfig"]["use_energy"],  # 是否使用能量特征
            low_freq=self.dataset_config["fbankConfig"]["low_freq"],  # 低频截止频率（Hz）
            high_freq=self.dataset_config["fbankConfig"]["high_freq"],  # 高频截止频率（Hz）
            htk_compat=self.dataset_config["fbankConfig"]["htk_compat"]  # 是否与HTK兼容
        )
        return fbank_features
    
    def collator(self, samples):
        assert samples is not None
        if self.inference_mode:
            padding_style = "left"
        else:
            padding_style = "right"
        padding_style = "left"
        input_feature_length = torch.stack([torch.tensor(s["input_feature_length"]) for s in samples])
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id,padding_style = padding_style)
                                    for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False,padding_style = padding_style)
                                        for s in samples])
        input_features_max_length = max([s['input_features'].shape[0] for s in samples])
        input_features = torch.stack([self.pad(s['input_features'], input_features_max_length, 0.0)
                                for s in samples])
        result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask ,
                "input_features": input_features ,
                "input_feature_length":input_feature_length,
        }
       
        if self.inference_mode:
            result["keys"] = [s['key'] for s in samples]
            result["targets"] = [s['target'] for s in samples]
        else:
            result["labels"] = torch.stack([self.pad(s['labels'], input_ids_max_length, self.tokenizer.default_ignore_token,padding_style = padding_style)
                                for s in samples])
        return result

    def spec_aug(self, x, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10):
        """ Do spec augmentation
            Inplace operation

            Args:
                x: input feature tensor
                num_t_mask: number of time mask to apply
                num_f_mask: number of freq mask to apply
                max_t: max width of time mask
                max_f: max width of freq mask

            Returns
                masked feat
        """
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for _ in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        return y

    def wav_reverb(self, x, p=0.3):
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()

        if random.random() > p:
            return y
        
        y = y / (1 << 15)
        rir_path = random.choice(self.rirs_list)
        rir, rir_sr = torchaudio.load(rir_path)
        rir = rir[0:1, :]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)

        corrupted = F.fftconvolve(y, rir)
        T = y.shape[-1]
        corrupted = corrupted[:, :T]

        return corrupted * (1 << 15)


class MultiTaskDynamicBatchDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, window_class) -> None:
        super().__init__()
        self.dp = dataset
        
        assert window_class is not None
        self.window_class = window_class
        self.collator = self.dp.collator
        self._buffer = []

    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, self._buffer):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._buffer
                del self._buffer
                self._buffer = [elem]
        if len(self._buffer) > 0:
            yield self._buffer
        del self._buffer
        self._buffer = []
         
    
def window_class(elem,buffer,max_frame_length,ds_rate):
    # return True 
    if len(buffer) == 0:
        return True
    max_frame = max(len(elem["input_ids"]) + (elem["input_feature_length"] // ds_rate) - 1,max([ len(_["input_ids"]) + (_["input_feature_length"] // ds_rate ) -1 for _ in buffer]))
    return (len(buffer) + 1) * max_frame > max_frame_length


def get_speech_dataset(dataset_config, tokenizer, split):
    if split != "train":
        dataset_config.spec_aug = False
        dataset_config.wav_reverb = False
    dataset = MultiTaskDataset(dataset_config, tokenizer, split)
    if split == "train":
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.train_max_frame_length,ds_rate = dataset_config.ds_rate))
    else:
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.eval_max_frame_length,ds_rate = dataset_config.ds_rate))
    return dataset



    
