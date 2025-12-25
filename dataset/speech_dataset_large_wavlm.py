import torch
from torch.utils.data import Dataset,IterableDataset
import whisper
import kaldiio
import types
import re
import soundfile
from functools import partial
import torch.distributed as dist
import copy
import numpy as np
import copy
from tqdm import tqdm
import os
import json
import random
import logging
import torchaudio
import torchaudio.functional as F
import torchaudio.compliance.kaldi as kaldi
from model.asr_feat import ASRFeatExtractor
import logging


logger = logging.getLogger(__name__)


class MultiTaskDataset(IterableDataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()
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
            if dataset_config.add_noise:
                self.noises_list = []
                with open(dataset_config.noises_path, encoding="utf8") as fin:
                    for line in fin:
                        self.noises_list.append(line.strip())
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

        # -- image ---
        self.image_processor = None
        self.processor_path = dataset_config.get("image_processor_path")
        self.max_pixels = dataset_config.get("max_pixels", 512 * 512)

        # -- wav prompt ---
        self.include_transcript = dataset_config.get("include_transcript")

    def get_samples(self, item):
        if self.include_transcript:
            target_ids = self.tokenizer(item['target'])
            samples = len(target_ids)
        else:
            samples = int(item['samples'])
        return samples

    def balance_distribute_task(self, world_size, rank):
        multitask_task_path = os.path.join(self.data_path,"multitask.jsonl")
        tasks_with_samples = []

        with open(multitask_task_path) as f_task:
            for data_index, line in enumerate(f_task):
                item = json.loads(line)
                samples = self.get_samples(item)
                    
                tasks_with_samples.append({
                    'item': item,
                    'samples': samples,
                })

        tasks_with_samples.sort(key=lambda x: x['samples'], reverse=False)
    
        total_samples = sum(task['samples'] for task in tasks_with_samples)
        
        worker_assignments = [[] for _ in range(world_size)]
        worker_sample_counts = [0] * world_size
        
        for task in tasks_with_samples:
            min_worker = min(range(world_size), key=lambda i: worker_sample_counts[i])
            
            worker_assignments[min_worker].append(task['item'])
            worker_sample_counts[min_worker] += task['samples']
        
        if rank == 0:
            for i in range(world_size):
                print(f"Worker {i}: {len(worker_assignments[i])}个任务, {worker_sample_counts[i]:.2f}个样本")
        
        current_tasks = worker_assignments[rank]
        return current_tasks
    

    def naive_distribute_task(self, world_size, rank):
        multitask_task_path = os.path.join(self.data_path,"multitask.jsonl")
        tasks = []
        total_samples_all_workers = [0] * world_size
        total_tasks_all_workers = [0] * world_size

        with open(multitask_task_path) as f_task:
            for data_index, line in enumerate(f_task):
                item = json.loads(line)
                worker_id = data_index % world_size
                total_tasks_all_workers[worker_id] += 1
                total_samples_all_workers[worker_id] += self.get_samples(item)

                if data_index % world_size == rank:
                    tasks.append(item)

        if rank == 0:
            for i in range(world_size):
                print(f"Worker {i}: {total_tasks_all_workers[i]}个任务, {total_samples_all_workers[i]}个样本 ")
        return tasks

    def __iter__(self):
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

        current_tasks = self.balance_distribute_task(total_num_workers, worker_rank)

        for item in current_tasks:
            audio_path = item["path"]
            key = item["key"]
            target = item["target"].lower()
            task = item["task"]

            if re.search(r'\.ark:\d+', audio_path):
                sample_rate, wav_np = kaldiio.load_mat(audio_path)
                audio_raw = wav_np.astype(np.float32) / 32768
            elif audio_path.endswith('wav'):
                audio_raw, sample_rate = soundfile.read(audio_path)
                if len(audio_raw.shape) > 1:
                    audio_raw = audio_raw[:, 0]
            
            if len(audio_raw) / self.sample_rate > self.max_audio_length or len(audio_raw) / self.sample_rate < 0.1: 
                continue

            if self.dataset_config.wav_reverb:
                wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
                wav_tensor = self.wav_reverb(wav_tensor, sample_rate, self.dataset_config.reverb_prob)
                wav_np = wav_tensor.squeeze(0).numpy()

            if self.dataset_config.add_noise:
                wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
                wav_tensor = self.add_noise(wav_tensor, sample_rate, self.dataset_config.noise_prob)
                wav_np = wav_tensor.squeeze(0).numpy()                       

            input_features = torch.from_numpy(audio_raw) 
            with torch.no_grad():
                input_features = torch.nn.functional.layer_norm(input_features , input_features.shape)
            input_feature_length = input_features.shape[0]
            
            # feature postprocessing
            if self.dataset_config.spec_aug:
                spec_aug_conf = self.dataset_config.spec_aug_conf
                input_features = self.spec_aug(input_features, **spec_aug_conf)

            prompt = random.choice(self.multitask_prompt_list[task])
            prompt = self.prompt_template.format(prompt)
            # item[task] = ''
            if task in self.append_info_tasks:
                prompt = prompt.format(item[task])
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_length = len(prompt_ids)
            prompt_ids = torch.tensor(prompt_ids)

            if not self.inference_mode:
                target_ids = self.tokenizer.encode(target)
                target_ids.append(self.tokenizer.eos_token_id)
                target_ids = torch.tensor(target_ids)
                input_ids = torch.cat([prompt_ids, target_ids])
                result = {
                    # 'transcript_ids': target_ids,
                    # 'transcript_length': target_ids.shape[0],
                }
            else:
                input_ids = prompt_ids
                result = {
                    'target': target,
                }
            attention_mask = input_ids.ge(-1)  
            result.update({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask ,
                    "input_features": input_features ,
                    "input_feature_length":input_feature_length,
                    'key': key,
            })

            if task == 'image':
                from PIL import Image
                import math
                image = Image.open(item['image']).convert('RGB')
        
                current_pixels = image.width * image.height
                
                if current_pixels > self.max_pixels:
                    scale = math.sqrt(self.max_pixels / current_pixels)
                    new_size = (int(image.width * scale), int(image.height * scale))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                result['image'] = image
                
            if not self.inference_mode:
                labels = copy.deepcopy(input_ids)
                labels[:prompt_length] = self.tokenizer.default_ignore_token
                result["labels"] = labels
            yield result
            
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
        padding_style = "right"

        # --- input id ---
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id,padding_style = padding_style)
                                    for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False,padding_style = padding_style)
                                        for s in samples])
        
        # --- input feature ---
        input_features_max_length = max([s['input_features'].shape[0] for s in samples])
        input_features = torch.stack([self.pad(s['input_features'], input_features_max_length, 0.0)
                                for s in samples])
        input_feature_length = torch.stack([torch.tensor(s["input_feature_length"]) for s in samples])

        result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask ,
                "input_features": input_features ,
                "input_feature_length":input_feature_length,
        }

        # --- transcript id ---
        if self.include_transcript:
            transcript_ids_max_length = max([s['transcript_ids'].shape[0] for s in samples])
            transcript_ids_length = torch.stack([torch.tensor(s["transcript_length"]) for s in samples])
            transcript_ids = torch.stack([self.pad(s['transcript_ids'], transcript_ids_max_length, 0)
                                        for s in samples])

            result["transcript_ids"] = transcript_ids
            result["transcript_length"] = transcript_ids_length

        self.process_samples(samples)
       
        if self.inference_mode:
            result["keys"] = [s['key'] for s in samples]
            result["targets"] = [s['target'] for s in samples]
        else:
            result["labels"] = torch.stack([self.pad(s['labels'], input_ids_max_length, self.tokenizer.default_ignore_token,padding_style = padding_style)
                                for s in samples])
        # print(result['input_ids'].shape, result['input_features'].shape)
        return result

    def process_samples(self, samples):
        return samples

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

    def wav_reverb(self, x, sample_rate, p=0.3):
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()

        if random.random() > p:
            return y
        
        y = y / (1 << 15)
        rir_path = random.choice(self.rirs_list)
        rir, rir_sr = torchaudio.load(rir_path)
        assert sample_rate == rir_sr
        rir = rir[0:1, :]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)

        corrupted = F.fftconvolve(y, rir)
        T = y.shape[-1]
        corrupted = corrupted[:, :T]

        return corrupted * (1 << 15)
    
    def add_noise(self, x, sample_rate, p=0.3):
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()

        if random.random() > p:
            return y
        
        y = y / (1 << 15)
        noise_path = random.choice(self.noises_list)
        noise, noise_sr = torchaudio.load(noise_path)
        assert sample_rate == noise_sr
        noise = noise[0:1 ,:]

        T_speech = y.shape[1]
        T_noise = noise.shape[1]
        # 处理噪声长度与语音长度不一致的情况
        if T_noise > T_speech:
            start = random.randint(0, T_noise - T_speech)
            noise_segment = noise[:, start : start + T_speech]
        elif T_noise < T_speech:
            noise_segment = torch.zeros_like(y)
            start = random.randint(0, T_speech - T_noise)
            noise_segment[:, start: start + T_noise] = noise
        else:
            noise_segment = noise

        # 生成随机 SNR
        snr_db = random.randint(5, 20)  
        snr_dbs = torch.tensor([snr_db]) 

        # 添加噪声
        noisy_speech = F.add_noise(y, noise_segment, snr_dbs)
        return noisy_speech * (1 << 15)


class MultiTaskDynamicBatchDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, window_class) -> None:
        super().__init__()
        self.dp = dataset
        
        assert window_class is not None
        self.window_class = window_class
        self.collator = self.dp.collator
        self._buffer = []

    def __iter__(self):
        max_sum_len, max_l = 0, 0
        rank = int(os.environ["LOCAL_RANK"])
        for elem in self.dp:
            if not self.window_class(elem, self._buffer):
                self._buffer.append(elem)
            else:
                # if rank == 0:
                #     l = max([ len(_["input_ids"]) + (_["input_feature_length"] // 8 ) -1 for _ in self._buffer])
                #     sum_len = sum([ len(_["input_ids"]) + (_["input_feature_length"] // 8 ) -1 for _ in self._buffer])
                #     if sum_len > max_sum_len:
                #         max_sum_len = sum_len
                #     if l > max_l:
                #         max_l = l
                #     print(max_sum_len, sum_len, max_l, l)
                buffer_to_yield = self._buffer
                self._buffer = [elem]
                yield buffer_to_yield
        if len(self._buffer) > 0:
            yield self._buffer
        self._buffer = []
         
    
def window_class(elem,buffer,max_frame_length,ds_rate):
    if len(buffer) == 0:
        return False
    max_frame = max(len(elem["input_ids"]) + (elem["input_feature_length"] // ds_rate) - 1,max([ len(_["input_ids"]) + (_["input_feature_length"] // ds_rate ) -1 for _ in buffer]))
    return (len(buffer) + 1) * max_frame > max_frame_length


def get_speech_dataset(dataset_config, tokenizer, split):
    ds_config = copy.deepcopy(dataset_config)
    if split != "train":
        ds_config.spec_aug = False
        ds_config.wav_reverb = False
        ds_config.add_noise = False
    dataset = MultiTaskDataset(ds_config, tokenizer, split)
    if split == "train":
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class, max_frame_length=ds_config.train_max_frame_length, ds_rate=ds_config.ds_rate))
    else:
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class, max_frame_length=ds_config.eval_max_frame_length, ds_rate=ds_config.ds_rate))
    return dataset



    
