import torch
from torch.utils.data import Dataset,IterableDataset
import whisper
import kaldiio
import types
from PIL import Image
import math
import re
import soundfile
from functools import partial
import torch.distributed as dist
import copy
import numpy as np
import copy
import math
from tqdm import tqdm
import os
import json
import random
import logging
import torchaudio
import torchaudio.functional as F
import torchaudio.compliance.kaldi as kaldi
from transformers import AutoProcessor
import logging


logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
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
        if self.processor_path:
            self.image_processor = AutoProcessor.from_pretrained(
                self.processor_path, 
                max_pixels=self.max_pixels
                ).image_processor

        # -- wav prompt ---
        self.include_transcript = dataset_config.get("include_transcript")

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

        self.current_tasks = self.naive_distribute_task(total_num_workers, worker_rank)

    def get_samples(self, item):
        if self.include_transcript:
            target_ids = self.tokenizer(item['target'])
            samples = len(target_ids)
        else:
            samples = int(item['samples'])
        return samples

    def balance_distribute_task(self, world_size, rank, verbose=False):
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
        
        worker_assignments = [[] for _ in range(world_size)]
        worker_sample_counts = [0] * world_size
        
        for task in tasks_with_samples:
            min_worker = min(range(world_size), key=lambda i: worker_sample_counts[i])
            
            worker_assignments[min_worker].append(task['item'])
            worker_sample_counts[min_worker] += task['samples']
        
        if rank == 0 and verbose:
            for i in range(world_size):
                print(f"Worker {i}: {len(worker_assignments[i])}个任务, {worker_sample_counts[i]:.2f}个样本")
        
        current_tasks = worker_assignments[rank]
        return current_tasks
    

    def naive_distribute_task(self, world_size, rank, verbose=False):
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

        if rank == 0 and verbose:
            for i in range(world_size):
                print(f"Worker {i}: {total_tasks_all_workers[i]}个任务, {total_samples_all_workers[i]}个样本 ")
        return tasks

    def __getitem__(self, idx, verbose=False):
        item = self.current_tasks[idx]
        audio_path = item["path"]
        
        DATA_DIR = os.environ.get('DATA_DIR')
        if DATA_DIR:
            audio_path = audio_path.replace('/aistor/sjtu/hpc_stor01/home/wangchencheng/data', DATA_DIR)
        key = item["key"]
        if self.dataset_config.lower:
            target = item["target"].lower()
        else:
            target = item["target"]
        task = item["task"]

        if re.search(r'\.ark:\d+', audio_path):
            sample_rate, wav_np = kaldiio.load_mat(audio_path)
            audio_raw = wav_np.astype(np.float32)
        elif audio_path.endswith('wav'):
            audio_raw, sample_rate = soundfile.read(audio_path)
            if len(audio_raw.shape) > 1:
                audio_raw = audio_raw[:, 0]
        
        # if len(audio_raw) / self.sample_rate > self.max_audio_length or len(audio_raw) / self.sample_rate < 0.1: 
        #     print(f'Skip {audio_path}(len {len(audio_raw) / self.sample_rate: .2f}s)')
        #     continue

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

        if verbose:
            def analyze_feature_norms(feature, name="Feature", num_bins=5):
                # 展平以便全局统计
                flat_norms = feature.flatten()
                
                # 2. 计算全局统计量
                mean_val = flat_norms.mean().item()
                std_val = flat_norms.std().item()
                var_val = flat_norms.var().item()
                max_val = flat_norms.max().item()
                min_val = flat_norms.min().item()
                
                print(f"统计报告: 【{name}】")
                print("-" * 65)
                print(f"形状: {list(feature.shape)} | 总 Token 数: {flat_norms.numel()}")
                print(f"全局统计 -> 均值: {mean_val:.6f} | 标准差: {std_val:.6f} | 方差: {var_val:.6f}")
                print(f"数值范围 -> 最小: {min_val:.6f} | 最大: {max_val:.6f}")
                print("-" * 65)
                
                # 3. 区间均值与方差统计 (Binning)
                print(f"{'Norm 分布区间':<25} | {'均值 (Mean)':<12} | {'方差 (Var)':<12} | {'占比 (%)'}")
                
                # 使用 linspace 创建等间距区间
                bins = torch.linspace(min_val, max_val, num_bins + 1)
                
                for i in range(num_bins):
                    lower, upper = bins[i], bins[i+1]
                    # 处理边界：最后一个区间包含最大值
                    if i == num_bins - 1:
                        mask = (flat_norms >= lower) & (flat_norms <= upper)
                    else:
                        mask = (flat_norms >= lower) & (flat_norms < upper)
                        
                    selected = flat_norms[mask]
                    
                    if selected.numel() > 0:
                        bin_mean = selected.mean().item()
                        bin_var = selected.var().item()
                        percentage = (selected.numel() / flat_norms.numel()) * 100
                        print(f"[{lower:>7.5f}, {upper:>7.5f}] | {bin_mean:>12.5f} | {bin_var:>12.5f} | {percentage:>8.5f}%")
                    else:
                        print(f"[{lower:>7.5f}, {upper:>7.5f}] | {'-':^12} | {'-':^12} | {0.00:>8.5f}%")
                        
                print("-" * 65 + "\n")

            input_features2 = torch.from_numpy(audio_raw / 32768)
            analyze_feature_norms(input_features2, "LN(raw/32768)")
            with torch.no_grad():
                input_features2 = torch.nn.functional.layer_norm(input_features2 , input_features2.shape)
            analyze_feature_norms(input_features,  "LN(ln)")
            analyze_feature_norms(input_features2, "LN(ln/32768)")
            input('')
        
        # feature postprocessing
        if self.dataset_config.spec_aug:
            spec_aug_conf = self.dataset_config.spec_aug_conf
            input_features = self.spec_aug(input_features, **spec_aug_conf)

        prompt = random.choice(self.multitask_prompt_list[task])
        prompt = self.prompt_template.format(prompt)
        if task in self.append_info_tasks:
            if self.dataset_config.lower:
                task_text = item[task].lower()
            else:
                task_text = item[task]
            prompt = prompt.format(task_text)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids)

        if not self.inference_mode:
            # print(f'target is: {target}')
            target_ids = self.tokenizer.encode(target)
            target_ids.append(self.tokenizer.eos_token_id)
            target_ids = torch.tensor(target_ids)
            input_ids = torch.cat([prompt_ids, target_ids])
            result = {
                'transcript_ids': target_ids,
                'transcript_length': target_ids.shape[0],
            }
        else:
            input_ids = prompt_ids
            result = {}
        attention_mask = input_ids.ge(-1)  
        result.update({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_features": input_features,
                "input_feature_length":input_feature_length,
                'key': key,
                'target': target,
        })

        if task == 'image':
            result['image'] = np.zeros((14, 14, 3), dtype=np.uint8)
            if item['image'] != '':
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
        return result
    
    def __len__(self):
        return len(self.current_tasks)
            
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
    
    def debug_verify_labels(self, input_ids, labels, tokenizer, is_attn_mask, num_samples=2):
        """
        可视化验证 labels 是否正确。
        绿色/明亮部分：计算 Loss 的部分 (Target)
        灰色/暗淡部分：被忽略的部分 (Prompt/Padding)
        """
        for i in range(min(len(input_ids), num_samples)):
            ids = input_ids[i].tolist()
            lbs = labels[i].tolist()

            segments = []
            current_tokens = []
            current_is_masked = None

            for token_id, label_id in zip(ids, lbs):
                token_text = tokenizer.decode([token_id])

                if is_attn_mask:
                    is_masked = label_id
                else:
                    is_masked = (label_id == -100)

                if current_is_masked is None:
                    current_is_masked = is_masked

                # 状态变化，切段
                if is_masked != current_is_masked:
                    segments.append((current_is_masked, "".join(current_tokens)))
                    current_tokens = []
                    current_is_masked = is_masked

                current_tokens.append(token_text)
            if current_tokens:
                segments.append((current_is_masked, "".join(current_tokens)))

            rendered = []
            for is_masked, text in segments:
                if is_masked:
                    rendered.append(f"[[{text}]]")
                else:
                    rendered.append(f"**{text}**")

            print("="*50)
            print("".join(rendered))
    
    def collator(self, samples):
        assert samples is not None
        padding_style = "left"

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

        # --- image ---
        if self.image_processor:
            pixel_values = []
            pixel_value_lens = []
            grid_thw = []

            for s in samples:
                image_out = self.image_processor(images=s['image'], return_tensors="pt")
                pixel_values.append(image_out['pixel_values'])
                pixel_value_lens.append(image_out['pixel_values'].shape[0])
                grid_thw.append(image_out['image_grid_thw'])
            result['pixel_values'] = torch.stack([self.pad(p, max(pixel_value_lens), False, padding_style = padding_style)
                                        for p in pixel_values])
            result['pixel_values_length'] = torch.stack([torch.tensor(l) for l in pixel_value_lens])
            result['grid_thw'] = torch.cat(grid_thw)

        # --- transcript id ---
        if self.include_transcript:
            transcript_ids_max_length = max([s['transcript_ids'].shape[0] for s in samples])
            transcript_ids_length = torch.stack([torch.tensor(s["transcript_length"]) for s in samples])
            transcript_ids = torch.stack([self.pad(s['transcript_ids'], transcript_ids_max_length, 0)
                                        for s in samples])

            result["transcript_ids"] = transcript_ids
            result["transcript_length"] = transcript_ids_length
       
        if self.inference_mode:
            result["keys"] = [s['key'] for s in samples]
            result["targets"] = [s['target'] for s in samples]
        else:
            result["labels"] = torch.stack([self.pad(s['labels'], input_ids_max_length, self.tokenizer.default_ignore_token,padding_style = padding_style)
                                for s in samples])
            
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

class FixedBatchDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, batch_size) -> None:
        super().__init__()
        self.dp = dataset
        self.batch_size = batch_size
        self.collator = self.dp.collator
        self.buffer = []

    def __iter__(self):
        for elem in self.dp:
            self.buffer.append(elem)
            if len(self.buffer) == self.batch_size:
                yield self.buffer
                self.buffer = []
        if len(self.buffer) > 0:
            yield self.buffer
        self.buffer = []
         
    
def window_class(elem,buffer,max_frame_length,ds_rate):
    if len(buffer) == 0:
        return False
    max_frame = max(len(elem["input_ids"]) + (elem["input_feature_length"] // ds_rate) - 1,max([ len(_["input_ids"]) + (_["input_feature_length"] // ds_rate ) -1 for _ in buffer]))
    return (len(buffer) + 1) * max_frame > max_frame_length


def get_speech_dataset(dataset_config, tokenizer, split, batching_strategy="dynamic", iterable=False):
    ds_config = copy.deepcopy(dataset_config)
    if split != "train":
        ds_config.spec_aug = False
        ds_config.wav_reverb = False
        ds_config.add_noise = False
    dataset = MultiTaskDataset(ds_config, tokenizer, split)
    if not iterable:
        return dataset
    if batching_strategy == "dynamic":
        max_frame_length = ds_config.train_max_frame_length if split == "train" else ds_config.eval_max_frame_length
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class, max_frame_length=max_frame_length, ds_rate=ds_config.ds_rate))

    return dataset

    
