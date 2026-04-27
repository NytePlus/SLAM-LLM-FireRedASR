import torch

def decode_replace_audios(token_ids, tokenizer, audio_length):
    decoded = []

    for tok in token_ids:
        if tok == tokenizer.default_speech_token:
            decoded.extend(['<AUDIO>'] * audio_length)
        else:
            decoded.append(tokenizer.decode([tok], skip_special_tokens=False))
    return decoded

def find_generated_spans(input_ids, labels):
    B, T = input_ids.shape
    starts, lengths = [], []

    for b in range(B):
        valid_pos = (labels[b] != -100).nonzero(as_tuple=True)[0]

        if valid_pos.numel() == 0:
            starts.append(None)
            lengths.append(0)
            continue

        start_idx = valid_pos[0].item()
        end_idx   = valid_pos[-1].item() + 1
        length    = end_idx - start_idx

        starts.append(start_idx)
        lengths.append(length)

    return starts, lengths

def remove_batch_audio_feature(batch, audio_token_id):
    """
    将 batch 伪装成“无音频”状态，同时保留张量结构。
    1. 将 input_features 变为 (batch_size, 0, embed_dim)
    2. 将 input_feature_length 全部置为 0
    3. 从序列中移除 audio_token_id 并重新对齐
    """
    # 1. 处理音频特征张量 (保持最后一维，前面归零)
    if 'input_features' in batch:
        device = batch['input_features'].device
        dtype = batch['input_features'].dtype
        B, _ = batch['input_features'].shape
        batch['input_features'] = torch.empty((B, 0), device=device, dtype=dtype)
        
    if 'input_feature_length' in batch:
        device = batch['input_feature_length'].device
        B = batch['input_feature_length'].size(0)
        # 长度全部设为 0
        batch['input_feature_length'] = torch.zeros(B, device=device, dtype=torch.int64)

    # 2. 从文本序列中剔除音频占位符 (这一步必须做，否则 LLM 会在词表中找这个特殊 Token)
    if 'input_ids' in batch:
        input_ids = batch['input_ids']
        non_audio_mask = (input_ids != audio_token_id)
        
        # 准备重构的列表
        new_ids, new_mask, new_labels = [], [], []
        has_mask = 'attention_mask' in batch
        has_labels = 'labels' in batch

        for i in range(input_ids.size(0)):
            row_m = non_audio_mask[i]
            new_ids.append(input_ids[i][row_m])
            if has_mask:
                new_mask.append(batch['attention_mask'][i][row_m])
            if has_labels:
                new_labels.append(batch['labels'][i][row_m])

        # 3. 重新对齐 Padding
        from torch.nn.utils.rnn import pad_sequence
        batch['input_ids'] = pad_sequence(new_ids, batch_first=True, padding_value=0)
        
        if has_mask:
            # 你的 mask 是 bool 类型，pad 值为 False
            batch['attention_mask'] = pad_sequence(new_mask, batch_first=True, padding_value=False)
            
        if has_labels:
            # labels 默认忽略值为 -100
            batch['labels'] = pad_sequence(new_labels, batch_first=True, padding_value=-100)

    return batch