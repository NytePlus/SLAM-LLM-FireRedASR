import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 参数
b, t1, t2, d = 2, 3, 5, 3  # batch, max non-text tokens, text tokens, embedding dim
placeholder_token_id = 3
image_token_id = 4
pad_token_id = 0
dtype = torch.float32

num_nontext_tokens = torch.tensor([3, 1])
nontext_features = torch.ones((b, t1, d), dtype=dtype) * 1
for i in range(b):
    nontext_features[i, num_nontext_tokens[i]:, :] = pad_token_id

inputs_embeds = torch.ones((b, t2, d), dtype=dtype) * 2

input_ids = torch.ones((b, t2), dtype=torch.long) * 2
for i in range(b):
    input_ids[i, 0] = placeholder_token_id
    input_ids[i, 1] = image_token_id

    inputs_embeds[i, 0] = torch.zeros((d), dtype=dtype)
    inputs_embeds[i, 1] = torch.zeros((d), dtype=dtype)

num_text_tokens = torch.tensor([5, 3])
attention_mask = torch.ones((b, t2), dtype=torch.long)
for i in range(b):
    attention_mask[i, num_text_tokens[i]:] = pad_token_id
    attention_mask[i, :num_text_tokens[i]] = 1

labels = input_ids.clone()

# ✅ 打印形状和检查
print("nontext_features.shape:", nontext_features.shape,)
print("num_nontext_tokens.shape:", num_nontext_tokens.shape, num_nontext_tokens.dtype)
print("inputs_embeds.shape:", inputs_embeds.shape)
print("input_ids.shape:", input_ids.shape)
print("attention_mask.shape:", attention_mask.shape)
print("labels.shape:", labels.shape)

def _merge_input_ids_with_nontext_features(
            nontext_features, 
            num_nontext_tokens, 
            inputs_embeds, 
            input_ids, 
            attention_mask, 
            labels, 
            placeholder_token_id
        ):
        device = torch.npu.set_device(1)
        nontext_features = nontext_features.to(device)
        num_nontext_tokens = num_nontext_tokens.to(device)
        inputs_embeds = inputs_embeds.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        num_features, max_feature_tokens, embed_dim = nontext_features.shape
        
        # Create mask for valid non-text features
        nontext_features_mask = torch.arange(max_feature_tokens).expand(num_features, max_feature_tokens).to(
            num_nontext_tokens.device
        ) < num_nontext_tokens.unsqueeze(1)
        print(torch.arange(max_feature_tokens).expand(num_features, max_feature_tokens).shape, num_nontext_tokens.unsqueeze(1).shape)
        masked_nontext_features = nontext_features[nontext_features_mask].view(-1, embed_dim)
        
        batch_size, sequence_length = input_ids.shape
        
        # Detect padding direction
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                left_padding = True
            else:
                raise ValueError(f"Invalid attention_mask: {attention_mask}")

        # 1. Create a mask to know where placeholder tokens are
        placeholder_mask = input_ids == placeholder_token_id
        num_placeholder_tokens = torch.sum(placeholder_mask, dim=-1)

        # Ensure all tensors are on the correct device
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_nontext_tokens = num_nontext_tokens.to(target_device)
        
        # Find indices of non-placeholder tokens (regular text tokens)
        batch_indices, non_placeholder_indices = torch.where(
            (input_ids != placeholder_token_id) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Each placeholder token will be replaced by `num_nontext_tokens - 1` text tokens
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[placeholder_mask] = num_nontext_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_feature_pad = max_token_num - 1 - new_token_positions[:, -1]
        
        if left_padding:
            new_token_positions += nb_feature_pad[:, None]  # offset for left padding
        
        text_to_overwrite = new_token_positions[batch_indices, non_placeholder_indices]
        batch_indices, non_placeholder_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_placeholder_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), 0, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings for text tokens
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_placeholder_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_placeholder_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_placeholder_indices]
        
        # Handle labels if provided
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            ignore_id = -1
            final_labels = torch.full(
                (batch_size, max_token_num), ignore_id, dtype=input_ids.dtype, device=inputs_embeds.device
            ).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_placeholder_indices]
        
        # 5. Fill the embeddings corresponding to the non-text features
        feature_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        feature_to_overwrite[batch_indices, text_to_overwrite] = False
        
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        feature_to_overwrite &= val

        # Validate that we have the correct number of feature tokens
        if feature_to_overwrite.sum() != num_nontext_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of placeholder tokens is {num_placeholder_tokens} while"
                f" the number of non-text features given to the model is {num_features}. This prevents correct indexing."
            )

        # Fill non-text features
        final_embedding[feature_to_overwrite] = (
            masked_nontext_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= feature_to_overwrite
        
        # Generate position IDs
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids = _merge_input_ids_with_nontext_features(nontext_features, 
            num_nontext_tokens, 
            inputs_embeds, 
            input_ids, 
            attention_mask, 
            labels, 
            placeholder_token_id)

print(final_embedding.shape, final_attention_mask.shape, final_labels.shape, position_ids.shape, final_input_ids.shape)

print(final_embedding)

print(final_attention_mask)

t3 = 5
num_audio_tokens = torch.tensor([4, 5])
audio_features = torch.ones((b, t3, d), dtype=dtype) * 3
for i in range(b):
    audio_features[i, num_audio_tokens[i]:, :] = pad_token_id

final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids = _merge_input_ids_with_nontext_features(audio_features, 
            num_audio_tokens, 
            final_embedding, 
            final_input_ids, 
            final_attention_mask, 
            final_labels, 
            image_token_id)

print(final_embedding)

print(final_attention_mask)

tokenizer = AutoTokenizer.from_pretrained("/aistor/sjtu/hpc_stor01/home/xiyu/models/vicuna-7b-v1.5")

DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IGNORE_TOKEN = -100
special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN, DEFAULT_IMAGE_TOKEN]}
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.default_ignore_token = DEFAULT_IGNORE_TOKEN
tokenizer.default_speech_token = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
tokenizer.default_image_token = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

print(tokenizer.default_image_token)