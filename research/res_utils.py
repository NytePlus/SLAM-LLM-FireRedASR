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