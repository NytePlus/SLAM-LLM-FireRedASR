import torch
import torch.nn as nn
from typing import Optional
from funasr.models.bicif_paraformer.cif_predictor import cif

class Adapter(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.ds, feat_dim * self.ds
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class CifPredictorV3(torch.nn.Module):
    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        tf2torch_tensor_name_prefix_torch="predictor",
        tf2torch_tensor_name_prefix_tf="seq2seq/cif",
        smooth_factor2=1.0,
        noise_threshold2=0
    ):
        super(CifPredictorV3, self).__init__()

        self.pad = torch.nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = torch.nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = torch.nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

        self.smooth_factor2 = smooth_factor2
        self.noise_threshold2 = noise_threshold2

    def forward(
        self,
        hidden,
        target_label=None,
        mask=None,
        ignore_id=-1,
        target_label_length=None,
    ):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))

        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mask is not None:
            alphas = alphas.squeeze(-1)
            alphas = alphas * mask
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            norm_alphas = alphas * (target_length / token_num).unsqueeze(1)

        acoustic_embeds, cif_peak = cif(hidden, norm_alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak


class FunasrCIFAdapter(nn.Module):
    """
    Wrapper for FunASR CIF to match LLM integration interface.
    - Outputs: [B, transcript_length, llm_dim]
    - Returns alphas for debug / loss
    """
    def __init__(self, encoder_dim: int, llm_dim: int, cif_kwargs: dict = None):
        super().__init__()
        if cif_kwargs is None:
            cif_kwargs = {
                "smooth_factor2": 0.25,
                "noise_threshold2": 0.01
            } 

        self.cif_module = CifPredictorV3(idim=encoder_dim, l_order=2, r_order=1, **cif_kwargs)
        self.linear_out = nn.Linear(encoder_dim, llm_dim)

    def forward(self, x: torch.Tensor, valid_lens, transcript_length: torch.Tensor = None):
        """
        x: [B, T, encoder_dim]  -> encoder output
        valid_lens: [B]         -> valid length of each sequence
        transcript_length: [B]  -> target transcript length
        """
        mask = None
        if valid_lens is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < valid_lens.unsqueeze(1)

        # Call official CIF
        acoustic_embeds, token_num, alphas, cif_peak = self.cif_module(
            hidden=x,
            mask=mask,
            target_label_length=transcript_length
        )

        quantity_loss = None
        if transcript_length is not None:
            quantity_loss = torch.abs(token_num - transcript_length).mean()

        outputs = self.linear_out(acoustic_embeds)

        return outputs, transcript_length, quantity_loss

class CIFAdapter(nn.Module):
    def __init__(self, encoder_dim, llm_dim):
        super().__init__()
        self.linear1 = nn.Linear(encoder_dim, llm_dim + 1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x, valid_lens, transcript_length = None): # [batch_size]
        x = self.linear1(x).to(torch.float32)

        enc = x[:, :, :-1] # [batch_size, seq_len, llm_dim]
        alpha = torch.sigmoid(x[:, :, -1]) # [batch_size, seq_len]
        batch_size, seq_len, llm_dim = enc.shape
        device = alpha.device

        mask = torch.arange(alpha.size(1), device=device).unsqueeze(0) < valid_lens.unsqueeze(1)
        alpha = alpha * mask

        quantity_loss = None
        if transcript_length is not None:
            quantity_loss = torch.abs(alpha.sum(dim=1) - transcript_length).mean()
            alpha = alpha * (transcript_length.unsqueeze(-1) / (alpha.sum(dim=1, keepdim=True)))

        outputs, final_valid_lens = [], []
        for b in range(batch_size):
            fired_vectors, accum, acc_vec = [], 0.0, torch.zeros((llm_dim), device=device)
            valid_len = valid_lens[b].item()

            for t in range(valid_len):
                a_t = alpha[b, t]
                h_t = enc[b, t]

                accum = accum + a_t
                acc_vec = acc_vec + a_t * h_t

                if accum >= 1.0 and t == valid_len - 1:
                    fired_vectors.append(acc_vec)

                    accum = accum - 1.0
                    acc_vec = min(1.0, accum) * h_t

            miss_length = transcript_length[b] - len(fired_vectors)
            fired_vectors = torch.stack(fired_vectors, dim=0)
            if miss_length > 0:
                fired_vectors = torch.cat([fired_vectors, torch.zeros((miss_length, llm_dim), device=device)], dim=0)
            outputs.append(fired_vectors)
            final_valid_lens.append(fired_vectors.shape[0])

        max_len = max(final_valid_lens)
        out_dim = enc.shape[-1]

        padded = x.new_zeros((batch_size, max_len, out_dim))
        for b in range(batch_size):
            padded[b, :outputs[b].size(0)] = outputs[b]
        final_valid_lens = torch.tensor(final_valid_lens)

        output = self.relu(padded)
        output = self.linear2(output)

        return output, final_valid_lens, quantity_loss


class EncoderProjectorConcat(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.ds, feat_dim * self.ds
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class EncoderProjectorCov1d(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.ds, stride=self.ds, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x