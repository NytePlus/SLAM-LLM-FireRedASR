import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from funasr.models.bicif_paraformer.cif_predictor import cif
from .cif import *

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
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x



class SphereEmbedding(nn.Module):
    def __init__(self, R, K):
        super().__init__()
        self.R = R
        self.K = K

        self.max_R = 0

    def forward(self, x, verbose=False):
        x_scaled = x / torch.sqrt(torch.tensor(self.K, dtype=x.dtype, device=x.device)) # (bs, seq_len, feat_dim)

        norm_sq = torch.sum(x_scaled ** 2, dim=-1, keepdim=True) # (bs, seq_len)
        R_sq = torch.tensor(self.R**2, dtype=x.dtype, device=x.device)
        padding_value = torch.sqrt(torch.clamp(R_sq - norm_sq, min=0.0)) # (bs, seq_len)
        padding = padding_value.expand(*x.shape[:-1], self.K) # (bs, seq_len, K)

        out = torch.cat([x_scaled, padding], dim=-1) / self.R # (bs, seq_len, feat_dim + K)
        if verbose:
            import math
            self.max_R = max(self.max_R, math.sqrt(norm_sq.max()))
            if self.max_R > self.R:
                print(f'warning: max_R {self.max_R}')
        return out

class KernelLinear(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.kernel = SphereEmbedding(8, 1)
        self.projector = EncoderProjectorConcat(encoder_dim + 1, llm_dim, downsample_rate)

    def forward(self, x):
        x_type = torch.bfloat16
        kernel_out = self.kernel(x)
        out = self.projector(kernel_out)
        out = F.normalize(out, p=2, dim=-1).to(x_type)
        return out
