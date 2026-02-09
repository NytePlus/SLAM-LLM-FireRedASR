import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List

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


class FunasrCIFAdapter(nn.Module): # 3.4s/it
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

class CIFAdapter(nn.Module): # 3.4s/it
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
class CIFAdapter2(nn.Module): # 4s/it
    def __init__(self, 
            encoder_embed_dim, 
            cif_embedding_dim, 
            cif_threshold=0.99, 
            produce_weight_type="conv", 
            conv_cif_width=3, 
            conv_cif_dropout=0.1, 
            apply_scaling=True, 
            apply_tail_handling=True, 
            tail_handling_firing_threshold=0.5
        ):
        super().__init__()

        # Load configurations
        self.cif_threshold = cif_threshold
        self.cif_output_dim = cif_embedding_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.produce_weight_type = produce_weight_type
        self.conv_cif_width = conv_cif_width
        self.conv_cif_dropout = conv_cif_dropout
        self.apply_scaling = apply_scaling
        self.apply_tail_handling = apply_tail_handling
        self.tail_handling_firing_threshold = tail_handling_firing_threshold

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.dense_proj = self.Linear(
                self.encoder_embed_dim, self.encoder_embed_dim)
            self.weight_proj = self.Linear(
                self.encoder_embed_dim, 1)
        elif self.produce_weight_type == "conv":
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                self.encoder_embed_dim,
                self.conv_cif_width,
                stride=1, padding=int(self.conv_cif_width / 2),
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            )
            self.conv_dropout = torch.nn.Dropout(
                p=self.conv_cif_dropout)
            self.weight_proj = self.Linear(
                self.encoder_embed_dim, 1)
        else:
            self.weight_proj = self.Linear(
                self.encoder_embed_dim, 1)

        # Build the final projection layer (if encoder_embed_dim is not equal to cif_output_dim)
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = self.Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False)
            
    def make_padding_mask(self, encoder_valid_length, max_len=None):
        """
        Args:
            encoder_valid_length: LongTensor [B]
            max_len: int, optional
        Returns:
            padding_mask: BoolTensor [B, T], padding=1, valid=0
        """
        B = encoder_valid_length.size(0)
        if max_len is None:
            max_len = encoder_valid_length.max().item()

        seq_range = torch.arange(
            max_len, device=encoder_valid_length.device
        ).unsqueeze(0)                          # [1, T]

        padding_mask = seq_range >= encoder_valid_length.unsqueeze(1)
        return padding_mask


    def forward(self, encoder_outputs, encoder_valid_length, target_lengths=None):
        """
        Args:
            encoder_outputs: a dictionary that includes
                encoder_raw_out:
                    the raw outputs of acoustic encoder, with shape B x T x C
                encoder_padding_mask:
                    the padding mask (whose padded regions are filled with ones) of encoder outputs, with shape B x T
            target_lengths: the length of targets (necessary when training), with shape B
        Return:
            A dictionary:
                cif_out:
                    the cif outputs
                cif_out_padding_mask:
                    the padding infomation for cif outputs (whose padded regions are filled with zeros)
                quantity_out:
                    the sum of weights for the calculation of quantity loss
        """

        # Collect inputs
        B, T, C = encoder_outputs.shape
        encoder_padding_mask = self.make_padding_mask(
            encoder_valid_length, max_len=T
        ) # B x T

        # Produce weights for integration (accumulation)
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_outputs)
            weight = torch.sigmoid(sig_input)
        # weight has shape B x T x 1

        not_padding_mask = ~encoder_padding_mask
        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()  # weight has shape B x T
        org_weight = weight

        # Apply scaling strategies
        if self.training and self.apply_scaling and target_lengths is not None:
            # Conduct scaling when training
            weight_sum = weight.sum(-1)             # weight_sum has shape B
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1)    # normalize_scalar has shape B x 1
            weight = weight * normalize_scalar

        # Prepare for Integrate and fire
        batch_size = encoder_outputs.size(0)
        max_length = encoder_outputs.size(1)
        encoder_embed_dim = encoder_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        device = encoder_outputs.device
        accumulated_weights = torch.zeros(batch_size, 0).to(device)
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim).to(device)
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim).to(device)

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = torch.zeros([batch_size]).to(device) if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = \
                torch.zeros([batch_size, encoder_embed_dim]).to(device) if i == 0 else accumulated_states[:, i - 1, :]

            # Decide whether to fire a boundary
            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(dim=-1)
            # cur_is_fired with shape B x 1

            # Update the accumulated weights
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape B x 1
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            # prev_accumulated_weight also has shape B x 1
            remained_weight = torch.ones_like(prev_accumulated_weight).to(device) - prev_accumulated_weight
            # remained_weight with shape B x 1

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight)  # B x 1

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_outputs[:, i, :])  # B x C

            # Obtain fired state of current step:
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).to(device))  # B x C

            # Handle the tail
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than tail_handling_firing_threshold,
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i == padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                    # shape B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim]) <= self.tail_handling_firing_threshold,
                        # shape B x C
                        torch.zeros([batch_size, encoder_embed_dim]).to(device),
                        # less equal than tail_handling_firing_threshold, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than tail_handling_firing_threshold, normalized and kept.
                    ), cur_fired_state)
                # shape B x T

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero vectors
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).to(device) >
                padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).to(device), cur_fired_state)

            # Update accumulation-related values: T_c stands for the length of integrated features
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1)                       # B x T_c
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1)     # B x T_c x C
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1)                 # B x T_c x C

        # Extract cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()    # B x T_c
        fired_utt_length = fired_marks.sum(-1)                          # B
        fired_max_length = fired_utt_length.max().int()                 # The maximum of fired times in current batch
        cif_outputs = torch.zeros([0, fired_max_length, encoder_embed_dim]).to(device)

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
            assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == index] for index in range(num_partitions)]

        # Loop over all samples
        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]             # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)         # The total number of firing
            pad_length = fired_max_length - cur_utt_length  # Get padded length
            cur_utt_output = torch.cat(
                (cur_utt_output, torch.full([pad_length, encoder_embed_dim], 0.0).to(device)), dim=0
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to 1 x T_c x C

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # cif_out_padding_mask has shape B x T_c, where locations with value 0 are the padded locations.

        if self.training:
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        quantity_loss = None
        if target_lengths is not None:
            quantity_loss = torch.abs(quantity_out - target_lengths).mean()

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        cif_valid_length = cif_out_padding_mask.sum(dim=1)

        return cif_outputs, cif_valid_length, quantity_loss

    def Linear(self, in_features, out_features, bias=True):
        m = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(m.weight)
        if bias:
            nn.init.constant_(m.bias, 0.0)
        return m
    
class FlashCIFAdapter(nn.Module): # 1.3s/it
    """
    Wrapper for FunASR CIF using `cif_function` with input MLP for LLM integration.
    - Inputs go through a small MLP before generating alpha.
    - Outputs: [B, transcript_length, llm_dim]
    - Returns alphas for debug / quantity loss.
    """
    def __init__(self, encoder_dim: int, llm_dim: int, mlp_hidden: int = None, cif_kwargs: dict = None):
        super().__init__()
        if cif_kwargs is None:
            cif_kwargs = {}
        self.cif_kwargs = cif_kwargs
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        if mlp_hidden is None:
            mlp_hidden = llm_dim

        # Input MLP: encoder_dim -> mlp_hidden -> encoder_dim
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, encoder_dim)
        )

        # Linear projection to generate alpha (frame-level CIF weights)
        self.alpha_layer = nn.Linear(encoder_dim, 1)
        nn.init.xavier_uniform_(self.alpha_layer.weight)
        if self.alpha_layer.bias is not None:
            nn.init.constant_(self.alpha_layer.bias, 0.0)

        # Project CIF output to LLM embedding dimension
        self.linear_out = nn.Linear(encoder_dim, llm_dim)

    def forward(
        self,
        x: torch.Tensor,                # [B, S, encoder_dim]
        valid_lens: Optional[Tensor],   # [B]
        transcript_length: Optional[Tensor] = None  # [B]
    ):
        """
        x: [B, S, encoder_dim]  -> encoder output
        valid_lens: [B]         -> valid length of each sequence
        transcript_length: [B]  -> target transcript length
        """
        # Build padding mask (1 = padded)
        mask = None
        if valid_lens is not None:
            seq_len = x.size(1)
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            mask = mask.int()

        # Pass inputs through MLP
        mlp_out = self.mlp(x)  # [B, S, encoder_dim]

        # Compute alpha (weights) for each frame
        alpha = torch.sigmoid(self.alpha_layer(mlp_out)).squeeze(-1)  # [B, S]

        # Call cif_function using MLP output as inputs
        cif_out_dict = cif_function(
            inputs=mlp_out,
            alpha=alpha,
            padding_mask=mask,
            target_lengths=transcript_length,
            **self.cif_kwargs
        )

        # Extract CIF output and valid lengths
        acoustic_embeds = cif_out_dict["cif_out"][0]        # [B, T, C]
        token_num = cif_out_dict["alpha_sum"][0] # [B], number of tokens after CIF

        # Compute quantity loss if target lengths are given
        quantity_loss = None
        if transcript_length is not None and token_num is not None:
            quantity_loss = torch.abs(token_num - transcript_length.float()).mean()

        # Project to LLM dim
        outputs = self.linear_out(acoustic_embeds)       # [B, T, llm_dim]

        return outputs, transcript_length, quantity_loss

def cif_function(
    inputs: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
    unbound_alpha: bool = False
) -> Dict[str, List[Tensor]]:
    r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235

    Shapes:
        N: batch size
        S: source (encoder) sequence length
        C: source feature dimension
        T: target sequence length

    Args:
        inputs (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            inputs. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the inputs. 1 is padding, 0 is not.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4
        unbound_alpha (bool, optional): Whether to check if 0 <= alpha <= 1.

    Returns -> Dict[str, List[Tensor]]: Key/values described below.
        cif_out: (N, T, C) The output integrated from the source.
        cif_lengths: (N,) The output length for each element in batch.
        alpha_sum: (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays: (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights: (N,) During inference, return the tail.
        scaled_alpha: (N, S) alpha after applying weight scaling.
        cumsum_alpha: (N, S) cumsum of alpha after scaling.
        right_indices: (N, S) right scatter indices, or floor(cumsum(alpha)).
        right_weights: (N, S) right scatter weights.
        left_indices: (N, S) left scatter indices.
        left_weights: (N, S) left scatter weights.
    """
    B, S, C = inputs.size()
    assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
    assert not torch.isnan(alpha).any(), "Nan in alpha tensor."
    assert unbound_alpha or (alpha.le(1.0 + eps).all() and alpha.ge(0.0 - eps).all()), (
        "Incorrect values in alpha tensor"
        ", 0.0 <= tensor <= 1.0"
    )

    dtype = alpha.dtype
    alpha = alpha.float()
    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        assert not padding_mask[:, 0].any(), "Expected right-padded inputs."
        alpha = alpha.masked_fill(padding_mask, 0)

    if target_lengths is not None:
        assert target_lengths.size() == (B,)
        feat_lengths = target_lengths.long()
        desired_sum = beta * target_lengths.type_as(inputs) + eps
        alpha_sum = alpha.sum(1)
        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        T = feat_lengths.max()
    else:
        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

    # aggregate and integrate
    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # indices used for scattering
        right_idx = (csum / beta).floor().long().clip(max=T)
        left_idx = right_idx.roll(1, dims=1)
        left_idx[:, 0] = 0

        # count # of fires from each source
        fire_num = right_idx - left_idx
        extra_weights = (fire_num - 1).clip(min=0)

    # The extra entry in last dim is for tail
    output = inputs.new_zeros((B, T + 1, C))
    delay = inputs.new_zeros((B, T + 1))
    source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(inputs)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = fire_num > 0
    right_weight = torch.where(
        fire_mask,
        csum - right_idx.type_as(alpha) * beta,
        zero
    ).type_as(inputs)
    output.scatter_add_(
        1,
        right_idx.unsqueeze(-1).expand(-1, -1, C),
        right_weight.unsqueeze(-1) * inputs
    )
    delay.scatter_add_(
        1,
        right_idx,
        right_weight * source_range / beta
    )

    # left scatter
    left_weight = (
        alpha - right_weight - extra_weights.type_as(alpha) * beta
    ).type_as(inputs)
    output.scatter_add_(
        1,
        left_idx.unsqueeze(-1).expand(-1, -1, C),
        left_weight.unsqueeze(-1) * inputs
    )
    delay.scatter_add_(
        1,
        left_idx,
        left_weight * source_range / beta
    )

    # extra scatters
    if extra_weights.ge(0).any():
        extra_steps = extra_weights.max().item()
        tgt_idx = left_idx
        src_feats = inputs * beta
        for _ in range(extra_steps):
            tgt_idx = (tgt_idx + 1).clip(max=T)
            # (B, S, 1)
            src_mask = (extra_weights > 0)
            output.scatter_add_(
                1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                src_feats * src_mask.unsqueeze(2)
            )
            delay.scatter_add_(
                1,
                tgt_idx,
                source_range * src_mask
            )
            extra_weights -= 1

    # tail handling
    if target_lengths is not None:
        # training time -> ignore tail
        output = output[:, :T, :]
        delay = delay[:, :T]
    else:
        # find out contribution to output tail
        # note: w/o scaling, extra weight is all 0
        zero = right_weight.new_zeros((1,))
        r_mask = right_idx == feat_lengths.unsqueeze(1)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        l_mask = left_idx == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        # a size (B,) mask that extends position that passed threshold.
        extend_mask = tail_weights >= tail_thres

        # extend 1 fire and upscale the weights
        if extend_mask.any():
            # (B, T, C), may have infs so need the mask
            upscale = (
                torch.ones_like(output)
                .scatter(
                    1,
                    feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                    beta / (
                        tail_weights
                        .masked_fill(~extend_mask, beta)
                        .view(B, 1, 1)
                        .expand(-1, -1, C)),
                )
                .detach()
            )
            output *= upscale
            feat_lengths += extend_mask.long()
            T = feat_lengths.max()
        output = output[:, :T, :]
        delay = delay[:, :T]

        # a size (B, T) mask to erase weights
        tail_mask = torch.arange(T, device=output.device).unsqueeze(0) >= feat_lengths.unsqueeze(1)
        output[tail_mask] = 0

    return {
        "cif_out": [output],
        "cif_lengths": [feat_lengths],
        "alpha_sum": [alpha_sum.to(dtype)],
        "delays": [delay],
        "tail_weights": [tail_weights] if target_lengths is None else [],
        "scaled_alpha": [alpha],
        "cumsum_alpha": [csum],
        "right_indices": [right_idx],
        "right_weights": [right_weight],
        "left_indices": [left_idx],
        "left_weights": [left_weight],
    }
    
