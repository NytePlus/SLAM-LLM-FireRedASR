import torch
import torch.nn as nn
import torch.nn.functional as F
from model.adapter import EncoderProjectorConcat, EncoderProjectorCov1d, CIFAdapter, FunasrCIFAdapter, KernelLinear, FlashCIFAdapter
from model.conformer_encoder import ConformerEncoder

import os
import sys
import logging
import types
from typing import List, Optional, Tuple, Union

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from wavlm.WavLM import WavLM, WavLMConfig
from utils.metric import compute_accuracy
from utils.config_utils import generate_peft_config
from utils.model_utils import print_model_size, print_module_size
from utils.npu_flash_attn import patch_npu_flash_attn
logger = logging.getLogger(__name__)

def extract_variable_length_features(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        # x = (x + self.positional_embedding).to(x.dtype)
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    DEFAULT_SPEECH_TOKEN = "<speech>"
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IGNORE_TOKEN = -100
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN, DEFAULT_IMAGE_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.default_ignore_token = DEFAULT_IGNORE_TOKEN
    tokenizer.default_speech_token = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    tokenizer.default_image_token = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_name = model_config.encoder_name
    if encoder_name == "conformer":
        encoder = ConformerEncoder(**model_config["encoder_config"])
    elif encoder_name == "wavlm":
        checkpoint = torch.load(model_config['encoder_path'])
        cfg = WavLMConfig(checkpoint['cfg'])
        encoder = WavLM(cfg)
        encoder.load_state_dict(checkpoint['model'])
    elif encoder_name == "whisper":
        import whisper
        encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)

    else:
        raise NotImplementedError(
            f"Unsupported encoder_name: '{encoder_name}'. "
        )
    print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()
        print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder
    


def setup_encoder_projector(train_config, model_config, **kwargs):
    encoder_name = model_config.encoder_name
    if encoder_name == "conformer":
        encoder_dim = model_config["encoder_config"]["d_model"]
    elif encoder_name == "wavlm":
        encoder_dim = model_config['encoder_dim']
    elif encoder_name == 'whisper':
        encoder_dim = model_config['encoder_dim']
    else:
        raise NotImplementedError(
            f"Unsupported encoder_name: '{encoder_name}'. "
        )

    projector_name = model_config.encoder_projector
    if projector_name == "linear":
        encoder_projector = EncoderProjectorConcat(encoder_dim,model_config["llm_dim"],model_config["encoder_projector_ds_rate"])
    elif projector_name == "cov1d-linear":
        encoder_projector = EncoderProjectorCov1d(encoder_dim,model_config["llm_dim"],model_config["encoder_projector_ds_rate"])
    elif projector_name == "kernel-linear":
        encoder_projector = KernelLinear(encoder_dim,model_config["llm_dim"],model_config["encoder_projector_ds_rate"])
    elif projector_name == "CIF":
        encoder_projector = CIFAdapter(encoder_dim, model_config["llm_dim"])
    elif projector_name == "FunasrCIF":
        encoder_projector = FunasrCIFAdapter(encoder_dim, model_config["llm_dim"])
    elif projector_name == "FlashCIF":
        encoder_projector = FlashCIFAdapter(encoder_dim, model_config["llm_dim"])

    print_module_size(encoder_projector, f"{projector_name} adapter", int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    if train_config.freeze_projector:
        for name, param in encoder_projector.named_parameters():
            param.requires_grad = False
        encoder_projector.eval()
        print_module_size(encoder_projector, "adapter", int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder_projector


def setup_llm(train_config, model_config, **kwargs):
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None

    config = AutoConfig.from_pretrained(model_config.llm_path)
    config.use_cache=use_cache
    config._attn_implementation=model_config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_config.llm_path,
        config=config,
    )

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()
        
    if train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model

def setup_vl(train_config, model_config, **kwargs):
    if model_config.vl_path is None:
        return None, None
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config.vl_path,
    ).visual
    model.eval()
    image_encoder_projector = EncoderProjectorConcat(model_config['vl_dim'], model_config["llm_dim"], downsample_rate=1)

    print_module_size(image_encoder_projector, "image adapter", int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    print_module_size(model, model_config.vl_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model, image_encoder_projector


def model_factory(train_config, model_config, **kwargs):
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    
    # TODO: image encoder projector
    image_encoder, image_encoder_projector = setup_vl(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # encoder
    encoder = setup_encoder(train_config, model_config, **kwargs)

    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )

    model = slam_model_asr(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        image_encoder=image_encoder,
        image_encoder_projector=image_encoder_projector,
        cif_loss_weight=model_config.cif_loss_weight,
        ctc_loss_weight=model_config.ctc_loss_weight,
        **kwargs,
    )
    firered_path = model_config.get("firered_path", None)
    if firered_path is not None and firered_path != '':
        logger.info("loading pretrain parts from: {}".format(firered_path))
        firered_dict = torch.load(firered_path, map_location="cpu")
        model.load_state_dict(firered_dict["model_state_dict"], strict=False)

    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None and ckpt_path != '':
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )

    # # model = model.module
    # model_state_dict = model.state_dict()
    # state_dict = OrderedDict()

    # for name, para in model.named_parameters():
    #     if para.requires_grad:
    #         state_dict[name] = model_state_dict[name]
    #     else:
    #         if name.startswith("encoder"):
    #             state_dict[name] = model_state_dict[name]
    # save_dir = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/FireRedASR-LLM-v8/"
    # save_full_path = os.path.join(save_dir, "model.pth.tar")
    # # 构造 args 结构
    # import argparse
    # args = argparse.Namespace(
    #     encoder_path=save_dir,
    #     llm_dir=save_dir,
    #     freeze_encoder=False,
    #     freeze_llm=0,
    #     use_flash_attn=0,
    #     use_fp16=0,
    #     use_lora=1,
    #     encoder_downsample_rate=2,
    # )
    # # 构造 checkpoint
    # checkpoint = {
    #     "model_state_dict": state_dict,  # 模型权重
    #     "args": args  # 额外的训练参数
    # }

    # # 保存到 .pth.tar
    # torch.save(checkpoint, "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/FireRedASR-LLM-v8/model.pth.tar")

    # exit(1)
    return model, tokenizer


class slam_model_asr(torch.nn.Module):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        image_encoder = None,
        image_encoder_projector = None,
        cif_loss_weight = None,
        ctc_loss_weight = None,
        **kwargs,
    ):
        super().__init__()
        # modality encoder 
        self.encoder = encoder

        # image encoder
        self.image_encoder = image_encoder

        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # image projector
        self.image_encoder_projector = image_encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if train_config.get("enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)
        
        self.cif_loss_weight = cif_loss_weight
        self.ctc_loss_weight = ctc_loss_weight

    # --- TODO: 融入图像模态 ---
    def encode_image(self, image_embed, pixel_values_length, grid_thw, inputs_embeds, attention_mask, labels, input_ids):
        # image_embed(batch_size, max_seq_len, vl_dim)
        chunks = [image_embed[i, :pixel_values_length[i]] for i in range(image_embed.size(0))]
        image_embed_flat = torch.cat(chunks, dim=0)

        image_encoder_outs = self.image_encoder(image_embed_flat, grid_thw=grid_thw) # (batch * seq_len, vl_dim)
        image_encoder_feature_length = pixel_values_length // 4

        max_len = image_encoder_feature_length.max().item()
        outputs = []
        start = 0
        for length in image_encoder_feature_length:
            length = int(length)
            end = start + length
            chunk = image_encoder_outs[start:end]   # (seq_len_i, D)
            if length < max_len:
                pad_len = max_len - length
                chunk = F.pad(chunk, (0, 0, 0, pad_len))
            outputs.append(chunk)
            start = end

        image_encoder_outs = torch.stack(outputs, dim=0) 
        image_projector_outs = self.image_encoder_projector(image_encoder_outs) # (batch, seq_len, llm_dim)

        # print(f'token length(audio+text|image): {inputs_embeds.shape[1]}|{image_projector_outs.shape[1]}|')
        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_nontext_features(
                image_projector_outs, image_encoder_feature_length, inputs_embeds, input_ids, attention_mask, labels, self.tokenizer.default_image_token
            )

        return inputs_embeds, attention_mask, labels, position_ids
    # --- end ---

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

    def forward(self,
                input_ids: torch.LongTensor = None,
                input_features: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_length: Optional[torch.Tensor] = None,
                grid_thw: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                input_feature_length : Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                transcript_ids: Optional[torch.Tensor] = None,
                transcript_length: Optional[torch.Tensor] = None,
                experiment_name: str= ""
                ):
        
        # print(input_features.shape, input_ids.shape, pixel_values.shape) # torch.Size([2, 217101]) torch.Size([2, 153]) torch.Size([2, 24, 1176])

        if type(self.encoder).__name__ == 'WavLM':
            encoder_outs, _ = self.encoder.extract_features(input_features)
            encoder_feature_length = self.encoder.compute_feature_length(input_feature_length)
            # print(input_feature_length, encoder_feature_length) # tensor([199130, 217101], device='npu:0') tensor([678, 622], device='npu:0')
        elif type(self.encoder).__name__ == 'ConformerEncoder':
            encoder_outs,encoder_feature_length,_ = self.encoder(input_features,input_feature_length) # bs*seq*dim
        elif type(self.encoder).__name__ == 'AudioEncoder':
            encoder_outs = self.encoder(input_features) # bs*seq*dim
            encoder_feature_length = input_feature_length // 2
        
        if type(self.encoder_projector).__name__ in ["EncoderProjectorConcat", "EncoderProjectorCov1d", "KernelLinear"]:
            projector_outs = self.encoder_projector(encoder_outs)
            projector_feature_length = encoder_feature_length // self.encoder_projector.ds
        elif type(self.encoder_projector).__name__ in ['CIFAdapter', 'FunasrCIFAdapter', 'FlashCIFAdapter']:
            projector_outs, projector_feature_length, quantity_loss = self.encoder_projector(encoder_outs, encoder_feature_length, transcript_length)
        
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        # print(projector_outs.shape, inputs_embeds.shape) # torch.Size([2, 339, 4096]) torch.Size([2, 153, 4096])

        inputs_embeds, attention_mask, labels, position_ids, input_ids, audio_mask = self._merge_input_ids_with_audio_features(
                projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
            )
        # self.debug_verify_labels(input_ids, attention_mask, self.tokenizer, is_attn_mask=True, )
        # self.debug_verify_labels(input_ids, labels, self.tokenizer, is_attn_mask=False, )
        # self.debug_verify_labels(input_ids, audio_mask, self.tokenizer, is_attn_mask=True, )
        
        if self.image_encoder is not None:
            inputs_embeds, attention_mask, labels, position_ids = self.encode_image(
                pixel_values, pixel_values_length, grid_thw, 
                inputs_embeds, attention_mask, labels, input_ids)
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, position_ids=position_ids)
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=self.tokenizer.default_ignore_token)
                # vocab_size = self.tokenizer.vocab_size
                # pred_ids, label_ids = preds[0, :-1].clamp(0, vocab_size-1), labels[0, 1:].clamp(0, vocab_size-1)
                # print(f'predict: {self.tokenizer.decode(pred_ids)} \ngt: {self.tokenizer.decode(label_ids)}\n')
                # self.debug_verify_labels(pred_ids.unsqueeze(0), audio_mask, self.tokenizer, is_attn_mask=True, )

        if experiment_name == "wavprompt":
            transcript_embed = self.llm.get_input_embeddings()(transcript_ids)
            mask = torch.arange(transcript_ids.size(1), device=transcript_ids.device).unsqueeze(0) < transcript_length.unsqueeze(1)
            mask = mask.unsqueeze(-1)  # [B, T, 1]

            assert projector_outs.shape[1] == transcript_embed.shape[1], f'Adapter not align: Projector out {projector_outs.shape[1]}, Transcript {transcript_embed.shape[1]}'
            mse = (projector_outs - transcript_embed) ** 2
            mse = mse * mask

            embed_loss = mse.sum() / (mask.sum() + 1e-8)
            model_outputs.embed_loss = embed_loss
            model_outputs.quantity_loss = quantity_loss
            model_outputs.loss = model_outputs.loss + self.cif_loss_weight * embed_loss + 0.05 * quantity_loss
        elif experiment_name == "ctc":
            log_probs = F.log_softmax(model_outputs.logits, dim=-1)
            log_probs = log_probs.transpose(0, 1) # (B, T, V) -> (T, B, V)
            
            model_outputs.ctc_loss = self.ctc_loss_weight * masked_ctc_loss(
                log_probs,
                transcript_ids,
                audio_mask,
                transcript_length,
                blank=0,
                reduction="mean",
                zero_infinity=True
            )
            model_outputs.loss = model_outputs.loss + model_outputs.ctc_loss

        return model_outputs, acc
    
    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                input_features: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_length: Optional[torch.Tensor] = None,
                grid_thw: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                input_feature_length : Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs
                ):
        

        if type(self.encoder).__name__ == 'WavLM':
            encoder_outs, _ = self.encoder.extract_features(input_features)
            encoder_feature_length = self.encoder.compute_feature_length(input_feature_length)

        elif type(self.encoder).__name__ == 'ConformerEncoder':
            encoder_outs,encoder_feature_length,_ = self.encoder(input_features,input_feature_length) # bs*seq*dim

        elif type(self.encoder).__name__ == 'AudioEncoder':
            encoder_outs = self.encoder(input_features) # bs*seq*dim
            encoder_feature_length = input_feature_length // 2

        if type(self.encoder_projector).__name__ in ["EncoderProjectorConcat", "EncoderProjectorCov1d", "KernelLinear"]:
            projector_outs = self.encoder_projector(encoder_outs)
            projector_feature_length = encoder_feature_length // self.encoder_projector.ds
        elif type(self.encoder_projector).__name__ in ['CIFAdapter', 'FunasrCIFAdapter', 'FlashCIFAdapter']:
            projector_outs, projector_feature_length, quantity_loss = self.encoder_projector(encoder_outs, encoder_feature_length)

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, input_ids, _ = self._merge_input_ids_with_audio_features(
                projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
            )
        # self.debug_verify_labels(input_ids, attention_mask, self.tokenizer, is_attn_mask=True, )
        if self.image_encoder is not None:
            inputs_embeds, attention_mask, labels, position_ids = self.encode_image(
                pixel_values, pixel_values_length, grid_thw, 
                inputs_embeds, attention_mask, labels, input_ids)

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 3),
            do_sample=kwargs.get("do_sample", True),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 3.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            # position_ids=position_ids,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs
    
    
    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        """
        Merge input_ids with with audio features into final embeddings
        
        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)

        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim).contiguous()
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = True
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.tokenizer.default_speech_token
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.tokenizer.default_speech_token) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
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
            (batch_size, max_token_num), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full((batch_size, max_token_num),self.tokenizer.default_ignore_token,dtype=input_ids.dtype, device=inputs_embeds.device).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]
        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids, audio_to_overwrite

    def _merge_input_ids_with_nontext_features(
            self, 
            nontext_features, # (b, t1, llm_dim)
            num_nontext_tokens, # (b)
            inputs_embeds, # (b, t2, llm_dim)
            input_ids, # (b, t2)
            attention_mask, # (b, t2)
            labels, # (b, t2)
            placeholder_token_id
        ):
        """
        Generic function to merge input_ids with non-text features into final embeddings
        
        Args:
            nontext_features (`torch.Tensor` of shape `(num_features, max_feature_tokens, embed_dim)`):
                All non-text feature vectors in the batch
            num_nontext_tokens (`torch.LongTensor` of shape `(num_features)`):
                The length of feature embeddings for each feature
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with non-text embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with placeholder tokens
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional`)
                Labels need to be recalculated to support training (if provided)
            placeholder_token_id (`int`):
                The token id used as placeholder for non-text features
            ignore_token_id (`int`, *optional*):
                Token id to use for ignoring positions in labels (defaults to tokenizer's default_ignore_token)
        
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids
        """
        num_features, max_feature_tokens, embed_dim = nontext_features.shape
        
        # Create mask for valid non-text features

        nontext_features_mask = torch.arange(max_feature_tokens).expand(num_features, max_feature_tokens).to(
            num_nontext_tokens.device
        ) < num_nontext_tokens.unsqueeze(1)
        masked_nontext_features = nontext_features[nontext_features_mask].view(-1, embed_dim).contiguous()
        
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
            (batch_size, max_token_num), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings for text tokens
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_placeholder_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_placeholder_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_placeholder_indices]
        
        # Handle labels if provided
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            ignore_id = getattr(self.tokenizer, 'default_ignore_token', -100)
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
    
def masked_ctc_loss(
    log_probs,                 # (T, B, C)
    transcript_ids,            # 1D 拼接后的 targets
    mask,                      # (B, T)  bool
    target_lengths,            # (B,)
    blank=0,
    reduction="mean",
    zero_infinity=True
):
    T, B, C = log_probs.shape
    device = log_probs.device

    masked_log_probs_list = []
    input_lengths = []

    for b in range(B):
        cur_mask = mask[b]  # (T,)
        cur_lp = log_probs[:, b, :]  # (T, C)
        cur_lp_masked = cur_lp[cur_mask]

        masked_log_probs_list.append(cur_lp_masked)
        input_lengths.append(cur_lp_masked.size(0))

    max_len = max(input_lengths)
    padded = log_probs.new_full(
        (max_len, B, C),
        fill_value=float("-inf")
    )

    for b in range(B):
        cur_len = input_lengths[b]
        padded[:cur_len, b, :] = masked_log_probs_list[b]

    input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
    loss = F.ctc_loss(
        padded,
        transcript_ids,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity
    )

    return loss
