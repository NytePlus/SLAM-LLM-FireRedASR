from model.aispeech_asr import *

from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
from research.modeling_llama import LlamaForResearch

def setup_reasearch_llm(train_config, model_config, **kwargs):
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None

    config = AutoConfig.from_pretrained(model_config.llm_path)
    config.use_cache=use_cache
    config._attn_implementation='eager'

    model = LlamaForResearch.from_pretrained(
        model_config.llm_path,
        config=config,
    )
    return model

def model_factory(train_config, model_config, **kwargs):
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    
    # TODO: image encoder projector
    image_encoder, image_encoder_projector = setup_vl(train_config, model_config, **kwargs)

    llm = setup_reasearch_llm(train_config, model_config, **kwargs)

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
        **kwargs,
    )
    firered_path = model_config.get( "firered_path", None)
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

    return model, tokenizer