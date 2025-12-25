import soundfile as sf
import torch

from typing import List, Optional, Tuple, Union
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class omni_asr:
    def __init__(self):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")

        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
                ],
            },
        ]

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
        ):
        # set use audio in video
        USE_AUDIO_IN_VIDEO = True

        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio
        text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
