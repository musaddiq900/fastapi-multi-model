import torch
import os
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from huggingface_hub import InferenceClient
from ..utils.config import settings

class TextToVideo:
    def __init__(self):
        self.wan_model = None
        self.ltx_model = None
        self.client = None
        self.device = settings.DEVICE
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
    def load_wan_model(self):
        if self.wan_model is None:
            print("Loading Wan2.1 model...")
            self.wan_model = DiffusionPipeline.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B", 
                dtype=self.dtype, 
                device_map=self.device
            )
            print("Wan2.1 model loaded!")
    
    def load_ltx_model(self):
        if self.ltx_model is None:
            print("Loading LTX-Video model...")
            self.ltx_model = DiffusionPipeline.from_pretrained(
                "Lightricks/LTX-Video", 
                dtype=self.dtype, 
                device_map=self.device
            )
            self.ltx_model.to(self.device)
            print("LTX-Video model loaded!")
    
    def init_client(self):
        if self.client is None and settings.HF_TOKEN:
            self.client = InferenceClient(
                provider="fal-ai",
                api_key=settings.HF_TOKEN,
            )
    
    def text_to_video(self, prompt: str, use_api: bool = False):
        if use_api and settings.HF_TOKEN:
            self.init_client()
            if self.client:
                video = self.client.text_to_video(
                    prompt,
                    model="tencent/HunyuanVideo",
                )
                return video
        else:
            self.load_wan_model()
            frames = self.wan_model(prompt).frames[0]
            return frames
    
    def image_to_video(self, prompt: str, image_url: str):
        self.load_ltx_model()
        image = load_image(image_url)
        output = self.ltx_model(image=image, prompt=prompt).frames[0]
        return output

text_to_video = TextToVideo()