import torch
from diffusers import DiffusionPipeline
from PIL import Image
import io
from ..utils.config import settings

class TextToImage:
    def __init__(self):
        self.sd_model = None
        self.flux_model = None
        self.device = settings.DEVICE
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
    def load_sd_model(self):
        if self.sd_model is None:
            print("Loading Stable Diffusion model...")
            self.sd_model = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                dtype=self.dtype, 
                device_map=self.device
            )
            print("Stable Diffusion model loaded!")
    
    def load_flux_model(self):
        if self.flux_model is None:
            print("Loading FLUX model...")
            self.flux_model = DiffusionPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                dtype=self.dtype, 
                device_map=self.device
            )
            print("FLUX model loaded!")
    
    def generate(self, prompt: str, model_type: str = "stable-diffusion") -> Image.Image:
        if model_type == "stable-diffusion":
            if self.sd_model is None:
                self.load_sd_model()
            image = self.sd_model(prompt).images[0]
        elif model_type == "flux":
            if self.flux_model is None:
                self.load_flux_model()
            image = self.flux_model(prompt).images[0]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return image

text_to_image = TextToImage()