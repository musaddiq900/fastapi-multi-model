from pydantic import BaseModel
from typing import Optional

class DeepThinkRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512

class TextToSpeechRequest(BaseModel):
    text: str

class TextToImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "stable-diffusion"  # "stable-diffusion" or "flux"

class TextToVideoRequest(BaseModel):
    prompt: str

class ImageToVideoRequest(BaseModel):
    prompt: str
    image_url: str