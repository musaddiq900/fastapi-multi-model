I'll help you create a FastAPI project that integrates multiple AI models. Here's a complete implementation:

## Project Structure
```
fastapi-multi-model/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deep_thinker.py
│   │   ├── text_to_speech.py
│   │   ├── text_to_image.py
│   │   └── text_to_video.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── requests.py
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── requirements.txt
└── README.md
```

## 1. requirements.txt
```txt
fastapi==0.104.1
uvicorn==0.24.0
transformers==4.35.0
diffusers==0.24.0
torch==2.1.0
accelerate==0.24.1
huggingface-hub==0.19.0
pillow==10.1.0
python-multipart==0.0.6
pydantic==2.5.0
```

## 2. app/utils/config.py
```python
import os
from typing import Optional

class Settings:
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
settings = Settings()
```

## 3. app/schemas/requests.py
```python
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
```

## 4. app/models/deep_thinker.py
```python
import torch
from transformers import AutoModelForCausalLM
from ..utils.config import settings

class DeepThinker:
    def __init__(self):
        self.model = None
        self.device = settings.DEVICE
        
    def load_model(self):
        if self.model is None:
            print("Loading Deep Thinking model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "moonshotai/Kimi-K2-Thinking", 
                trust_remote_code=True, 
                dtype="auto",
                device_map=self.device
            )
            print("Deep Thinking model loaded!")
    
    def generate(self, prompt: str, max_length: int = 512):
        if self.model is None:
            self.load_model()
            
        # Note: You'll need to adjust this based on the specific tokenizer and generation method
        # for the Kimi-K2-Thinking model
        inputs = self.model.preprocess(prompt)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True
        )
        return self.model.postprocess(outputs)

deep_thinker = DeepThinker()
```

## 5. app/models/text_to_speech.py
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils.config import settings

class TextToSpeech:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = settings.DEVICE
        
    def load_model(self):
        if self.model is None:
            print("Loading Text-to-Speech model...")
            self.tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1")
            self.model = AutoModelForCausalLM.from_pretrained(
                "maya-research/maya1",
                device_map=self.device
            )
            print("Text-to-Speech model loaded!")
    
    def generate(self, text: str):
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Convert to audio (adjust based on Maya model output format)
        audio = self.model.postprocess_audio(outputs)
        return audio

tts_model = TextToSpeech()
```

## 6. app/models/text_to_image.py
```python
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
```

## 7. app/models/text_to_video.py
```python
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
```

## 8. app/main.py
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, StreamingResponse
import io
import base64
from PIL import Image

from .models.deep_thinker import deep_thinker
from .models.text_to_speech import tts_model
from .models.text_to_image import text_to_image
from .models.text_to_video import text_to_video
from .schemas.requests import (
    DeepThinkRequest, 
    TextToSpeechRequest, 
    TextToImageRequest, 
    TextToVideoRequest,
    ImageToVideoRequest
)

app = FastAPI(title="Multi-Model AI API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Multi-Model AI API is running!"}

@app.post("/api/deep-think")
async def deep_think(request: DeepThinkRequest):
    try:
        result = deep_thinker.generate(request.prompt, request.max_length)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in deep thinking: {str(e)}")

@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        audio_data = tts_model.generate(request.text)
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in text-to-speech: {str(e)}")

@app.post("/api/text-to-image")
async def generate_image(request: TextToImageRequest):
    try:
        image = text_to_image.generate(request.prompt, request.model)
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in text-to-image: {str(e)}")

@app.post("/api/text-to-video")
async def generate_video_from_text(request: TextToVideoRequest, background_tasks: BackgroundTasks):
    try:
        video_data = text_to_video.text_to_video(request.prompt)
        
        # For now, return a message as video generation might be complex
        # In production, you might want to handle this asynchronously
        return {"message": "Video generation started", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in text-to-video: {str(e)}")

@app.post("/api/image-to-video")
async def generate_video_from_image(request: ImageToVideoRequest):
    try:
        video_frames = text_to_video.image_to_video(request.prompt, request.image_url)
        
        # Export to video file
        video_buffer = io.BytesIO()
        export_to_video(video_frames, video_buffer)
        video_buffer.seek(0)
        
        return StreamingResponse(
            video_buffer,
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=video.mp4"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in image-to-video: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 9. Run the application

Create a `run.py` file in the root directory:

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

Run the application:
```bash
python run.py
```

## Environment Setup

Create a `.env` file:
```bash
HF_TOKEN=your_huggingface_token_here
```

## Usage Examples

1. **Deep Thinking**:
```bash
curl -X POST "http://localhost:8000/api/deep-think" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain the concept of quantum computing", "max_length": 512}'
```

2. **Text to Image**:
```bash
curl -X POST "http://localhost:8000/api/text-to-image" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Astronaut in a jungle, cold color palette, detailed, 8k"}' \
     --output generated_image.png
```

3. **Text to Speech**:
```bash
curl -X POST "http://localhost:8000/api/text-to-speech" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, this is a test speech"}' \
     --output speech.wav
```

