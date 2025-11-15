# FastAPI Multi-Model AI Project

A comprehensive FastAPI application that integrates multiple AI models for various tasks including deep thinking, text-to-speech, text-to-image, and text/image-to-video generation.

## 🚀 Features

- **Deep Thinking LLM**: Advanced reasoning with Kimi-K2-Thinking model
- **Text-to-Speech**: Natural voice generation with Maya model
- **Text-to-Image**: Image generation with Stable Diffusion XL and FLUX models
- **Text/Image-to-Video**: Video generation with multiple state-of-the-art models

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or Apple M1/M2 with MPS support
- Hugging Face token (for some models)
- At least 16GB RAM (32GB+ recommended)

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd fastapi-multi-model
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Hugging Face token
   ```

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_token_here
```

## 🎯 Usage

### Starting the Server

```bash
python run.py
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, access the interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📚 API Endpoints

### 1. Deep Thinking
- **POST** `/api/deep-think`
- Generate thoughtful responses using Kimi-K2-Thinking model

```json
{
  "prompt": "Explain quantum computing",
  "max_length": 512
}
```

### 2. Text-to-Speech
- **POST** `/api/text-to-speech`
- Convert text to speech using Maya model

```json
{
  "text": "Hello, this is a test speech"
}
```

### 3. Text-to-Image
- **POST** `/api/text-to-image`
- Generate images from text prompts

```json
{
  "prompt": "Astronaut in a jungle, detailed, 8k",
  "model": "stable-diffusion"  # or "flux"
}
```

### 4. Text-to-Video
- **POST** `/api/text-to-video`
- Generate videos from text prompts

```json
{
  "prompt": "A young man walking on the street"
}
```

### 5. Image-to-Video
- **POST** `/api/image-to-video`
- Generate videos from images and prompts

```json
{
  "prompt": "A man playing electric guitar",
  "image_url": "https://example.com/image.png"
}
```

## 🏗 Project Structure

```
fastapi-multi-model/
├── app/
│   ├── models/           # AI model implementations
│   ├── schemas/          # Pydantic models
│   ├── utils/            # Configuration and utilities
│   └── main.py          # FastAPI application
├── requirements.txt
├── run.py
└── README.md
```

## 🔧 Models Included

### Deep Thinking
- **Model**: `moonshotai/Kimi-K2-Thinking`
- **Purpose**: Advanced reasoning and thoughtful responses

### Text-to-Speech
- **Model**: `maya-research/maya1`
- **Purpose**: Natural speech synthesis

### Text-to-Image
- **Models**: 
  - `stabilityai/stable-diffusion-xl-base-1.0`
  - `black-forest-labs/FLUX.1-dev`
- **Purpose**: High-quality image generation

### Text/Image-to-Video
- **Models**:
  - `Wan-AI/Wan2.1-T2V-14B`
  - `Lightricks/LTX-Video`
  - `tencent/HunyuanVideo` (via API)
- **Purpose**: Video generation from text and images

## 🚨 Important Notes

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM recommended
- **CPU**: Multi-core processor
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Storage**: SSD with at least 50GB free space for models

### Memory Management
- Models are loaded on-demand to conserve memory
- Consider implementing model unloading for production
- Video generation may require significant resources

### Rate Limiting
- Implement rate limiting for production use
- Consider queue systems for resource-intensive operations

## 🔒 Security Considerations

- Use environment variables for sensitive tokens
- Implement authentication for production deployment
- Consider rate limiting and request throttling
- Validate all input data

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch sizes
   - Use smaller models
   - Enable model offloading

2. **Model Loading Failures**
   - Check internet connection
   - Verify Hugging Face token
   - Clear Hugging Face cache

3. **CUDA Errors**
   - Update GPU drivers
   - Check CUDA compatibility
   - Use CPU fallback if needed

### Getting Help

Check the API documentation at `/docs` for detailed endpoint information and testing.

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]
```

## 3. app/README.md
```markdown
# App Module

This directory contains the main FastAPI application and all its components.

## Structure

### models/
Contains the implementation for all AI models:

- `deep_thinker.py` - Kimi-K2-Thinking model for advanced reasoning
- `text_to_speech.py` - Maya model for text-to-speech conversion
- `text_to_image.py` - Stable Diffusion and FLUX models for image generation
- `text_to_video.py` - Multiple models for video generation

### schemas/
Pydantic models for request/response validation:

- `requests.py` - Request schemas for all API endpoints

### utils/
Utility functions and configuration:

- `config.py` - Application settings and configuration

### main.py
FastAPI application entry point with all route definitions.

## Model Loading Strategy

Models are loaded using lazy initialization:

1. Models are not loaded at application startup
2. Each model is loaded on first use
3. Loaded models are cached for subsequent requests
4. This conserves memory when not all models are used

## Adding New Models

To add a new model:

1. Create a new file in `models/`
2. Implement a class with `load_model()` and `generate()` methods
3. Add request schema in `schemas/requests.py`
4. Add API endpoint in `main.py`

Example:
```python
class NewModel:
    def __init__(self):
        self.model = None
    
    def load_model(self):
        if self.model is None:
            self.model = load_your_model()
    
    def generate(self, input_data):
        if self.model is None:
            self.load_model()
        return self.model.process(input_data)
```
```

## 4. models/README.md
```markdown
# AI Models

This directory contains implementations for all AI models used in the application.

## Available Models

### Deep Thinking (`deep_thinker.py`)
- **Model**: `moonshotai/Kimi-K2-Thinking`
- **Purpose**: Advanced reasoning and thoughtful text generation
- **Features**: Causal language modeling with deep thinking capabilities
- **Input**: Text prompt
- **Output**: Generated text response

### Text-to-Speech (`text_to_speech.py`)
- **Model**: `maya-research/maya1`
- **Purpose**: Convert text to natural speech
- **Features**: Causal LM for speech synthesis
- **Input**: Text string
- **Output**: Audio data (WAV format)

### Text-to-Image (`text_to_image.py`)
- **Models**: 
  - `stabilityai/stable-diffusion-xl-base-1.0`
  - `black-forest-labs/FLUX.1-dev`
- **Purpose**: Generate images from text descriptions
- **Features**: Diffusion-based image generation
- **Input**: Text prompt
- **Output**: PIL Image object

### Text/Image-to-Video (`text_to_video.py`)
- **Models**:
  - `Wan-AI/Wan2.1-T2V-14B` (text-to-video)
  - `Lightricks/LTX-Video` (image-to-video)
  - `tencent/HunyuanVideo` (via API)
- **Purpose**: Generate videos from text or images
- **Features**: Multiple video generation approaches
- **Input**: Text prompt and/or image URL
- **Output**: Video frames or file

## Model Management

### Lazy Loading
Models are loaded on first use to conserve memory:
```python
def generate(self, input_data):
    if self.model is None:
        self.load_model()
    return self.model.process(input_data)
```

### Device Management
- Automatically uses CUDA, MPS, or CPU based on availability
- Supports mixed precision for GPU acceleration

### Error Handling
- Comprehensive error handling for model loading and inference
- Graceful fallbacks where possible

## Memory Considerations

- Each model can require significant memory
- Consider implementing model unloading for production
- Monitor GPU memory usage during operation
- Use smaller models or quantized versions if memory is limited

## Adding New Models

1. Create a new Python file in this directory
2. Implement a class with standard interface
3. Add proper error handling
4. Update the main application to include the new model
```

## 5. schemas/README.md
```markdown
# API Schemas

This directory contains Pydantic models for request and response validation.

## Request Schemas

All API endpoints use Pydantic models for input validation:

### `DeepThinkRequest`
- **Endpoint**: `/api/deep-think`
- **Fields**:
  - `prompt` (str): Required input text
  - `max_length` (int, optional): Maximum response length, default 512

### `TextToSpeechRequest`
- **Endpoint**: `/api/text-to-speech`
- **Fields**:
  - `text` (str): Required text to convert to speech

### `TextToImageRequest`
- **Endpoint**: `/api/text-to-image`
- **Fields**:
  - `prompt` (str): Required image description
  - `model` (str, optional): Model to use ("stable-diffusion" or "flux")

### `TextToVideoRequest`
- **Endpoint**: `/api/text-to-video`
- **Fields**:
  - `prompt` (str): Required video description

### `ImageToVideoRequest`
- **Endpoint**: `/api/image-to-video`
- **Fields**:
  - `prompt` (str): Required video description
  - `image_url` (str): Required source image URL

## Response Types

Responses vary by endpoint:
- JSON for text responses
- Image data for image generation
- Audio data for speech synthesis
- Video files for video generation

## Validation

All schemas include:
- Type validation
- Required field checking
- Optional field defaults
- Input sanitization

## Extending Schemas

To add new fields or endpoints:

1. Update the existing schema or create a new one
2. Add proper type hints and validation
3. Update the API endpoint to use the new schema
4. Update documentation accordingly
```

## 6. run.py
```python
#!/usr/bin/env python3
"""
FastAPI Multi-Model AI Application Runner
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"Starting FastAPI server on {host}:{port}")
    print(f"Reload mode: {reload}")
    print(f"API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
```

## 7. .env.example
```env
# Hugging Face Token (required for some models)
HF_TOKEN=your_huggingface_token_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=false

# Model Configuration
DEVICE=auto  # auto, cuda, mps, cpu
```

