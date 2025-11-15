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