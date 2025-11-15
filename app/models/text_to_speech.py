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