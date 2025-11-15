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