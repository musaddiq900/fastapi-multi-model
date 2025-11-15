import os
from typing import Optional

class Settings:
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
settings = Settings()