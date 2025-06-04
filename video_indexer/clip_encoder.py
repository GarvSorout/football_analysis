import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union, Dict

class ClipEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text descriptions into CLIP embeddings."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        return text_features.cpu().numpy()
        
    def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Encode video frames into CLIP embeddings."""
        # Convert numpy arrays to PIL Images
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        return image_features.cpu().numpy()
        
    def encode_event(self, event_dict: Dict) -> np.ndarray:
        """Encode both the event description and frames into a combined embedding."""
        text_embedding = self.encode_text([event_dict["description"]])
        frame_embedding = self.encode_frames(event_dict["frames"])
        
        # Combine text and visual embeddings (simple average for now)
        combined_embedding = (text_embedding + frame_embedding) / 2
        return combined_embedding 