import os
import safetensors
import torch
from PIL import Image
from transformers import pipeline, AutoConfig, AutoProcessor, CLIPProcessor, CLIPModel

class CityAesthetics:
    def __init__(self, pathname, clip_pathname, device='cpu'):
        super().__init__()
        self.device = device
        if self.device == 'cuda':
            self.device += ':0'
        self.pathname = pathname
        self.clip_pathname = clip_pathname
        self.initialize_model()

    def initialize_model(self):
        # Load the model and processor
        self.config = AutoConfig.from_pretrained("city96/CityAesthetics")
        self.processor = AutoProcessor.from_pretrained("city96/CityAesthetics")
        self.model = CLIPModel.from_pretrained("city96/CityAesthetics", torch_dtype=torch.float16).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.pipe = pipeline("image-classification", model=self.model, image_processor=self.processor, device=self.device)

    def score(self, prompt, image):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)

        # Process image and text with CLIP
        inputs = self.clip_processor(text=[prompt], images=pil_image, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        text_embeds = self.clip_model.get_text_features(**inputs)
        image_embeds = self.clip_model.get_image_features(**inputs)

        # Normalize embeddings
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Concatenate embeddings and get aesthetic score
        global_features = torch.cat([text_embeds, image_embeds], dim=-1)
        score = self.model(global_features)[0][0].item() * 10

        return score