import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class ResBlock(nn.Module):
    """Linear block with residuals"""

    def __init__(self, ch):
        super().__init__()
        self.join = nn.ReLU()
        self.long = nn.Sequential(
            nn.Linear(ch, ch),
            nn.LeakyReLU(0.1),
            nn.Linear(ch, ch),
            nn.LeakyReLU(0.1),
            nn.Linear(ch, ch),
        )

    def forward(self, x):
        return self.join(self.long(x) + x)


class PredictorModel(nn.Module):
    """Main predictor class"""

    def __init__(self, features=768, outputs=1, hidden=1024):
        super().__init__()
        self.features = features
        self.outputs = outputs
        self.hidden = hidden
        self.up = nn.Sequential(
            nn.Linear(self.features, self.hidden),
            ResBlock(ch=self.hidden),
        )
        self.down = nn.Sequential(
            nn.Linear(self.hidden, 128),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, self.outputs),
        )
        self.out = nn.Softmax(dim=1) if self.outputs > 1 else nn.Tanh()

    def forward(self, x):
        y = self.up(x)
        z = self.down(y)
        if self.outputs > 1:
            return self.out(z)
        else:
            return (self.out(z) + 1.0) / 2.0


class CityAestheticsScorer:
    def __init__(self, pathname, device='cpu'):
        self.device = device
        self.pathname = pathname
        self.initialize_model()

    def initialize_model(self):
        # Initialize CLIP model and processor
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14-336",
            torch_dtype=torch.float32,
        ).to(self.device)
        self.clip_model.eval()

        # Load the custom PredictorModel with state_dict from safetensors
        statedict = load_file(self.pathname)  # Load safetensor file
        assert tuple(statedict["up.0.weight"].shape) == (1024, 768), "Unexpected model architecture."
        self.city_model = PredictorModel(outputs=1)  # Initialize PredictorModel
        self.city_model.load_state_dict(statedict)  # Load weights into the model
        self.city_model.to(self.device)  # Move model to the device
        self.city_model.eval()  # Set model to evaluation mode

    def score(self, prompt, image):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image).convert("RGB")
            else:
                raise ValueError(f"Image file {image} does not exist.")
        else:
            raise TypeError("Image must be a PIL.Image.Image instance or a valid file path.")

        # Extract CLIP embeddings
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.device, dtype=torch.float32)
        with torch.no_grad():
            clip_embeddings = self.clip_model(**inputs).image_embeds

        # Obtain CityAesthetics score using CLIP embeddings
        with torch.no_grad():
            score = self.city_model(clip_embeddings)

        # Scale score from [0, 1] to [0, 10]
        return score.detach().cpu().item() * 10.0
