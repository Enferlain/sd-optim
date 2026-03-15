# Scorer class for AnatomyFlaws models using DINOv37b embeddings
# and the custom HybridHeadModel head.

import os
import json
import traceback
import logging

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoModel

logger = logging.getLogger(__name__)

# Import your HybridHeadModel
try:
    from sd_optim.builtin.scorers.models.lumi_model import HybridHeadModel

    logger.debug("Successfully imported HybridHeadModel.")
except ImportError:
    raise ImportError("Could not import HybridHeadModel. Make sure lumi_model.py is accessible.")


class Dinov3AnatomyScorer:
    """
    Scorer class for AnatomyFlaws models using DINOv3 7B 8-bit BnB embeddings
    and the custom HybridHeadModel head. Outputs a score 0-10.
    """

    DINOV3_PATCH_SIZE = 16
    MAX_DINOV3_RESOLUTION = 4096  # Memory protection limit

    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.config_path = config_path

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model head file not found: {self.model_path}")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Use float16 only on CUDA, default to float32 otherwise
        self.compute_dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        logger.info("Using compute dtype: %s", self.compute_dtype)

        # Initialize models
        self.config = None
        self.base_vision_model_name = None
        self.processor = None
        self.model = None
        self.head_model: HybridHeadModel | None = None
        self.num_classes = 0

        self.initialize_models()

    def initialize_models(self):
        # Load config
        logger.info("Loading config from: %s", self.config_path)
        try:
            with open(self.config_path, encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            raise OSError(f"Failed to load config: {e}")

        # Get vision model name (expecting DINOv3)
        self.base_vision_model_name = self.config.get("base_vision_model")
        if not self.base_vision_model_name or "dinov3" not in self.base_vision_model_name.lower():
            raise ValueError(f"Config missing or specifies non-DINOv3 'base_vision_model': {self.base_vision_model_name}")

        logger.info("Using base vision model: %s", self.base_vision_model_name)

        # Initialize DINOv3 Model and Processor
        logger.info("Initializing DINOv3 vision model and processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.base_vision_model_name)

            # Check if this is an 8-bit model that requires CUDA
            is_8bit_model = "8bit" in self.base_vision_model_name.lower() or "bnb" in self.base_vision_model_name.lower()

            # For 8-bit models, we need to load directly on CUDA and they don't work on CPU
            if is_8bit_model:
                if self.device == "cpu":
                    raise RuntimeError(
                        f"8-bit model '{self.base_vision_model_name}' cannot be used on CPU. Please use CUDA device instead."
                    )
                if not torch.cuda.is_available():
                    raise RuntimeError(f"8-bit model '{self.base_vision_model_name}' requires CUDA, but CUDA is not available.")
                # Load 8-bit model directly on CUDA without calling .to() afterward
                self.model = AutoModel.from_pretrained(
                    self.base_vision_model_name,
                    torch_dtype=self.compute_dtype,
                    trust_remote_code=True,
                    device_map="auto",  # This will automatically place the model on available GPUs
                )
                # Set to eval mode (this should work even for 8-bit models)
                self.model.eval()
                logger.info("Loaded 8-bit DINOv3 model on CUDA: %s", self.model.__class__.__name__)
            else:
                # Load regular model
                self.model = (
                    AutoModel.from_pretrained(
                        self.base_vision_model_name,
                        torch_dtype=self.compute_dtype,
                        trust_remote_code=True,
                    )
                    .to(self.device)
                    .eval()
                )
            logger.info("Loaded DINOv3 model: %s", self.model.__class__.__name__)
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv3 vision model/processor '{self.base_vision_model_name}': {e}")

        # Initialize Custom HybridHeadModel Head
        logger.info("Initializing HybridHeadModel head from: %s", self.model_path)
        try:
            # Load state dict
            state_dict = load_file(self.model_path, device="cpu")

            # Get parameters from config
            features = self.config.get("features", 4096)  # DINOv3-7B feature dim
            self.num_classes = self.config.get("num_classes", 2)

            # Load hyperparameters from config
            hidden_dim = self.config.get("hidden_dim", 1536)
            use_attention = self.config.get("use_attention", True)
            num_attn_heads = self.config.get("num_attn_heads", 32)
            attn_dropout = self.config.get("attn_dropout", 0.218)
            num_res_blocks = self.config.get("num_res_blocks", 4)
            dropout_rate = self.config.get("dropout_rate", 0.218)
            rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
            output_mode = self.config.get("output_mode", "linear")

            # Instantiate the HybridHeadModel
            self.head_model = HybridHeadModel(
                features=features,
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                use_attention=use_attention,
                num_attn_heads=num_attn_heads,
                attn_dropout=attn_dropout,
                num_res_blocks=num_res_blocks,
                dropout_rate=dropout_rate,
                rms_norm_eps=rms_norm_eps,
                output_mode=output_mode,
            )

            # Load the state dict
            missing, unexpected = self.head_model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys when loading head state_dict: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys when loading head state_dict: %s", unexpected)

            self.head_model.to(self.device).eval()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize or load HybridHeadModel head: {e}")

        logger.info("Models initialized successfully.")

    def score(self, image: Image.Image | str, prompt=None) -> float:
        """Calculates the 'Good Anatomy' score (0-10) for an image."""

        try:
            # Image Loading
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            elif isinstance(image, str) and os.path.isfile(image):
                pil_image = Image.open(image).convert("RGB")
            else:
                raise TypeError("Input must be a PIL Image or a valid file path.")
        except Exception as e:
            logger.error("Error loading image: %s", e)
            return -1.0

        try:
            # DINOv3 Image Processing
            current_w, current_h = pil_image.size
            img_to_process = pil_image

            # Optional: Add max resolution limit for memory protection
            if max(current_w, current_h) > self.MAX_DINOV3_RESOLUTION:
                scale = self.MAX_DINOV3_RESOLUTION / max(current_w, current_h)
                current_w = int(current_w * scale)
                current_h = int(current_h * scale)
                img_to_process = pil_image.resize((current_w, current_h), Image.Resampling.LANCZOS)
                logger.info("Scaling down image to fit within %spx limit.", self.MAX_DINOV3_RESOLUTION)

            # Validate reasonable image sizes after potential downscaling
            if current_w < self.DINOV3_PATCH_SIZE or current_h < self.DINOV3_PATCH_SIZE:
                raise ValueError(
                    f"Image too small: {current_w}x{current_h}. Minimum size is {self.DINOV3_PATCH_SIZE}x{self.DINOV3_PATCH_SIZE} pixels."
                )

            if current_w > 4096 or current_h > 4096:
                logger.warning("Very large image size %sx%s may cause memory issues.", current_w, current_h)

            # Ensure image dimensions are multiples of 16 (DINOV3_PATCH_SIZE)
            new_w = ((current_w + self.DINOV3_PATCH_SIZE - 1) // self.DINOV3_PATCH_SIZE) * self.DINOV3_PATCH_SIZE
            new_h = ((current_h + self.DINOV3_PATCH_SIZE - 1) // self.DINOV3_PATCH_SIZE) * self.DINOV3_PATCH_SIZE

            if new_w != current_w or new_h != current_h:
                logger.info(
                    "Adjusting dims (%sx%s) -> (%sx%s) to be multiples of %s for DINOv3.",
                    current_w,
                    current_h,
                    new_w,
                    new_h,
                    self.DINOV3_PATCH_SIZE,
                )
                img_to_process = img_to_process.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Use processor for ToTensor/Normalize
            inputs = self.processor(images=[img_to_process], return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device=self.device, dtype=self.compute_dtype)

            # Call DINOv3 model
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)

            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                raise ValueError("DINOv3 model did not return last_hidden_state.")

            # Use mean pooling of patch tokens (exclude CLS and register tokens)
            # DINOv3-7B has 4 register tokens. We skip CLS (pos 0) and registers (pos 1-4).
            nreg = getattr(self.model.config, "num_register_tokens", 0)
            patch_embeddings = last_hidden_state[:, 1 + nreg :]  # Skip CLS and Registers
            emb = torch.mean(patch_embeddings, dim=1)  # [B, embed_dim]

            # Apply L2 normalization (DINOv3 typically requires this)
            norm = torch.linalg.norm(emb.float(), dim=-1, keepdim=True).clamp(min=1e-8)
            emb = emb / norm.to(emb.dtype)

            # Get Score from HybridHeadModel Head
            with torch.no_grad():
                # Head expects float32 input
                prediction = self.head_model(emb.to(self.device, dtype=torch.float32))

            # Format Output (Score 0-10)
            prob_good_anatomy = 0.0
            output_mode = getattr(self.head_model, "output_mode", "linear")

            if self.num_classes == 1:
                # Single output neuron
                logit = prediction.squeeze().item()
                if output_mode == "linear":
                    prob_good_anatomy = torch.sigmoid(torch.tensor(logit)).item()
                elif output_mode == "sigmoid":
                    prob_good_anatomy = logit  # Already a probability
                elif output_mode == "tanh_scaled":
                    prob_good_anatomy = logit  # Already 0-1
                else:
                    logger.warning("Unknown output_mode '%s' for single output.", output_mode)

            elif self.num_classes == 2:
                # Two output neurons
                if output_mode == "linear":
                    # Apply softmax to logits
                    probs = F.softmax(prediction.squeeze(), dim=-1)
                    prob_good_anatomy = probs[1].item()  # Index 1 = "Good Anatomy"
                elif output_mode in ["sigmoid", "softmax"]:
                    # Assume output is already probabilities [P(bad), P(good)]
                    probs = prediction.squeeze()
                    prob_good_anatomy = probs[1].item()  # Index 1 = "Good Anatomy"
                else:
                    logger.warning("Unknown output_mode '%s' for dual output.", output_mode)
            else:
                logger.warning("Unexpected num_classes (%s) for scorer.", self.num_classes)

            # Convert to 0-10 score
            final_score = max(0.0, min(10.0, prob_good_anatomy * 10.0))
            return final_score

        except Exception as e:
            logger.error("Error during scoring: %s\n%s", e, traceback.format_exc())
            return -1.0
