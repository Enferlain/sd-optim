# Scorer class for AnatomyFlaws models using SigLIP NaFlex embeddings
# and the custom HybridHeadModel head.

import os
import json
import traceback
import logging

import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin  # Import base Image and PngImageFile type
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoModel

# --- Make sure HybridHeadModel can be imported ---
try:
    # Assuming lumi_model.py is in the same directory or accessible via PYTHONPATH
    from sd_optim.extensions.bundled.scorers.models.lumi_model import HybridHeadModel

    logger = logging.getLogger(__name__)
    logger.debug("Successfully imported HybridHeadModel.")
except ImportError:
    raise ImportError("Could not import HybridHeadModel. Make sure lumi_model.py is accessible.")

# Assuming utils defines get_embed_params if needed elsewhere, but not strictly needed here
# from utils import get_embed_params

logger = logging.getLogger(__name__)


# --- Preprocessing Function (Only FitPad needed as fallback/placeholder if we adapt later) ---
# Although SigLIP Naflex uses the processor directly, having a preprocess func
# available might be useful for consistency or future adaptations.
# Let's keep FitPad from the original scorer for now.
def preprocess_fit_pad(img_pil, target_size=512, fill_color=(0, 0, 0)):
    """Resizes image to fit, pads to target size."""
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0:
        return None
    target_w, target_h = target_size, target_size
    scale = min(target_w / original_width, target_h / original_height)
    new_w = int(original_width * scale)
    new_h = int(original_height * scale)
    if new_w == 0:
        new_w = 1
    if new_h == 0:
        new_h = 1
    try:
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_padded = Image.new(img_pil.mode, (target_w, target_h), fill_color)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        img_padded.paste(img_resized, (pad_left, pad_top))
        return img_padded
    except Exception as e:
        logger.error("Error during preprocess_fit_pad: %s", e)
        return None


# --- End Preprocessing ---


class HybridAnatomyScorer:
    """
    Scorer class for AnatomyFlaws models using SigLIP NaFlex embeddings
    and the custom HybridHeadModel head. Outputs a score 0-10.
    """

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

        # Models and processor attributes
        self.config = None
        self.base_vision_model_name = None
        self.vision_model = None
        self.hf_processor = None
        self.head_model: HybridHeadModel | None = None  # Type hint for clarity
        self.num_classes = 0

        self.initialize_models()

    def initialize_models(self):
        # --- Load Config ---
        logger.info("Loading config from: %s", self.config_path)
        try:
            # <<< CHANGE THIS LINE >>>
            # Original:
            # with open(self.config_path, 'r') as f: self.config = json.load(f)
            # Fixed:
            with open(self.config_path, encoding="utf-8") as f:
                self.config = json.load(f)
            # <<< END CHANGE >>>
        except Exception as e:
            raise OSError(f"Failed to load config: {e}")

        # Get vision model name (expecting SigLIP NaFlex)
        self.base_vision_model_name = self.config.get("base_vision_model")
        if (
            not self.base_vision_model_name
            or "siglip" not in self.base_vision_model_name.lower()
            or "naflex" not in self.base_vision_model_name.lower()
        ):
            raise ValueError(f"Config missing or specifies non-SigLIP-NaFlex 'base_vision_model': {self.base_vision_model_name}")
        logger.info("Using base vision model: %s", self.base_vision_model_name)

        # --- Initialize Vision Model (SigLIP NaFlex Only) ---
        logger.info("Initializing SigLIP vision model and processor...")
        try:
            self.hf_processor = AutoProcessor.from_pretrained(self.base_vision_model_name)
            self.vision_model = (
                AutoModel.from_pretrained(
                    self.base_vision_model_name,
                    torch_dtype=self.compute_dtype,
                    trust_remote_code=True,  # May not be needed for SigLIP but safe to include
                )
                .to(self.device)
                .eval()
            )
            logger.info("Loaded HF model: %s", self.vision_model.__class__.__name__)
        except Exception as e:
            raise RuntimeError(f"Failed to load SigLIP vision model/processor '{self.base_vision_model_name}': {e}")

        # --- Initialize Custom HybridHeadModel Head ---
        logger.info("Initializing HybridHeadModel head from: %s", self.model_path)
        try:
            # Load state dict first to potentially infer num_classes if needed
            state_dict = load_file(self.model_path, device="cpu")

            # Infer features (should be 1152 for so400m)
            features = self.config.get("features", 1152)  # Get from config or default

            # Infer num_classes robustly - check state dict, then config
            self.num_classes = 0
            potential_final_bias_keys = [k for k in state_dict if k.endswith(".bias")]
            if potential_final_bias_keys:
                # Assume final layer is the last one in mlp_head sequential
                max_idx = -1
                final_bias_key = None
                for k in potential_final_bias_keys:
                    if k.startswith("mlp_head."):
                        parts = k.split(".")
                        if len(parts) == 3 and parts[1].isdigit():
                            idx = int(parts[1])
                            if idx > max_idx:
                                max_idx = idx
                                final_bias_key = k
                if final_bias_key:
                    self.num_classes = state_dict[final_bias_key].shape[0]
                    logger.debug("Inferred num_classes=%s from state_dict key '%s'", self.num_classes, final_bias_key)

            if self.num_classes == 0:  # Fallback to config if inference failed
                self.num_classes = self.config.get("num_classes", 2)  # Default to 2 if missing
                logger.debug("Using num_classes=%s from config (or default).", self.num_classes)

            # Load other hyperparameters from config (use top-level args saved by write_config)
            output_mode = self.config.get("output_mode", self.config.get("head_output_mode", "linear"))
            hidden_dim = self.config.get("hidden_dim", self.config.get("head_hidden_dim", 1280))
            num_res_blocks = self.config.get("num_res_blocks", self.config.get("head_num_res_blocks", 3))
            dropout_rate = self.config.get("dropout_rate", self.config.get("head_dropout_rate", 0.1))
            use_attention = self.config.get("use_attention", False)  # Default OFF for Hybrid unless specified
            num_attn_heads = self.config.get("num_attn_heads", 16)
            attn_dropout = self.config.get("attn_dropout", 0.1)
            rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)

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
            missing, unexpected = self.head_model.load_state_dict(state_dict, strict=False)  # Use strict=False for robustness
            if missing:
                logger.warning("Missing keys when loading head state_dict: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys when loading head state_dict: %s", unexpected)

            self.head_model.to(self.device).eval()  # Move to device and set to eval

        except Exception as e:
            raise RuntimeError(f"Failed to initialize or load HybridHeadModel head: {e}")

        logger.info("Models initialized successfully.")

    def score(self, image: Image.Image | str | PngImagePlugin.PngImageFile, prompt=None) -> float:
        """Calculates the 'Good Anatomy' score (0-10) for an image."""
        pil_image = None
        try:  # --- Image Loading ---
            if isinstance(image, (Image.Image, PngImagePlugin.PngImageFile)):
                pil_image = image.convert("RGB")
            elif isinstance(image, str) and os.path.isfile(image):
                pil_image = Image.open(image).convert("RGB")
            else:
                raise TypeError("Input must be a PIL Image or a valid file path.")
        except Exception as e:
            logger.error("Error loading image: %s", e)
            return -1.0

        try:
            emb = None
            # --- 1. Extract SigLIP NaFlex Embedding ---
            with torch.no_grad():
                # Expects raw PIL image
                inputs = self.hf_processor(images=[pil_image], return_tensors="pt", max_num_patches=1024)  # Use 1024 default
                pixel_values = inputs.get("pixel_values").to(device=self.device, dtype=self.compute_dtype)
                attention_mask = inputs.get("pixel_attention_mask").to(device=self.device)
                spatial_shapes = inputs.get("spatial_shapes")
                model_call_kwargs = {
                    "pixel_values": pixel_values,
                    "attention_mask": attention_mask,
                    "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(self.device),
                }

                # Call model (SigLIP specific methods)
                vision_model_component = getattr(self.vision_model, "vision_model", None)
                if vision_model_component:
                    emb = vision_model_component(**model_call_kwargs).pooler_output
                elif hasattr(self.vision_model, "get_image_features"):
                    # Need to handle potential arg differences if using get_image_features directly
                    kwargs_for_get = {
                        k: v for k, v in model_call_kwargs.items() if k in ["pixel_values", "attention_mask", "spatial_shapes"]
                    }
                    emb = self.vision_model.get_image_features(**kwargs_for_get)
                else:
                    raise AttributeError("SigLIP Model missing expected methods.")

                if emb is None:
                    raise ValueError("Failed to get embedding.")

                # Apply L2 Norm (matching v6.6+ training)
                norm = torch.linalg.norm(emb.float(), dim=-1, keepdim=True).clamp(min=1e-8)
                emb = emb / norm.to(emb.dtype)

            # --- 2. Obtain Score from HybridHeadModel Head ---
            with torch.no_grad():
                # Head expects float32 input
                prediction = self.head_model(emb.to(self.device, dtype=torch.float32))

            # --- 3. Format Output (Score 0-10) ---
            prob_good_anatomy = 0.0
            output_mode = getattr(self.head_model, "output_mode", "linear")  # Get mode from loaded head

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
                elif output_mode == "sigmoid" or output_mode == "softmax":
                    # Assume output is already probabilities [P(bad), P(good)]
                    probs = prediction.squeeze()
                    prob_good_anatomy = probs[1].item()  # Index 1 = "Good Anatomy"
                else:
                    logger.warning("Unknown output_mode '%s' for dual output.", output_mode)
            else:  # More than 2 classes? Not expected for scorer.
                logger.warning("Unexpected num_classes (%s) for scorer.", self.num_classes)

            final_score = max(0.0, min(10.0, prob_good_anatomy * 10.0))  # Clamp score 0-10
            return final_score

        except Exception as e:
            logger.error("Error during scoring: %s\n%s", e, traceback.format_exc())
            return -1.0  # Indicate error
