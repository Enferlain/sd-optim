# model.py
# Version 2.0.0: Enhanced flexibility and configuration

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# --- ResBlock (remains the same) ---
class ResBlock(nn.Module):
    """Standard Residual Block with LayerNorm."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.long = nn.Sequential(
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.Linear(ch, ch),
        )

    def forward(self, x):
        return x + self.long(self.norm(x))


# --- Enhanced PredictorModel ---
class PredictorModel(nn.Module):
    """
    Enhanced Predictor/Classifier head with configurable options.

    Args:
        features (int): Dimension of the input embedding (e.g., 768, 1152).
        hidden_dim (int): Dimension of the main hidden layer.
        num_classes (int): Number of output neurons.
                           - 1 for regression/scoring (MSE/L1) or binary classification with BCEWithLogitsLoss.
                           - 2 (or more) for classification with CrossEntropyLoss/FocalLoss.
        use_attention (bool): Whether to use a self-attention layer on the input embedding.
        num_attn_heads (int): Number of heads for the MultiheadAttention layer (if used).
        attn_dropout (float): Dropout rate for the MultiheadAttention layer (if used).
        num_res_blocks (int): Number of ResBlocks in the main processing block.
        dropout_rate (float): Dropout rate for the final hidden layers.
        output_mode (str): Determines the final activation/output format.
                           - 'linear': Raw logits (for BCEWithLogitsLoss, CrossEntropyLoss, FocalLoss, MSELoss, L1Loss).
                           - 'sigmoid': Apply sigmoid (for standalone binary prediction probability).
                           - 'softmax': Apply softmax (for multi-class prediction probability).
                           - 'tanh_scaled': Apply tanh and scale to [0, 1] (for normalized score prediction).
    """

    def __init__(
        self,
        features=1152,
        hidden_dim=1280,
        num_classes=2,  # Default for binary classification (CE/Focal)
        use_attention=True,
        num_attn_heads=8,
        attn_dropout=0.1,
        num_res_blocks=1,
        dropout_rate=0.1,
        output_mode="linear",  # Default to outputting raw logits
    ):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.output_mode = output_mode.lower()

        logger.debug("PredictorModel v2.0.0 init")
        logger.debug("features=%s, hidden_dim=%s, num_classes=%s", features, hidden_dim, num_classes)
        logger.debug("use_attention=%s, heads=%s, attn_drop=%s", use_attention, num_attn_heads, attn_dropout)
        logger.debug("num_res_blocks=%s, dropout_rate=%s, output_mode='%s'", num_res_blocks, dropout_rate, output_mode)

        # --- Optional Attention Layer ---
        self.attention = None
        self.norm_attn = None
        if self.use_attention:
            # Ensure num_heads is valid
            actual_num_heads = num_attn_heads
            if features % num_attn_heads != 0:
                possible_heads = [h for h in [1, 2, 4, 6, 8, 12, 16] if features % h == 0]
                if not possible_heads:
                    raise ValueError(f"Features ({features}) not divisible by any standard head count.")
                actual_num_heads = min(possible_heads, key=lambda x: abs(x - num_attn_heads))
                logger.warning(
                    "Adjusting attention heads from %s to %s for features=%s.",
                    num_attn_heads,
                    actual_num_heads,
                    features,
                )
            self.attention = nn.MultiheadAttention(
                embed_dim=self.features,
                num_heads=actual_num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.norm_attn = nn.LayerNorm(self.features)
        # --- End Attention Layer ---

        # --- MLP Head ---
        # Initial projection to hidden_dim
        self.initial_proj = nn.Linear(self.features, self.hidden_dim)
        self.norm_initial = nn.LayerNorm(self.hidden_dim)
        self.activation_initial = nn.GELU()

        # Configurable number of ResBlocks
        self.res_blocks = nn.Sequential(*[ResBlock(ch=self.hidden_dim) for _ in range(num_res_blocks)])

        # Down projection block with configurable dropout
        self.down = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        # Final layer to output the required number of classes/logits
        self.final_layer = nn.Linear(32, self.num_classes)
        # --- End MLP Head ---

        # --- Validate Output Mode ---
        valid_modes = ["linear", "sigmoid", "softmax", "tanh_scaled"]
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        if self.output_mode == "softmax" and self.num_classes <= 1:
            logger.warning("output_mode='softmax' usually used with num_classes > 1 (got %s).", self.num_classes)
        if self.output_mode in ["sigmoid", "tanh_scaled"] and self.num_classes != 1:
            logger.warning(
                "output_mode='%s' usually used with num_classes=1 (got %s).",
                self.output_mode,
                self.num_classes,
            )

    def forward(self, x):
        # x original shape: [Batch, Features]

        # --- Apply Attention (Optional) ---
        if self.use_attention and self.attention is not None:
            # Add sequence dimension: [Batch, SeqLen=1, Features]
            x_seq = x.unsqueeze(1)
            attn_output, _ = self.attention(x_seq, x_seq, x_seq)
            # Residual connection + LayerNorm
            x = self.norm_attn(x + attn_output.squeeze(1))
        # --- End Apply Attention ---

        # --- MLP Head ---
        x = self.initial_proj(x)
        x = self.norm_initial(x)
        x = self.activation_initial(x)
        x = self.res_blocks(x)
        x = self.down(x)
        logits = self.final_layer(x)
        # --- End MLP Head ---

        # --- Apply Final Activation based on output_mode ---
        if self.output_mode == "linear":
            # Return raw logits - expected by BCEWithLogitsLoss, CrossEntropyLoss, FocalLoss, MSELoss, L1Loss
            output = logits
        elif self.output_mode == "sigmoid":
            # Apply sigmoid - use with nn.BCELoss (less stable) or for direct probability
            output = torch.sigmoid(logits)
        elif self.output_mode == "softmax":
            # Apply softmax - use with nn.NLLLoss or for multi-class probabilities
            output = F.softmax(logits, dim=-1)  # Use dim=-1 for flexibility
        elif self.output_mode == "tanh_scaled":
            # Apply tanh and scale to [0, 1] - for normalized scores
            output = (torch.tanh(logits) + 1.0) / 2.0
        else:
            # Should be caught by __init__, but for safety
            raise RuntimeError(f"Invalid output_mode '{self.output_mode}' in forward pass.")

        # Ensure single output modes return shape [Batch] if num_classes=1
        if self.num_classes == 1 and output.ndim == 2 and output.shape[1] == 1:
            output = output.squeeze(-1)

        return output
