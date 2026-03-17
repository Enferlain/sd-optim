# hybrid_model.py
# Combines PredictorModel's optional self-attention with HeadModel's RMSNorm/SwiGLU MLP.
# Designed for single vector embeddings [B, F].

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# --- Helper Modules (Inspired by head_model.py, using RMSNorm/SwiGLU) ---


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate sqrt(E[x^2] + eps)
        norm = x.norm(2, dim=-1, keepdim=True)  # L2 norm
        rms = norm * (x.shape[-1] ** -0.5)  # RMS = L2 / sqrt(dim)
        # Original RMSNorm: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Let's stick to the simpler version if it works, or use the commented one if needed.
        return x / (rms + self.eps)
        # return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # Original calculation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply norm and scale by weight
        # Input assumed to be Float32 for stability
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 8 / 3 / 2 * 2)  # Standard SwiGLU expansion factor calculation
        # Ensure hidden_features is multiple of 2 for chunking
        hidden_features = (hidden_features + 1) // 2 * 2

        self.w12 = nn.Linear(in_features, hidden_features * 2, bias=False)  # Combined gate & up projection
        self.act = act_layer()
        self.dropout1 = nn.Dropout(dropout)  # Dropout after activation * value
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)  # Down projection
        self.dropout2 = nn.Dropout(dropout)  # Dropout before final output

    def forward(self, x):
        gate_val, up_val = self.w12(x).chunk(2, dim=-1)
        x = self.dropout1(self.act(gate_val) * up_val)
        x = self.dropout2(self.w3(x))
        return x


class ResBlockRMS(nn.Module):
    """Residual Block using RMSNorm and SwiGLUFFN"""

    def __init__(self, ch: int, dropout: float = 0.0, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(ch, eps=rms_norm_eps)
        self.ffn = SwiGLUFFN(in_features=ch, dropout=dropout)
        # Optional: Add LayerScale (like in head_model.py)
        # self.gamma = nn.Parameter(1e-6 * torch.ones(ch))

    def forward(self, x):
        # return x + self.gamma * self.ffn(self.norm(x)) # With LayerScale
        return x + self.ffn(self.norm(x))  # Without LayerScale


# --- New Hybrid Model ---


class HybridHeadModel(nn.Module):
    """
    Combines PredictorModel's optional self-attention with HeadModel's RMSNorm/SwiGLU MLP.
    Takes single embedding vectors [B, F] as input.
    """

    def __init__(
        self,
        features: int,
        hidden_dim: int = 1280,
        num_classes: int = 2,
        # PredictorModel style self-attention params
        use_attention: bool = True,
        num_attn_heads: int = 16,
        attn_dropout: float = 0.1,
        # HeadModel style MLP params
        num_res_blocks: int = 3,
        dropout_rate: float = 0.1,  # Overall dropout rate for FFN/proj
        rms_norm_eps: float = 1e-6,
        output_mode: str = "linear",
    ):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.output_mode = output_mode.lower()

        logger.debug("HybridHeadModel init")
        logger.debug("features=%s, hidden_dim=%s, num_classes=%s", features, hidden_dim, num_classes)
        logger.debug("use_attention=%s, heads=%s, attn_drop=%s", use_attention, num_attn_heads, attn_dropout)
        logger.debug(
            "num_res_blocks=%s, dropout_rate=%s, RMSNorm(eps=%s), output_mode='%s'",
            num_res_blocks,
            dropout_rate,
            rms_norm_eps,
            output_mode,
        )

        # --- Optional Self-Attention Layer (from PredictorModel) ---
        self.attention = None
        self.norm_attn = None
        if self.use_attention:
            actual_num_heads = num_attn_heads
            # Adjust heads based on input 'features' dimension
            if features % num_attn_heads != 0:
                possible_heads = [h for h in [1, 2, 4, 8, 16] if features % h == 0]  # Common powers of 2
                if not possible_heads:
                    raise ValueError(f"Attention Error: Features ({features}) not divisible by any standard head count.")
                actual_num_heads = min(possible_heads, key=lambda x: abs(x - num_attn_heads))
                logger.warning(
                    "Adjusting self-attention heads from %s to %s for features=%s.",
                    num_attn_heads,
                    actual_num_heads,
                    features,
                )

            self.attention = nn.MultiheadAttention(
                embed_dim=self.features,
                num_heads=actual_num_heads,
                dropout=attn_dropout,
                batch_first=True,
                bias=True,  # PredictorModel used bias=True here
            )
            # Use RMSNorm for the attention residual connection too? Or keep LayerNorm? Let's try RMSNorm.
            self.norm_attn = RMSNorm(self.features, eps=rms_norm_eps)
            logger.info("Using initial self-attention layer (Heads: %s).", actual_num_heads)
        # --- End Attention Layer ---

        # --- MLP Head (using RMSNorm / SwiGLU) ---
        mlp_layers = []
        # Initial projection + Norm
        mlp_layers.append(nn.Linear(self.features, self.hidden_dim))
        mlp_layers.append(RMSNorm(self.hidden_dim, eps=rms_norm_eps))
        # Note: No activation here like GELU in PredictorModel, SwiGLU handles activation in blocks

        # Configurable number of ResBlocks (using RMSNorm/SwiGLU version)
        for _ in range(num_res_blocks):
            mlp_layers.append(ResBlockRMS(ch=self.hidden_dim, dropout=dropout_rate, rms_norm_eps=rms_norm_eps))

        # Down projection block using SwiGLU
        # Example: hidden -> hidden//2 -> num_classes
        mlp_layers.append(RMSNorm(self.hidden_dim, eps=rms_norm_eps))  # Norm before down proj
        # Use SwiGLUFFN for the down projection step
        down_proj_hidden = self.hidden_dim // 2  # Or keep same? Let's reduce.
        mlp_layers.append(
            SwiGLUFFN(
                in_features=self.hidden_dim,
                hidden_features=down_proj_hidden,  # Adjust internal hidden dim if needed
                out_features=down_proj_hidden,
                dropout=dropout_rate,  # Apply dropout within SwiGLU
            )
        )
        mlp_layers.append(RMSNorm(down_proj_hidden, eps=rms_norm_eps))  # Norm after first down proj
        # Final linear layer
        mlp_layers.append(nn.Linear(down_proj_hidden, self.num_classes))

        self.mlp_head = nn.Sequential(*mlp_layers)
        # --- End MLP Head ---

        # --- Validate Output Mode ---
        valid_modes = ["linear", "sigmoid", "softmax", "tanh_scaled"]
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        # Print warnings based on num_classes and output_mode
        if self.output_mode == "softmax" and self.num_classes <= 1:
            logger.warning("output_mode='softmax' usually used with num_classes > 1.")
        if self.output_mode in ["sigmoid", "tanh_scaled"] and self.num_classes != 1:
            logger.warning("output_mode='%s' usually used with num_classes=1.", self.output_mode)

    def forward(self, x: torch.Tensor):
        # Input x shape: [Batch, Features]

        # --- Apply Self-Attention (Optional) ---
        if self.use_attention and self.attention is not None:
            # Add sequence dimension: [Batch, SeqLen=1, Features]
            x_seq = x.unsqueeze(1)
            attn_output, _ = self.attention(x_seq, x_seq, x_seq)  # Q=K=V
            # Residual connection + Norm
            x = self.norm_attn(x + attn_output.squeeze(1))  # Back to [Batch, Features]
        # --- End Apply Attention ---

        # --- MLP Head ---
        logits = self.mlp_head(x.float())  # Ensure input to MLP is float32 for RMSNorm stability
        # --- End MLP Head ---

        # --- Apply Final Activation based on output_mode ---
        output = None
        if self.output_mode == "linear":
            output = logits
        elif self.output_mode == "sigmoid":
            output = torch.sigmoid(logits)
        elif self.output_mode == "softmax":
            output = F.softmax(logits, dim=-1)
        elif self.output_mode == "tanh_scaled":
            output = (torch.tanh(logits) + 1.0) / 2.0
        else:
            raise RuntimeError(f"Invalid output_mode '{self.output_mode}'.")  # Should be caught by init

        # Ensure single output modes return shape [Batch] if num_classes=1
        if self.num_classes == 1 and output.ndim == 2 and output.shape[1] == 1:
            output = output.squeeze(-1)

        return output
