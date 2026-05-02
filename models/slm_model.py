import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# LoRA Layer
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super().__init__()

        self.linear = linear_layer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_features) * 0.01)

    def forward(self, x, alpha=1.0):
        # x: (B, input_dim)

        base = self.linear(x)                 # (B, d_model)
        lora_update = x @ self.A @ self.B     # (B, d_model)

        alpha = alpha.view(-1, 1)             # (B,1)

        return base + alpha * lora_update


# -----------------------------
# SLM Transformer with AP-LoRA
# -----------------------------
class SLMTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=64, num_heads=4, num_layers=2, num_channels=4):
        super(SLMTransformer, self).__init__()

        # -----------------------------
        # Input Projection (LoRA)
        # -----------------------------
        self.input_linear = LoRALinear(nn.Linear(input_dim, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.input_act = nn.ReLU()

        # -----------------------------
        # Transformer Encoder
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # -----------------------------
        # Heads
        # -----------------------------
        self.power_head = nn.Linear(d_model, 1)
        self.channel_head = nn.Linear(d_model, num_channels)

        self.num_channels = num_channels

    # -----------------------------
    # Adaptive Scaling α(S, σ)
    # -----------------------------
    def compute_alpha(self, state, confidence):
        # state: (B,10)
        # confidence: (B,)

        interference = state[:, 8]

        lambda1 = 0.5
        lambda2 = 0.5

        alpha = lambda1 * interference + lambda2 * (1 - confidence)

        return alpha  # (B,)

    # -----------------------------
    # Forward Pass (FIXED)
    # -----------------------------
    def forward(self, x):

        # x: (B, input_dim)

        # dummy confidence for initial phase
        dummy_conf = torch.ones(x.size(0), device=x.device)

        # compute adaptive scaling
        alpha = self.compute_alpha(x, dummy_conf)

        # -----------------------------
        # LoRA projection (2D only)
        # -----------------------------
        x = self.input_linear(x, alpha=alpha)

        # -----------------------------
        # Add sequence dimension
        # -----------------------------
        x = x.unsqueeze(1)  # (B,1,d_model)

        x = self.input_norm(x)
        x = self.input_act(x)

        # -----------------------------
        # Transformer
        # -----------------------------
        x = self.transformer(x)
        x = x.squeeze(1)

        # -----------------------------
        # Outputs
        # -----------------------------
        power = torch.sigmoid(self.power_head(x))

        logits = self.channel_head(x)
        temperature = 2.0
        probs = F.softmax(logits / temperature, dim=-1)
        # -----------------------------
        # Confidence (entropy-based)
        # -----------------------------
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        max_entropy = torch.log(
            torch.tensor(self.num_channels, dtype=torch.float32, device=x.device)
        )

        confidence = 1 - (entropy / max_entropy)

        return power, logits, probs, confidence, None