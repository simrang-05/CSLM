import torch
import numpy as np
from slm_model import SLMTransformer
from logger import log_to_file
import os

if os.path.exists("results/threshold_analysis.jsonl"):
    os.remove("results/threshold_analysis.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = SLMTransformer(input_dim=10).to(device)
model.load_state_dict(torch.load("slm_model.pth"))
model.eval()

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load("X.npy")
y_channel = np.load("y_channel.npy")

X = torch.tensor(X, dtype=torch.float32).to(device)
y_channel = torch.tensor(y_channel).to(device)

# -----------------------------
# THRESHOLDS TO TEST
# -----------------------------
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

print("\n🔍 Threshold Analysis\n")

for tau in thresholds:

    slm_used = 0
    correct = 0
    total = 0

    for i in range(len(X)):

        x = X[i].unsqueeze(0)

        with torch.no_grad():
            _, logits, probs, confidence, _ = model(x)

        sigma = confidence.item()

        if sigma >= tau:
            slm_used += 1

            pred_channel = torch.argmax(probs).item()
            true_channel = y_channel[i].item()

            if pred_channel == true_channel:
                correct += 1

        total += 1

    slm_ratio = slm_used / total
    fallback_ratio = 1 - slm_ratio
    acc = correct / slm_used if slm_used > 0 else 0

    print(f"τ = {tau}")
    print(f"SLM Usage: {slm_ratio:.3f}")
    print(f"Fallback Usage: {fallback_ratio:.3f}")
    print(f"SLM Accuracy: {acc:.3f}")
    print("-" * 40)

    # ✅ LOG INSIDE LOOP
    log_entry = {
        "threshold": tau,
        "slm_usage": float(slm_ratio),
        "fallback_usage": float(fallback_ratio),
        "slm_accuracy": float(acc)
    }

    log_to_file("threshold_analysis.jsonl", log_entry)