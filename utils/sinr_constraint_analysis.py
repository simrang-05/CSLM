import numpy as np
import torch
import os
from hybrid_system import hybrid_inference
from logger import log_to_file

# -----------------------------
# CLEAR OLD LOGS
# -----------------------------
if os.path.exists("results/sinr_analysis.jsonl"):
    os.remove("results/sinr_analysis.jsonl")

# -----------------------------
# LOAD DATA
# -----------------------------
import pandas as pd

df = pd.read_csv("wireless_dataset.csv")

# Extract RAW features (NO scaling)
X_raw = df[[
    "h1", "h2", "h3", "h4",
    "sinr1", "sinr2", "sinr3", "sinr4",
    "interference", "noise"
]].values

# Load scaled input for model
X_scaled = np.load("X.npy")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# -----------------------------
# SINR THRESHOLDS
# -----------------------------
sinr_thresholds = [0.5, 1.0, 1.5, 2.0]

print("\n📡 SINR Threshold Analysis\n")

# -----------------------------
# RUN HYBRID ONCE (IMPORTANT)
# -----------------------------
actions, sources, _ = hybrid_inference(X_tensor, threshold=0.8)

# -----------------------------
# LOOP OVER GAMMA
# -----------------------------
for gamma in sinr_thresholds:

    total = len(X_raw)
    violations = 0
    avg_power = 0
    avg_sinr = 0

    for i in range(total):

        state = X_raw[i]
        power, channel = actions[i]

        # -----------------------------
        # CORRECT SINR CALCULATION
        # -----------------------------
        h = state[channel]
        interference = state[8]
        noise = state[9]

        sinr = (power * h) / (interference + noise + 1e-6)

        avg_power += power
        avg_sinr += sinr

        if sinr < gamma:
            violations += 1

    avg_power /= total
    avg_sinr /= total
    violation_rate = violations / total

    print(f"Γ = {gamma}")
    print(f"Violation Rate: {violation_rate:.3f}")
    print(f"Avg Power: {avg_power:.3f}")
    print(f"Avg SINR: {avg_sinr:.3f}")
    print("-" * 40)

    log_entry = {
        "sinr_threshold": gamma,
        "violation_rate": float(violation_rate),
        "avg_power": float(avg_power),
        "avg_sinr": float(avg_sinr)
    }

    log_to_file("sinr_analysis.jsonl", log_entry)