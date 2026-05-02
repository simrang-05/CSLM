import numpy as np
import torch
import pandas as pd
import os
from hybrid_system import hybrid_inference
from logger import log_to_file
from energy_model import EnergyModel

# -----------------------------
# CLEAR OLD LOGS
# -----------------------------
if os.path.exists("results/metrics.jsonl"):
    os.remove("results/metrics.jsonl")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("wireless_dataset.csv")

X_raw = df[[
    "h1", "h2", "h3", "h4",
    "sinr1", "sinr2", "sinr3", "sinr4",
    "interference", "noise"
]].values

X_scaled = np.load("X.npy")
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# -----------------------------
# RUN SYSTEM
# -----------------------------
actions, sources, confidence = hybrid_inference(X_tensor, threshold=0.8)

# -----------------------------
# METRIC WEIGHTS
# -----------------------------
ALPHA = 2.0   # throughput weight
BETA = 0.3    # interference penalty
GAMMA = 0.2   # latency penalty
DELTA = 0.1   # power penalty

print("\n📊 Simulation Metrics\n")
energy_model = EnergyModel()

total_energy = 0
energy_list = []

slm_energy = 0
sac_energy = 0
slm_count = 0
sac_count = 0

total = len(X_raw)

avg_T = 0
avg_I = 0
avg_L = 0
avg_R = 0

# -----------------------------
# LOOP
# -----------------------------
for i in range(total):

    state = X_raw[i]
    power, channel = actions[i]
    
    source = sources[i]
    # -----------------------------
    # ENERGY CALCULATION
    # -----------------------------
    energy = energy_model.compute(power, source)

    total_energy += energy
    energy_list.append(energy)

    if source == "SLM":
        slm_energy += energy
        slm_count += 1
    else:
        sac_energy += energy
        sac_count += 1

    h = state[channel]
    interference = state[8]
    noise = state[9]

    # -----------------------------
    # COMPUTE METRICS
    # -----------------------------
    sinr = (power * h) / (interference + noise + 1e-6)
    throughput = np.log2(1 + sinr)
    latency = 1 / (sinr + 1e-6)

    SINR_THRESHOLD = 1.0

    penalty = 0

    if sinr < SINR_THRESHOLD:
        penalty = -2.0 * (SINR_THRESHOLD - sinr)   # strong penalty

    reward = (
        2.0 * throughput
        - 0.3 * interference
        - 0.2 * latency
        - 0.1 * power
        + penalty
    )

    avg_T += throughput
    avg_I += interference
    avg_L += latency
    avg_R += reward

    # -----------------------------
    # LOG EACH SAMPLE
    # -----------------------------
    log_entry = {
        "throughput": float(throughput),
        "energy": float(energy),
        "interference": float(interference),
        "latency": float(latency),
        "reward": float(reward),
        "power": float(power),
        "channel": int(channel),
        "source": sources[i]
    }

    log_to_file("results/metrics.jsonl", log_entry)

# -----------------------------
# AVERAGES
# -----------------------------
avg_T /= total
avg_I /= total
avg_L /= total
avg_R /= total

print(f"Avg Throughput: {avg_T:.3f}")
print(f"Avg Interference: {avg_I:.3f}")
print(f"Avg Latency: {avg_L:.3f}")
print(f"Avg Reward: {avg_R:.3f}")
# -----------------------------
# ENERGY RESULTS
# -----------------------------
avg_energy = total_energy / total
print(f"Avg Energy: {avg_energy:.3f}")

energy_efficiency = avg_T / avg_energy
print(f"Energy Efficiency: {energy_efficiency:.3f}")

if slm_count > 0:
    print(f"SLM Avg Energy: {slm_energy/slm_count:.3f}")

if sac_count > 0:
    print(f"SAC Avg Energy: {sac_energy/sac_count:.3f}")
    
# -----------------------------
# BASELINE COMPARISON
# -----------------------------
E_slm = 1.0
E_rl = 5.0

slm_only_energy = 0
sac_only_energy = 0

for i in range(total):
    power, _ = actions[i]

    # SLM-only (no RL ever)
    slm_only_energy += E_slm + power

    # SAC-only (RL always used)
    sac_only_energy += E_slm + E_rl + power

slm_only_energy /= total
sac_only_energy /= total

print("\n📊 Baseline Energy Comparison")
print(f"SLM Only Energy: {slm_only_energy:.3f}")
print(f"SAC Only Energy: {sac_only_energy:.3f}")
print(f"CSLM (Hybrid) Energy: {avg_energy:.3f}")