import numpy as np
import torch
import pandas as pd
import json
from energy_model import EnergyModel
from stable_baselines3 import SAC, PPO, DDPG

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load("X.npy")
X = torch.tensor(X, dtype=torch.float32)

df = pd.read_csv("wireless_dataset.csv")

X_raw = df[[
    "h1","h2","h3","h4",
    "sinr1","sinr2","sinr3","sinr4",
    "interference","noise"
]].values


# -----------------------------
# LOAD MODELS
# -----------------------------
sac_model = SAC.load("sac_cr_model.zip")
ppo_model = PPO.load("ppo_cr_model.zip")
ddpg_model = DDPG.load("ddpg_cr_model.zip")  # only if available

energy_model = EnergyModel()
# -----------------------------
# RL POLICY
# -----------------------------
def rl_policy(model, X):
    actions = []

    for i in range(len(X)):
        action, _ = model.predict(X[i].numpy(), deterministic=True)

        p = float(action[0])
        c = int(abs(action[1]) * 4) % 4

        actions.append((p, c))

    return actions


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(actions):

    T, I, L, R = 0,0,0,0
    violations = 0
    total_energy = 0

    for i in range(len(actions)):

        p, c = actions[i]
        # -----------------------------
        # ENERGY (RL ALWAYS USED)
        # -----------------------------
        energy = energy_model.compute(p, "SAC")
        total_energy += energy
        state = X_raw[i]

        h = state[c]
        interference = state[8]
        noise = state[9]

        sinr = (p * h) / (interference + noise + 1e-6)

        t = np.log2(1 + sinr)
        l = 1 / (sinr + 1e-6)
        i_val = interference

        v = 1 if sinr < 1.0 else 0

        r = t - 0.5*i_val - 0.3*l - v

        T += t
        I += i_val
        L += l
        R += r
        violations += v

    n = len(actions)

    avg_energy = total_energy / n
    energy_efficiency = (T/n) / avg_energy

    return {
        "throughput": T/n,
        "interference": I/n,
        "latency": L/n,
        "reward": R/n,
        "violations": violations/n,
        "energy": avg_energy,
        "energy_efficiency": energy_efficiency
    }


# -----------------------------
# RUN COMPARISON
# -----------------------------
results = []

methods = {
    "RL (PPO)": rl_policy(ppo_model, X),
    "RL (SAC)": rl_policy(sac_model, X),
}

# add DDPG only if model exists
try:
    methods["RL (DDPG)"] = rl_policy(ddpg_model, X)
except:
    print("DDPG not found, skipping...")

for name, actions in methods.items():

    metrics = evaluate(actions)

    entry = {"method": name}
    entry.update(metrics)

    results.append(entry)

    print(f"\n{name}")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")


# -----------------------------
# SAVE
# -----------------------------
with open("rl_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Saved rl_comparison.json")