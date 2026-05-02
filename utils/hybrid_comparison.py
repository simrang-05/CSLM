import numpy as np
import torch
import pandas as pd
import json
from stable_baselines3 import DDPG
from slm_model import SLMTransformer
from hybrid_system import hybrid_inference
from stable_baselines3 import SAC, PPO


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
slm = SLMTransformer(input_dim=10)
slm.load_state_dict(torch.load("slm_model.pth"))
slm.eval()

sac_model = SAC.load("sac_cr_model.zip")
ppo_model = PPO.load("ppo_cr_model.zip")
ddpg_model = DDPG.load("ddpg_cr_model")

# -----------------------------
# POLICIES
# -----------------------------
def slm_policy(X):
    actions = []
    with torch.no_grad():
        power, logits, probs, confidence, _ = slm(X)

    for i in range(len(X)):
        p = power[i].item()
        c = torch.argmax(probs[i]).item()
        actions.append((p, c))

    return actions


def rl_policy(model, X):
    actions = []
    for i in range(len(X)):
        action, _ = model.predict(X[i].numpy(), deterministic=True)

        p = float(action[0])
        c = int(abs(action[1]) * 4) % 4

        actions.append((p, c))

    return actions

def hybrid_ddpg(X):
    actions = []

    with torch.no_grad():
        power, logits, probs, confidence, _ = slm(X)

    for i in range(len(X)):

        sigma = confidence[i].item()

        if sigma >= 0.8:
            p = power[i].item()
            c = torch.argmax(probs[i]).item()
        else:
            action, _ = ddpg_model.predict(X[i].numpy(), deterministic=True)

            p = float(action[0])
            c = int(abs(action[1]) * 4) % 4

        actions.append((p, c))

    return actions

def hybrid_sac(X):
    actions, _, _ = hybrid_inference(X, threshold=0.8)
    return actions


def hybrid_ppo(X):
    actions = []

    with torch.no_grad():
        power, logits, probs, confidence, _ = slm(X)

    for i in range(len(X)):

        sigma = confidence[i].item()

        if sigma >= 0.8:
            p = power[i].item()
            c = torch.argmax(probs[i]).item()
        else:
            action, _ = ppo_model.predict(X[i].numpy(), deterministic=True)

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

    for i in range(len(actions)):

        p, c = actions[i]
        state = X_raw[i]

        h = state[c]
        interference = state[8]
        noise = state[9]

        sinr = (p * h) / (interference + noise + 1e-6)

        t = np.log2(1 + sinr)
        l = 1 / (sinr + 1e-6)
        i_val = interference

        v = 1 if sinr < 1.0 else 0

        r = t - 0.3*i_val - 0.2*l - 0.1*v

        T += t
        I += i_val
        L += l
        R += r
        violations += v

    n = len(actions)

    return {
        "throughput": T/n,
        "interference": I/n,
        "latency": L/n,
        "reward": R/n,
        "violations": violations/n
    }


# -----------------------------
# RUN COMPARISON
# -----------------------------
results = []

methods = {
    "SLM": slm_policy(X),
    "RL (PPO)": rl_policy(ppo_model, X),
    "RL (DDPG)": rl_policy(ddpg_model, X),
    "RL (SAC)": rl_policy(sac_model, X),
    "Hybrid (PPO)": hybrid_ppo(X),
    "Hybrid (DDPG)": hybrid_ddpg(X),
    "Hybrid (SAC)": hybrid_sac(X)
}

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
with open("hybrid_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Saved hybrid_comparison.json")