import torch
import numpy as np
import pandas as pd
import json

from slm_model import SLMTransformer


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
# LOAD MODEL
# -----------------------------
model = SLMTransformer(input_dim=10)
model.load_state_dict(torch.load("slm_model.pth"))
model.eval()


# -----------------------------
# ABLATION MODES
# -----------------------------

def slm_no_lora(X):
    actions = []

    with torch.no_grad():
        power, logits, probs, confidence, _ = model(X)

    for i in range(len(X)):
        p = power[i].item()
        c = torch.argmax(probs[i]).item()
        actions.append((p, c))

    return actions


def slm_lora_static(X, alpha=0.3):
    actions = []

    with torch.no_grad():
        power, logits, probs, confidence, _ = model(X)

    for i in range(len(X)):
        p = power[i].item() * (1 + alpha*0.2)
        c = torch.argmax(probs[i]).item()
        actions.append((min(p,1.0), c))

    return actions


def slm_alora(X):
    actions = []

    with torch.no_grad():
        power, logits, probs, confidence, _ = model(X)

    for i in range(len(X)):

        sigma = confidence[i].item()
        state = X_raw[i]

        interference = state[8]
        noise = state[9]

        # normalized interference
        I_norm = interference / (interference + noise + 1e-6)

        # controlled adaptive factor (VERY IMPORTANT: small range)
        alpha = 0.9 * I_norm + 0.5 * (1 - sigma)

        # channel remains SAME (important for stability)
        c = torch.argmax(probs[i]).item()

        # controlled power adjustment (balanced)
        p = power[i].item() * (1 + alpha)

        # slight correction to avoid instability
        p = min(max(p, 0.2), 1.0)

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

        r = t - 0.5*i_val - 0.3*l - v

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
# RUN
# -----------------------------
results = []

methods = {
    "SLM": slm_no_lora(X),
    "SLM + LoRA": slm_lora_static(X),
    "SLM + ALoRA": slm_alora(X)
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
with open("slm_ablation.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Saved slm_ablation.json")