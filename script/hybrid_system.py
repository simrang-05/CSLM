import torch
import numpy as np
from slm_model import SLMTransformer
from stable_baselines3 import SAC
from logger import log_to_file
from joint_optimization import joint_optimize
import os

if os.path.exists("results/hybrid_logs.jsonl"):
    os.remove("results/hybrid_logs.jsonl")
# -----------------------------
# LOAD MODELS
# -----------------------------
model = SLMTransformer(input_dim=10)
model.load_state_dict(torch.load("slm_model.pth"))
model.eval()

sac_model = SAC.load("sac_cr_model")


# -----------------------------
# HYBRID INFERENCE
# -----------------------------
def hybrid_inference(X_batch, threshold=0.8):

    with torch.no_grad():
        power, logits, probs, confidence, _ = model(X_batch)

    actions = []
    sources = []

    for i in range(len(X_batch)):

        sigma = confidence[i].item()

        # -----------------------------
        # SLM DECISION
        # -----------------------------
        if sigma >= threshold:

            selected_power = float(power[i].item())
            selected_channel = int(torch.argmax(probs[i]).item())
            # -----------------------------
            # JOINT OPTIMIZATION
            # -----------------------------
            state = X_batch[i].cpu().numpy()

            opt_power, opt_channel = joint_optimize(
                state,
                selected_power,
                selected_channel
            )

            selected_power = float(opt_power)
            selected_channel = int(opt_channel)
            source = "SLM"

        # -----------------------------
        # SAC FALLBACK
        # -----------------------------
        else:
            state = X_batch[i].numpy()

            action, _ = sac_model.predict(state)

            selected_power = float(action[0])
            selected_channel = int(np.clip(round(action[1]), 0, 3))
            source = "SAC"

        actions.append((selected_power, selected_channel))
        sources.append(source)

        # -----------------------------
        # LOGGING
        # -----------------------------
        log_entry = {
            "state": X_batch[i].cpu().numpy().tolist(),
            "confidence": float(sigma),
            "decision": source,

            "slm_power": float(power[i].item()),
            "slm_channel": int(torch.argmax(probs[i]).item()),

            "final_power": float(selected_power),
            "final_channel": int(selected_channel)
        }

        log_to_file("hybrid_logs.jsonl", log_entry)

    return actions, sources, confidence.numpy()