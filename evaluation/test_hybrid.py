import os
import torch
import numpy as np

from hybrid_system import hybrid_inference

if os.path.exists("results/hybrid_logs.jsoln"):
    os.remove("results/hybrid_logs.jsonl")
X = np.load("X.npy")
X = torch.tensor(X, dtype=torch.float32)

actions, sources, confidence = hybrid_inference(X[:10], threshold=0.8)

for i in range(10):
    power, channel = actions[i]
    print(f"\nSample {i}")
    print("Source:", sources[i])
    print(f"Action: ({float(power):.3f}, {int(channel)})")
    print(f"Confidence: {confidence[i]:.4f}")