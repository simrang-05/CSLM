import torch
import numpy as np
from slm_model import SLMTransformer

# Load data
X = np.load("X.npy")
X = torch.tensor(X, dtype=torch.float32)

# Model
model = SLMTransformer()

# Forward pass (first 5 samples)
power, probs, confidence = model(X[:5])

print("Power:\n", power)
print("\nChannel Probabilities:\n", probs)
print("\nConfidence:\n", confidence)