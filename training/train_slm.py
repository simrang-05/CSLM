import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from slm_model import SLMTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load("X.npy")
y_power = np.load("y_power.npy")
y_channel = np.load("y_channel.npy")

# 🔥 NORMALIZE POWER (IMPORTANT)
y_power = (y_power - y_power.min()) / (y_power.max() - y_power.min() + 1e-8)

X = torch.tensor(X, dtype=torch.float32).to(device)
y_power = torch.tensor(y_power, dtype=torch.float32).unsqueeze(1).to(device)
y_channel = torch.tensor(y_channel, dtype=torch.long).to(device)

# -----------------------------
# MODEL
# -----------------------------
model = SLMTransformer(input_dim=10).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)

# 🔥 BETTER LOSS
power_loss_fn = nn.SmoothL1Loss()
channel_loss_fn = nn.CrossEntropyLoss()

EPOCHS = 20
BATCH_SIZE = 64

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):

    perm = torch.randperm(len(X))
    total_loss = 0

    all_preds = []
    all_targets = []

    for i in range(0, len(X), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]

        xb = X[idx]
        pb = y_power[idx]
        cb = y_channel[idx]

        pred_power, logits, probs, confidence, _ = model(xb)

        loss_p = power_loss_fn(pred_power, pb)
        loss_c = channel_loss_fn(logits, cb)

        # 🔥 FIXED LOSS BALANCE
        loss = 0.3 * loss_p + 1.0 * loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # store for accuracy
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds)
        all_targets.append(cb)

    avg_loss = total_loss / (len(X) // BATCH_SIZE)

    # -----------------------------
    # CORRECT ACCURACY
    # -----------------------------
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    acc = (all_preds == all_targets).float().mean()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {avg_loss:.4f} | Channel Acc: {acc.item():.4f}")

    # -----------------------------
    # SAMPLE DEBUG
    # -----------------------------
    print("\nSample Predictions:")

    print("Power Pred vs True:")
    print(pred_power[:3].detach().squeeze().cpu().numpy(), " | ",
      pb[:3].detach().squeeze().cpu().numpy())

    print("Channel Pred vs True:")
    print(preds[:5].detach().cpu().numpy(), " | ",
      cb[:5].detach().cpu().numpy())

    print("-" * 50)

torch.save(model.state_dict(), "slm_model.pth")

print("\n✅ SLM training completed!")