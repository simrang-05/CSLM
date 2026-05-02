import numpy as np
import torch
from hybrid_system import hybrid_inference

def run_feedback_loop(X_scaled, X_raw, steps=5):

    print("\n🔁 Running Feedback Loop...\n")

    for step in range(steps):

        actions, sources, confidence = hybrid_inference(X_scaled, threshold=0.8)

        rewards = []

        for i in range(len(X_raw)):

            state = X_raw[i]
            power, channel = actions[i]

            h = state[channel]
            interference = state[8]
            noise = state[9]

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

            rewards.append(reward)

        avg_reward = np.mean(rewards)

        print(f"Step {step+1} | Avg Reward: {avg_reward:.3f}")