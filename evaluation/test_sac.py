import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import SAC
from cr_env import CR_Environment
import numpy as np

model = SAC.load("c:/Users/User/slm_cr/sac_cr_model.zip")

env = CR_Environment()

state, _ = env.reset()

print("\nTesting SAC Agent...\n")

for i in range(10):

    action, _ = model.predict(state)

    next_state, reward, terminated, truncated, _ = env.step(action)

    power = action[0]
    channel = int(np.clip(round(action[1]), 0, 3))

    print(f"Step {i}")
    print(f"Power: {power:.3f} | Channel: {channel}")
    print(f"Reward: {reward:.3f}")
    print("-" * 40)

    state = next_state