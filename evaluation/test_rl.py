import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from cr_env import CR_Environment

# Load model
model = PPO.load("C:/Users/User/slm_cr/ppo_cr_model")

env = CR_Environment()

state, _ = env.reset()

print("\nTesting RL Agent...\n")

for i in range(10):

    action, _ = model.predict(state)

    next_state, reward, terminated, truncated, _ = env.step(action)

    power = action[0]
    channel = int(round(action[1]))

    print(f"Step {i}")
    print(f"Power: {power:.3f} | Channel: {channel}")
    print(f"Reward: {reward:.3f}")
    print("-" * 40)

    state = next_state