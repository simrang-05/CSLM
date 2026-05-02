from stable_baselines3 import PPO
from rl_env import CognitiveRadioEnv

env = CognitiveRadioEnv()

model = PPO.load("ppo_cr_model")

state, _ = env.reset()

for _ in range(5):
    action, _ = model.predict(state)
    state, reward, done, truncated, _ = env.step(action)

    print("Action:", action)
    print("Reward:", reward)