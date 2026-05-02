from stable_baselines3 import DDPG
from cr_env import CR_Environment
# -----------------------------
# CREATE ENVIRONMENT
# -----------------------------
env = CR_Environment()
# -----------------------------
# TRAIN MODEL
# -----------------------------
model = DDPG("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=20000)

# -----------------------------
# SAVE
# -----------------------------
model.save("ddpg_cr_model")

print("✅ DDPG training complete")