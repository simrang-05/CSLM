from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from cr_env import CR_Environment

# -----------------------------
# CREATE ENVIRONMENT
# -----------------------------
env = make_vec_env(CR_Environment, n_envs=1)

# -----------------------------
# DEFINE MODEL
# -----------------------------
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

# -----------------------------
# TRAIN
# -----------------------------
print("Training PPO Agent...")
model.learn(total_timesteps=50000)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("ppo_cr_model")

print("✅ RL training completed!")