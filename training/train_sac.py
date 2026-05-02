from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from cr_env import CR_Environment
import numpy as np

# -----------------------------
# CUSTOM CALLBACK (REWARD LOGGER)
# -----------------------------
class RewardLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # rewards from vectorized env
        if "rewards" in self.locals:
            self.rewards.extend(self.locals["rewards"])

        # print every 1000 steps
        if len(self.rewards) >= 1000 and self.num_timesteps % 1000 == 0:
            avg_reward = np.mean(self.rewards[-1000:])
            print(f"Step {self.num_timesteps} | Avg Reward: {avg_reward:.3f}")

        return True


# -----------------------------
# ENV
# -----------------------------
env = make_vec_env(CR_Environment, n_envs=1)

# -----------------------------
# MODEL
# -----------------------------
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=0,  # we handle logging ourselves
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=50000,
    gamma=0.99,
    tau=0.005,
)

# -----------------------------
# TRAIN
# -----------------------------
print("Training SAC Agent...")

callback = RewardLogger()

model.learn(
    total_timesteps=50000,
    callback=callback
)

# -----------------------------
# SAVE
# -----------------------------
model.save("sac_cr_model")

print("✅ SAC training completed!")