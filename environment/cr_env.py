import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CR_Environment(gym.Env):

    def __init__(self):
        super(CR_Environment, self).__init__()

        self.num_channels = 4
        self.noise = 0.1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0.1, 0], dtype=np.float32),
            high=np.array([1.0, self.num_channels - 1], dtype=np.float32),
            dtype=np.float32
        )

        self.state = self._sample_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):

        power = float(action[0])
        channel = int(round(action[1]))

        power = np.clip(power, 0.1, 1.0)
        channel = np.clip(channel, 0, self.num_channels - 1)

        h = self.state[channel]
        interference = self.state[8]
        noise = self.state[9]

        sinr = (power * h) / (interference + noise + 1e-6)
        throughput = np.log2(1 + sinr)
        latency = 1 / (sinr + 1e-6)

        SINR_THRESHOLD = 1.0
        
        channel_penalty = 0.0
        if channel == 0:
            channel_penalty = -0.2
            
        power_penalty = 0.0
        if power > 0.95:
            power_penalty = -0.2
         
        penalty = 0.0
        if sinr < SINR_THRESHOLD:
            penalty = -2.0 * (SINR_THRESHOLD - sinr)
            penalty = max(penalty, -3.0)

        reward = (
            2.0 * throughput
            - 0.3 * interference
            - 0.2 * latency
            - 0.1 * power
            + penalty
            + channel_penalty
            + power_penalty
        )

        self.state = self._sample_state()

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def _sample_state(self):

        h = np.random.rayleigh(scale=1.0, size=4)
        interference = np.random.uniform(0.1, 2.0)
        sinr = h / (interference + self.noise + 1e-6)

        return np.array([
            h[0], h[1], h[2], h[3],
            sinr[0], sinr[1], sinr[2], sinr[3],
            interference,
            self.noise
        ], dtype=np.float32)