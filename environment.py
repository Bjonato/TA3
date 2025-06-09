import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """A simple trading environment for OHLCV data."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window_size: int = 10, commission: float = 0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.commission = commission
        self.current_step = window_size
        self.position = 0  # 1 long, -1 short, 0 flat
        self.cost_basis = 0.0
        self.total_profit = 0.0

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, df.shape[1]),
            dtype=np.float32,
        )
        # Actions: 0 hold, 1 buy, 2 sell
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0
        self.cost_basis = 0.0
        self.total_profit = 0.0
        return self._get_observation(), {}

    def step(self, action):
        done = False
        reward = 0.0
        price = self.df.loc[self.current_step, "close"]
        if action == 1 and self.position <= 0:  # buy
            self.position = 1
            self.cost_basis = price * (1 + self.commission)
        elif action == 2 and self.position >= 0:  # sell
            if self.position == 1:
                reward = (price * (1 - self.commission) - self.cost_basis)
                self.total_profit += reward
            self.position = -1
            self.cost_basis = price * (1 - self.commission)
        # hold or invalid actions yield zero reward

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_observation()
        info = {"profit": self.total_profit}
        return obs, reward, done, False, info

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        return window.values.astype(np.float32)

    def render(self):
        pass
