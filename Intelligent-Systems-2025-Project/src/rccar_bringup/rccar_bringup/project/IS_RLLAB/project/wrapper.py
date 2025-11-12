import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box


"""
TODO: This skeleton code serves as a guideline;
feel free to modify or replace any part of it.
"""


class RCCarEnvTrainWrapper(gym.Wrapper):
    def __init__(self, base_env, max_steer, min_speed, max_speed, time_limit):
        super().__init__(base_env)
        self.base_env = base_env
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.time_limit = time_limit

        # perform one reset to infer scan size and set spaces
        _, _, scan_sample = self.base_env.reset()[0]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=scan_sample.shape, dtype=np.float32)

        low = np.array([-self.max_steer, self.min_speed], dtype=np.float32)
        high = np.array([self.max_steer, self.max_speed], dtype=np.float32)
        self.action_space = Box(low=low, high=high, shape=(2,), dtype=np.float32)

        self.elapsed_steps = 0

        """
        TODO: Freely define attributes, methods, etc.
        """
        pass

    def reset(self, **kwargs):
        # init waypoint tracking
        self.elapsed_steps = 0

        obs, info = self.base_env.reset(**kwargs)
        _, _, scan = obs

        """
        TODO: Freely reset attributes, etc.
        """

        return scan, info

    def step(self, action):
        steer = np.clip(action[0], -self.max_steer, self.max_steer)
        speed = np.clip(action[1], self.min_speed, self.max_speed)
        wrapped_action = np.array([[steer, speed]], dtype=np.float32)

        obs, _, terminate, truncated, info = self.base_env.step(wrapped_action)
        _, _, scan = obs

        """
        TODO: Freely define reward
        """
        reward = 1.0
        # HINT: waypoint = int(info.get("waypoint", 0))

        self.elapsed_steps += 1

        return scan, reward, terminate, truncated, info
