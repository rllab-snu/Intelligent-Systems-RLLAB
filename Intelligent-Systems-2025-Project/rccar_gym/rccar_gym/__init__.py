import gymnasium as gym

gym.register(
    id="rccar-v0",
    entry_point="rccar_gym.envs:RCCarEnv",
)
