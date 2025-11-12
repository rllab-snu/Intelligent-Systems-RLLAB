import os
import time
from typing import Any, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from gymnasium.vector import VectorEnv

from tqdm import tqdm

from .network import ActorCritic


"""
TODO: This skeleton code serves as a guideline;
feel free to modify or replace any part of it.
"""


class PPO:
    def __init__(
        self,
        vec_env: VectorEnv,
        ent_coef: float,
        learning_rate: float,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        checkpoint_freq: int = int(2e5),
        checkpoint_prefix: Optional[str] = None,
        writer=None,
        use_wandb=False,
        # TODO: Freely change parameters
        obs_dim: int = 720,
        act_dim: int = 2,
        action_high: float = None,
        action_low: float = None,
    ):
        self.env = vec_env
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ent_coef = ent_coef
        self.lr = learning_rate
        self.seed = seed
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_prefix = checkpoint_prefix
        self.writer = writer
        self.use_wandb = use_wandb

        """
        TODO: Freely define attributes, methods, etc.
        """

        if self.use_wandb:
            import wandb

            self.wandb = wandb

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_envs = 1 if self.env is None else self.env.num_envs

        self.policy = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.total_steps = 0

        self.kwargs = {
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "seed": seed,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "checkpoint_freq": checkpoint_freq,
            "checkpoint_prefix": checkpoint_prefix,
            "writer": writer,
            "use_wandb": use_wandb,
            # TODO: Freely change parameters
            "action_high": action_high,
            "action_low": action_low,
        }

    def _select_action(self, obs_tensor: torch.Tensor):
        # You may use this method, but it's not necessary.
        with torch.no_grad():
            mu, value, logstd = self.policy(obs_tensor)

            std = logstd.exp()
            dist = Normal(mu, std)

            action = dist.sample()
            logprob = dist.log_prob(action)
            logprob = logprob.sum(-1)

        return action, logprob, value

    def _evaluate_actions(self, obs_tensor: torch.Tensor, actions: torch.Tensor):
        # You may use this method, but it's not necessary.
        mu, value, logstd = self.policy(obs_tensor)

        std = logstd.exp()
        dist = Normal(mu, std)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return logprob, entropy, value

    def learn(self, total_timesteps: int):
        obs, _ = self.env.reset()
        obs = np.array(obs)
        if obs.ndim == 1:
            obs = np.expand_dims(obs, 0)

        """
        TODO: train the agent using PPO algorithm
        (e.g.: collect rollouts into a buffer, compute GAE and losses, 
        perform minibatch updates, optimize the networks... etc.)
        """

        start_time = time.time()
        with tqdm(total=total_timesteps, ncols=100) as pbar:
            while self.total_steps < total_timesteps:
                old_total_steps = self.total_steps

                num_steps = self.n_steps
                # TODO: init rollout

                for step in range(num_steps):
                    # TODO: do rollout

                    # if "episode" in infos:
                    #     env_dones = infos.get("_episode", [])
                    #     for i, done in enumerate(env_dones):
                    #         if done:
                    #             self._log("charts/episode_return", infos["episode"]["r"][i], self.total_steps)
                    #             self._log("charts/episode_length", infos["episode"]["l"][i], self.total_steps)

                    # obs = np.array(next_obs)
                    # self.total_steps += self.num_envs
                    # if self.total_steps >= total_timesteps:
                    #     break

                    pass

                # TODO: calculate the advatages and returns 

                b_v_loss, b_p_loss, b_entropy = [], [], []  # for logging

                # TODO: update the model with PPO losses, calculated from the mini-batches

                self._log("losses/value_loss", np.mean(b_v_loss), self.total_steps)
                self._log("losses/policy_loss", np.mean(b_p_loss), self.total_steps)
                self._log("losses/entropy", np.mean(b_entropy), self.total_steps)
                self._log("charts/SPS", int(self.total_steps / (time.time() - start_time)), self.total_steps)

                if (old_total_steps // self.checkpoint_freq) < (self.total_steps // self.checkpoint_freq):
                    save_step = (self.total_steps // self.checkpoint_freq) * self.checkpoint_freq
                    self.save(f"{self.checkpoint_prefix}_{save_step}.pt")

                pbar.update(self.total_steps - old_total_steps)

        print("Learning finished")

    def save(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "kwargs": self.kwargs,
            },
            model_path,
        )
        print(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: str, vec_env: VectorEnv = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(model_path, map_location=device)

        kwargs = checkpoint.get("kwargs", {})

        model = cls(vec_env=vec_env, device=device, **kwargs)

        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.total_steps = checkpoint.get("total_steps", 0)

        print(f"Loaded model from {model_path}")
        return model

    def predict(self, raw_obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_arr = np.asarray(raw_obs)
        if obs_arr.ndim == 1:
            obs_arr = np.expand_dims(obs_arr, 0)

        obs_tensor = self._process_obs(obs_arr)
        with torch.no_grad():
            mu, _, logstd = self.policy(obs_tensor)

            std = logstd.exp()
            if deterministic:
                actions = mu
            else:
                dist = Normal(mu, std)
                actions = dist.sample()

            return self._process_act(actions)

    def _process_obs(self, scan_array: np.ndarray) -> torch.Tensor:
        """
        TODO: preprocess observations for model input
        """
        obs_tensor = torch.tensor(scan_array)
        return obs_tensor

    def _process_act(self, action_tensor: torch.Tensor) -> np.ndarray:
        """
        TODO: postprocess actions for model output
        """
        action_array = action_tensor.detach().cpu().numpy()
        return action_array

    def _log(self, tag: str, value: float, step: int):
        if self.use_wandb:
            self.wandb.log({tag: value, "global_step": step})
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
