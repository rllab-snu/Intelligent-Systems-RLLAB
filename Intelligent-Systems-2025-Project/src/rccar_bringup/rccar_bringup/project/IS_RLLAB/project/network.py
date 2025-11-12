from typing import Tuple

import numpy as np

import torch
import torch.nn as nn


"""
TODO: This skeleton code serves as a guideline;
feel free to modify or replace any part of it.
"""


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        """
        TODO: Freely define model layers
        """
        pass

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO: Freely define model outputs
        """
        mu = None
        value = None
        logstd = None

        return mu, value, logstd
