# models/mlp/value_net.py

import torch
from torch import nn

import numpy as np

from drone_explorer.models.base.network import Net
from drone_explorer.utils.utility import layer_init


class ValueNet(Net):
    """Setup Value Network (Critic) optimizer"""

    def __init__(self, in_dim, out_dim) -> None:
        super(ValueNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(in_dim, 64))
        self.layer2 = layer_init(nn.Linear(64, 64))
        self.layer3 = layer_init(nn.Linear(64, out_dim), std=1.0)
        self.relu = nn.ReLU()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = self.relu(self.layer1(obs))
        x = self.relu(self.layer2(x))
        out = self.layer3(x)  # head has linear activation
        return out

    def loss(self, values, returns):
        """ Objective function defined by mean-squared error.
            ValueNet is approximated via regression.
            Regression target y(t) is defined by Bellman equation or G(t) sample return
        """
        # return 0.5 * ((returns - values)**2).mean() # MSE loss
        return nn.MSELoss()(values, returns)