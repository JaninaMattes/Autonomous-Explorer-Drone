from typing import Tuple
import numpy as np
import torch

from drone_explorer.utils.utility import normalize


def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    next_value: torch.Tensor,
) -> torch.Tensor:
    """
    rewards: (T,)
    dones: (T,)
    """
    returns = torch.zeros_like(rewards)
    running_return = next_value

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return

    return returns


def compute_reinforce(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    norm_adv: bool = False,
    norm_returns: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Advantage Reinforce 

    :param rewards: Description
    :type rewards: torch.Tensor
    :param values: Description
    :type values: torch.Tensor
    :param dones: Description
    :type dones: torch.Tensor
    :param gamma: Description
    :type gamma: float
    :param lam: Description
    :type lam: float
    :param norm_adv: Description
    :type norm_adv: bool
    :param norm_returns: Description
    :type norm_returns: bool
    :return: Description
    :rtype: Tuple[Tensor, Tensor]

    (1) A(s_t, a_t) = G(t)
    (2) Discounted return: G(t) = R(t) + gamma * R(t-1) 

    Resource: 
    https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
    """
    device = rewards.device

    cum_returns = []
    for reward in reversed(rewards):
        discounted_reward = 0
        for r in reversed(reward):
            discounted_reward = reward + (gamma * discounted_reward)
            cum_returns.insert(0, discounted_reward)
    cum_returns = torch.tensor(
        np.array(cum_returns), device=device, dtype=torch.float)
    if norm_returns:
        cum_returns = normalize(cum_returns)

    advantages = torch.tensor(np.array(cum_returns),
                              device=device, dtype=torch.float)
    if norm_adv:
        advantages = normalize(advantages)
    return advantages, cum_returns



def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    norm_adv: bool = False,
    norm_returns: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advanatage Estimate

    :param rewards: Description
    :type rewards: torch.Tensor
    :param values: Description
    :type values: torch.Tensor
    :param dones: Description
    :type dones: torch.Tensor
    :param gamma: Description
    :type gamma: float
    :param lam: Description
    :type lam: float
    :param norm_adv: Description
    :type norm_adv: bool
    :param norm_returns: Description
    :type norm_returns: bool
    :return: Description
    :rtype: Tuple[Tensor, Tensor]

    (1) TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
    (2) GAE: A_t = δ_t + γ * λ * (1 - done) * A_{t+1}

    More info under:  https://nn.labml.ai/rl/ppo/gae.html
    """
    device = rewards.device
    advantages = []
    cum_returns = []

    # TODO: Update to better structure
    # Compute GAE using reverse iteration
    # (1) Calculate returns
    # discounted_reward = reward + discount * estimated return from the next step taking action a'
    for reward in reversed(rewards):
        discounted_r = 0
        for r in reversed(reward):
            discounted_r = r + (gamma * discounted_r)
            cum_returns.insert(0, discounted_r)  # reverse it again

    cum_returns = torch.tensor(
        np.array(cum_returns), device=device, dtype=torch.float)

    # (optional) normalize cummulated returns
    if norm_returns:
        cum_returns = normalize(cum_returns)

    # (2) Compute advantage
    prev_advantage = 0
    last_values = values[-1]
    for i in reversed(range(len(cum_returns))):
        delta = cum_returns[i] + (gamma * last_values) - values[i]
        prev_advantage = delta + \
            (gamma * lam * prev_advantage)

        advantages.insert(0, prev_advantage)  # reverse it again
        last_values = values[i]
    advantages = torch.tensor(
        np.array(advantages), device=device, dtype=torch.float)

    # (optional) normalize advantages
    if norm_adv:
        advantages = normalize(advantages)
    return advantages, cum_returns


def td_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    next_value: torch.Tensor,
) -> torch.Tensor:
    pass


if __name__ == "__main__":
