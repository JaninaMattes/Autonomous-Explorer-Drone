from collections import deque
import time
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
from distutils.util import strtobool
import numpy as np
from datetime import datetime
import os
import argparse

from stable_baselines3 import PPO
# gym environment
import gym

# logging python
import logging
import sys

# monitoring/logging ML
import wandb
from wrapper.stats_logger import StatsPlotter
from wrapper.stats_logger import CSVWriter

# hyperparameter tuning
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

# Paths and other constants
MODEL_PATH = './models/'
LOG_PATH = './log/'
VIDEO_PATH = './video/'
RESULTS_PATH = './results/'

# get current date and time
CURR_DATE = datetime.today().strftime('%Y-%m-%d')
CURR_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")


####################
####### TODO #######
####################

# Hint: Please if working on it mark a todo as (done) if done
# 1) Check current implementation against article: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# 3) Check calculation of advantage and GAE

####################
####################


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

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
        out = self.layer3(x) # head has linear activation
        return out
    
    def loss(self, values, returns):
        """ Objective function defined by mean-squared error.
            ValueNet is approximated via regression.
            Regression target y(t) is defined by Bellman equation or G(t) sample return
        """
        # return 0.5 * ((returns - values)**2).mean() # MSE loss
        return nn.MSELoss()(values, returns)

class PolicyNet(Net):
    """Setup Policy Network (Actor)"""
    def __init__(self, in_dim, out_dim) -> None:
        super(PolicyNet, self).__init__()
        self.layer1 = layer_init(nn.Linear(in_dim, 64))
        self.layer2 = layer_init(nn.Linear(64, 64))
        self.layer3 = layer_init(nn.Linear(64, out_dim), std=0.01)
        self.relu = nn.ReLU()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = self.relu(self.layer1(obs))
        x = self.relu(self.layer2(x))
        out = self.layer3(x) # head has linear activation (continuous space)
        return out
    
    def loss(self, advantages, batch_log_probs, curr_log_probs, clip_eps=0.2):
        """ Make the clipped surrogate objective function to compute policy loss.
                - The ratio is clipped to be close to 1. 
                - The clipping ensures that the update will not be too large so that training is more stable.
                - The minimum is taken, so that the gradient will pull π_new towards π_OLD 
                  if the ratio is not between 1-ϵ and 1+ϵ.
        """
        # ratio between pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        ratio = torch.exp(curr_log_probs - batch_log_probs)
        clip_1 = ratio * advantages
        clip_2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = (-torch.min(clip_1, clip_2)) # negative as Adam mins loss, but we want to max it
        # calc clip frac
        self.clip_fraction = (abs((ratio - 1.0)) > clip_eps).to(torch.float).mean()
        return policy_loss.mean() # return mean


####################
####################

class PPO_PolicyGradient_V1:

    def __init__(self, 
        env, 
        in_dim, 
        out_dim,
        total_training_steps,
        batch_size,
        n_rollout_steps,
        n_optepochs=5,
        lr_p=1e-3,
        lr_v=1e-3,
        gae_lambda=0.95,
        gamma=0.99,
        epsilon=0.22,
        adam_eps=1e-5,
        momentum=0.9,
        adam=True,
        render_steps=10,
        render_video=False,
        save_model=10,
        csv_writer=None,
        stats_plotter=None,
        log_video=False,
        device='cpu') -> None:
        
        # hyperparams
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_training_steps = total_training_steps
        self.batch_size = batch_size
        self.n_rollout_steps = n_rollout_steps
        self.n_optepochs = n_optepochs
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.gamma = gamma
        self.epsilon = epsilon
        self.adam_eps = adam_eps
        self.gae_lambda = gae_lambda
        self.momentum = momentum
        self.adam = adam

        # environment
        self.env = env
        self.render_steps = render_steps
        self.render_video = render_video
        self.save_model = save_model
        self.device = device

        # track video of gym
        self.log_video = log_video

        # keep track of rewards per episode
        self.ep_returns = deque(maxlen=batch_size)
        self.csv_writer = csv_writer
        self.stats_plotter = stats_plotter
        self.stats_data = {
            'experiment': [], 
            'timestep': [],
            'mean episodic runtime': [],
            'mean episodic length': [],
            'eval episodes': [],
            'mean episodic returns': [],
            'min episodic returns': [],
            'max episodic returns': [],
            'std episodic returns': [],
            'episodes': []
            }

        # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor) - (policy-based method) "How the agent behaves"
        self.value_net = ValueNet(self.in_dim, 1) # Setup Value Network (Critic) -  (value-based method) "How good the action taken is."

        # add optimizer for actor and critic
        if self.adam:
            self.policy_net_optim = Adam(self.policy_net.parameters(), lr=self.lr_p, eps=self.adam_eps) # Setup Policy Network (Actor) optimizer
            self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr_v, eps=self.adam_eps)  # Setup Value Network (Critic) optimizer  
        else:
            self.policy_net_optim = SGD(self.policy_net.parameters(), lr=self.lr_p, momentum=self.momentum)
            self.value_net_optim = SGD(self.value_net.parameters(), lr=self.lr_v, momentum=self.momentum)

class PPO_PolicyGradient_V2:
    """ Proximal Policy Optimization algorithm (PPO) (clip version)
        
        Paper: https://arxiv.org/abs/1707.06347
        Stable Baseline: https://github.com/hill-a/stable-baselines
        
        Proximal Policy Optimization (PPO) is an online policy gradient method.
        As an online policy method it updates the policy and then discards the experience (no replay buffer).
        Thus the agent does well in environments with dense reward signals.
        The clipped objective function in PPO allows to keep the policy close to the policy 
        that was used to sample the data resulting in a more stable training.

        :param env: The environment to learn from (if registered in Gym)
        :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
        :param batch_size: Minibatch size of collected experiences
        :param n_optepochs: Number of epoch when optimizing the surrogate loss
        :param gamma: Discount factor
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param epsilon: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
        :param normalize_advantage: Whether to normalize or not the advantage
        :param normalize_returns: Whether to normalize or not the return
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    """
    # Further reading
    # PPO experiments: https://nn.labml.ai/rl/ppo/experiment.html
    #                  https://nn.labml.ai/rl/ppo/index.html
    # PPO explained:   https://huggingface.co/blog/deep-rl-ppo

    def __init__(self, 
        env, 
        in_dim, 
        out_dim,
        total_training_steps,
        batch_size,
        n_rollout_steps,
        n_optepochs=5,
        lr_p=1e-3,
        lr_v=1e-3,
        gae_lambda=0.95,
        gamma=0.99,
        epsilon=0.22,
        adam_eps=1e-5,
        momentum=0.9,
        adam=True,
        render_steps=10,
        render_video=False,
        save_model=10,
        csv_writer=None,
        stats_plotter=None,
        log_video=False,
        device='cpu',
        exp_path='./log/',
        exp_name='PPO_V2_experiment',
        normalize_adv=False,
        normalize_ret=False) -> None:
        
        # hyperparams
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_training_steps = total_training_steps
        self.batch_size = batch_size
        self.n_rollout_steps = n_rollout_steps
        self.n_optepochs = n_optepochs
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.gamma = gamma
        self.epsilon = epsilon
        self.adam_eps = adam_eps
        self.gae_lambda = gae_lambda
        self.momentum = momentum
        self.adam = adam

        # environment
        self.env = env
        self.render_steps = render_steps
        self.render_video = render_video
        self.save_model = save_model
        self.device = device
        self.normalize_advantage = normalize_adv
        self.normalize_return = normalize_ret

        # keep track of information
        self.exp_path = exp_path
        self.exp_name = exp_name
        # track video of gym
        self.log_video = log_video

        # keep track of rewards per episode
        self.ep_returns = deque(maxlen=batch_size)
        self.csv_writer = csv_writer
        self.stats_plotter = stats_plotter
        self.stats_data = {
            'experiment': [], 
            'timestep': [],
            'mean episodic runtime': [],
            'mean episodic length': [],
            'eval episodes': [],
            'mean episodic returns': [],
            'min episodic returns': [],
            'max episodic returns': [],
            'std episodic returns': [],
            'episodes': [],
            }

        # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor) - (policy-based method) "How the agent behaves"
        self.value_net = ValueNet(self.in_dim, 1) # Setup Value Network (Critic) -  (value-based method) "How good the action taken is."

        # add optimizer for actor and critic
        if self.adam:
            self.policy_net_optim = Adam(self.policy_net.parameters(), lr=self.lr_p, eps=self.adam_eps) # Setup Policy Network (Actor) optimizer
            self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr_v, eps=self.adam_eps)  # Setup Value Network (Critic) optimizer  
        else:
            self.policy_net_optim = SGD(self.policy_net.parameters(), lr=self.lr_p, momentum=self.momentum)
            self.value_net_optim = SGD(self.value_net.parameters(), lr=self.lr_v, momentum=self.momentum)

    def get_continuous_policy(self, obs):
        """Make function to compute action distribution in continuous action space."""
        # Multivariate Normal Distribution Lecture 15.7 (Andrew Ng) https://www.youtube.com/watch?v=JjB58InuTqM
        # fixes the detection of outliers, allows to capture correlation between features
        # https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
        # 1) Use Normal distribution for continuous space
        action_prob = self.policy_net(obs) # query Policy Network (Actor) for mean action
        cov_matrix = torch.diag(torch.full(size=(self.out_dim,), fill_value=0.5))
        return MultivariateNormal(action_prob, covariance_matrix=cov_matrix)

    def get_action(self, dist):
        """Make action selection function (outputs actions, sampled from policy)."""
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def get_random_action(self, dist):
        """Make random action selection."""
        action = self.env.action_space.sample()
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def get_values(self, obs, actions):
        """Make value selection function (outputs values for obs in a batch)."""
        values = self.value_net(obs).squeeze()
        dist = self.get_continuous_policy(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def get_value(self, obs):
        return self.value_net(obs).squeeze()

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)"""
        action_dist = self.get_continuous_policy(obs) 
        action, log_prob, entropy = self.get_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy(), entropy.detach().numpy()

    def random_step(self, obs):
        action_dist = self.get_continuous_policy(obs) 
        action, log_prob, entropy = self.get_random_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy(), entropy.detach().numpy()

    def advantage_estimate(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ Calculating advantage estimate using TD error (Temporal Difference Error).
            TD Error can be used as an estimator for Advantage function,
            - bias-variance: TD has low variance, but IS biased
            - dones: only get reward at end of episode, not disounted next state value
        """
        # Step 4: Calculate returns
        advantages = []
        cum_returns = []
        for rewards in reversed(episode_rewards):  # reversed order
            for reward in reversed(rewards):
                cum_returns.insert(0, reward) # reverse it again
        cum_returns = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        #  A(s,a) = r - V(s_t)
        advantages = cum_returns - values
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns

    def advantage_reinforce(self, episode_rewards, normalized_adv=False, normalized_ret=False):
        """ Advantage Reinforce 
            A(s_t, a_t) = G(t)
            - G(t) = total disounted reward
            - Discounted return: G(t) = R(t) + gamma * R(t-1)
        """
        # Returns: https://gongybable.medium.com/reinforcement-learning-introduction-609040c8be36
        # Example Reinforce: https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
        
        # Step 4: Calculate returns
        # G(t) is the total disounted reward
        # return value: G(t) = R(t) + gamma * R(t-1)
        cum_returns = []
        for rewards in reversed(episode_rewards): # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                # R + discount * estimated return from the next step taking action a'
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward) # reverse it again
        cum_returns = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        # normalize for more stability
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        # A(s,a) = G(t)
        advantages = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        # normalize for more stability
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns

    def advantage_actor_critic(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ Advantage Actor-Critic
            Discounted return: G(t) = R(t) + gamma * R(t-1)
            Advantage: delta = G(t) - V(s_t)
        """
        # Cumulative rewards: https://gongybable.medium.com/reinforcement-learning-introduction-609040c8be36
        # Step 4: Calculate returns
        # G(t) is the total disounted reward
        # return value: G(t) = R(t) + gamma * R(t-1)
        cum_returns = []
        for rewards in reversed(episode_rewards): # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                # R + discount * estimated return from the next step taking action a'
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward) # reverse it again
        cum_returns = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        # normalize returns
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        # delta = G(t) - V(s_t)
        advantages = cum_returns - values 
        # normalize advantage for more stability
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns

    def advantage_TD_actor_critic(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ Advantage TD Actor-Critic 
            TD Error = δ_t = r_t + γ * V(s_t+1) − V(s_t)
            TD Error is used as an estimator for the advantage function
            A(s,a) = r_t + (gamma * V(s_t+1)) - V(s_t)
        """
        # Step 4: Calculate returns
        advantages = []
        cum_returns = []
        for rewards in reversed(episode_rewards):  # reversed order
            for reward in reversed(rewards):
                cum_returns.insert(0, reward) # reverse it again
        cum_returns = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        # TD error: A(s,a) = r + (gamma * V(s_t+1)) - V(s_t)
        last_values = values[-1]
        for i in reversed(range(len(cum_returns))):
            # TD residual of V with discount gamma
            # δ_t = r_t + γ * V(s_t+1) − V(s_t)
            delta = cum_returns[i] + (self.gamma * last_values) - values[i]
            advantages.insert(0, delta) # reverse it again
            last_values = values[i]
        advantages = torch.tensor(np.array(advantages), device=self.device, dtype=torch.float)
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns

    def generalized_advantage_estimate(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ The Generalized Advanatage Estimate
            δ_t = r_t + γ * V(s_t+1) − V(s_t)
            A_t = δ_t + γ * λ * A(t+1)
                - GAE allows to balance bias and variance through a weighted average of A_t
                - gamma (dicount factor): allows reduce variance by downweighting rewards that correspond to delayed effects
        """
        advantages = []
        cum_returns = []#
        # Step 4: Calculate returns
        for rewards in reversed(episode_rewards):  # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                # R + discount * estimated return from the next step taking action a'
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward) # reverse it again
        cum_returns = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        # δ_t = r_t + γ * V(s_t+1) − V(s_t)
        # A_t = δ_t + γ * λ * A(t+1)
        prev_advantage = 0
        last_values = values[-1]
        for i in reversed(range(len(cum_returns))):
            # TD residual of V with discount gamma
            # δ_t = r_t + γ * V(s_t+1) − V(s_t)
            delta = cum_returns[i] + (self.gamma * last_values) - values[i]
            # discounted sum of Bellman residual term
            # A_t = δ_t + γ * λ * A(t+1)
            prev_advantage = delta + (self.gamma * self.gae_lambda * prev_advantage)
            advantages.insert(0, prev_advantage) # reverse it again
            last_values = values[i]
        advantages = torch.tensor(np.array(advantages), device=self.device, dtype=torch.float)
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns


    def generalized_advantage_estimate_2(self, obs, next_obs, episode_rewards, dones, normalized_adv=False, normalized_ret=False):
        """ Generalized Advantage Estimate calculation
            - GAE defines advantage as a weighted average of A_t
            - advantage measures if an action is better or worse than the policy's default behavior
            - want to find the maximum Advantage representing the benefit of choosing a specific action
        """
        # general advantage estimage paper: https://arxiv.org/pdf/1506.02438.pdf
        # general advantage estimage other: https://nn.labml.ai/rl/ppo/gae.html

        s_values = self.get_value(obs).detach().numpy()
        ns_values = self.get_value(next_obs).detach().numpy()
        advantages = []
        returns = []

        # STEP 4: Calculate cummulated reward
        for rewards in reversed(episode_rewards):
            prev_advantage = 0
            returns_current = ns_values[-1]  # V(s_t+1)
            for i in reversed(range(len(rewards))):
                # STEP 5: compute advantage estimates A_t at step t
                mask = (1.0 - dones[i])
                gamma = self.gamma * mask
                td_target = rewards[i] + (gamma * ns_values[i])
                td_error = td_target - s_values[i]
                # A_t = δ_t + γ * λ * A(t+1)
                prev_advantage = td_error + gamma * self.gae_lambda * prev_advantage
                returns_current = rewards[i] + gamma * returns_current
                # reverse it again
                returns.insert(0, returns_current)
                advantages.insert(0, prev_advantage)
        advantages = np.array(advantages)
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        advantages = torch.tensor(np.array(advantages), device=self.device, dtype=torch.float)
        returns = torch.tensor(np.array(returns), device=self.device, dtype=torch.float)
        return advantages, returns

    def normalize_adv(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def normalize_ret(self, returns):
        eps = np.finfo(np.float32).eps.item()
        return (returns - returns.mean()) / (returns.std() + eps)

    def finish_episode(self):
        pass 

    def collect_rollout(self, n_steps=1):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)"""
        
        t_step, rewards = 0, []

        # log time
        episode_time = []

        # collect trajectories
        episode_obs = []
        episode_nextobs = []
        episode_actions = []
        episode_action_probs = []
        episode_dones = []
        episode_rewards = []
        episode_lens = []

        # Run Monte Carlo simulation for n timesteps per batch
        logging.info(f"Collecting trajectories in batch...")
        while t_step < n_steps:
            
            # rewards collected
            rewards, done = [], False 
            obs = self.env.reset()

            # measure time elapsed for one episode
            # torch.cuda.synchronize()
            start_epoch = time.time()

            # Run episode for a fixed amount of timesteps
            # to keep rollout size fixed and episodes independent
            for t_batch in range(0, self.batch_size):
                # render gym envs
                if self.render_video and t_batch % self.render_steps == 0:
                    self.env.render()
                
                t_step += 1 

                # action logic 
                # sampled via policy which defines behavioral strategy of an agent
                action, log_probability, _ = self.step(obs)

                # Perform action logic at random 
                # action, log_probability, _ = self.random_step(obs)
                        
                # STEP 3: collecting set of trajectories D_k by running action 
                # that was sampled from policy in environment
                __obs, reward, done, truncated = self.env.step(action)
                
                # collection of trajectories in batches
                episode_obs.append(obs)
                episode_nextobs.append(__obs)
                episode_actions.append(action)
                episode_action_probs.append(log_probability)
                rewards.append(reward)
                episode_dones.append(done)
                    
                obs = __obs

                # break out of loop if episode is terminated
                if done or truncated:
                    break
            
            # stop time per episode
            # Waits for everything to finish running
            # torch.cuda.synchronize()
            end_epoch = time.time()
            time_elapsed = end_epoch - start_epoch
            episode_time.append(time_elapsed)

            episode_lens.append(t_batch + 1) # as we started at 0
            episode_rewards.append(rewards)

        # convert trajectories to torch tensors
        obs = torch.tensor(np.array(episode_obs), device=self.device, dtype=torch.float)
        next_obs = torch.tensor(np.array(episode_nextobs), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(episode_actions), device=self.device, dtype=torch.float)
        action_log_probs = torch.tensor(np.array(episode_action_probs), device=self.device, dtype=torch.float)
        dones = torch.tensor(np.array(episode_dones), device=self.device, dtype=torch.float)

        return obs, next_obs, actions, action_log_probs, dones, episode_rewards, episode_lens, np.array(episode_time)
                

    def train(self, values, returns, advantages, batch_log_probs, curr_log_probs, epsilon):
        """Calculate loss and update weights of both networks."""
        logging.info("Updating network parameter...")
        # loss of the policy network
        self.policy_net_optim.zero_grad() # reset optimizer
        policy_loss = self.policy_net.loss(advantages, batch_log_probs, curr_log_probs, epsilon)
        policy_loss.backward() # backpropagation
        self.policy_net_optim.step() # single optimization step (updates parameter)

        # loss of the value network
        self.value_net_optim.zero_grad() # reset optimizer
        value_loss = self.value_net.loss(values, returns)
        value_loss.backward()
        self.value_net_optim.step()

        return policy_loss, value_loss

    def learn(self):
        """"""
        training_steps = 0
        done_so_far = 0
        while training_steps < self.total_training_steps:
            policy_losses, value_losses = [], []

            # Collect data over one episode
            # Episode = recording of actions and states that an agent performed from a start state to an end state
            # STEP 3: simulate and collect trajectories --> the following values are all per batch over one episode
            obs, next_obs, actions, batch_log_probs, dones, rewards, ep_lens, ep_time = self.collect_rollout(n_steps=self.n_rollout_steps)

            # experiences simulated so far
            training_steps += np.sum(ep_lens)

            # STEP 4-5: Calculate cummulated reward and advantage at timestep t_step
            values, _ , _ = self.get_values(obs, actions)
            # Calculate advantage function
            advantages, cum_returns = self.advantage_reinforce(rewards, normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.advantage_actor_critic(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.advantage_TD_actor_critic(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.generalized_advantage_estimate(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            
            # update network params 
            for _ in range(self.n_optepochs):
                # STEP 6-7: calculate loss and update weights
                values, curr_log_probs, _ = self.get_values(obs, actions)
                policy_loss, value_loss = self.train(values, cum_returns, advantages, batch_log_probs, curr_log_probs, self.epsilon)
                
                policy_losses.append(policy_loss.detach().numpy())
                value_losses.append(value_loss.detach().numpy())

            # log all statistical values to CSV
            self.log_stats(policy_losses, value_losses, rewards, ep_lens, training_steps, ep_time, done_so_far, exp_name=self.exp_name)
            
            # increment for each iteration
            done_so_far += 1

            # store model with checkpoints
            if training_steps % self.save_model == 0:
                env_name = self.env.unwrapped.spec.id
                env_model_path = os.path.join(self.exp_path, 'models')
                policy_net_name = os.path.join(env_model_path, f'{env_name}_policyNet.pth')
                value_net_name = os.path.join(env_model_path, f'{env_name}_valueNet.pth')
                torch.save({
                    'epoch': training_steps,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.policy_net_optim.state_dict(),
                    'loss': policy_loss,
                    }, policy_net_name)
                torch.save({
                    'epoch': training_steps,
                    'model_state_dict': self.value_net.state_dict(),
                    'optimizer_state_dict': self.value_net_optim.state_dict(),
                    'loss': value_loss,
                    }, value_net_name)

                if wandb:
                    wandb.save(policy_net_name)
                    wandb.save(value_net_name)

                # Log to CSV
                if self.csv_writer:
                    self.csv_writer(self.stats_data)
                    for value in self.stats_data.values():
                        del value[:]

        # Finalize and plot stats
        if self.stats_plotter:
            df = self.stats_plotter.read_csv() # read all files in folder
            self.stats_plotter.plot_seaborn_fill(df, x='timestep', y='mean episodic returns', 
                                                y_min='min episodic returns', y_max='max episodic returns',  
                                                title=f'{env_name}', x_label='Timestep', y_label='Mean Episodic Return', 
                                                color='blue', smoothing=6, wandb=wandb, xlim_up=self.total_training_steps)

            # self.stats_plotter.plot_box(df, x='timestep', y='mean episodic runtime', 
            #                             title='title', x_label='Timestep', y_label='Mean Episodic Time', wandb=wandb)
        if wandb:
            # save files in path
            wandb.save(os.path.join(self.exp_path, "*csv"))
            # Save any files starting with "ppo"
            wandb.save(os.path.join(wandb.run.dir, "ppo*"))


    def log_stats(self, p_losses, v_losses, batch_return, episode_lens, training_steps, time, done_so_far, exp_name='experiment'):
        """Calculate stats and log to W&B, CSV, logger """
        if torch.is_tensor(batch_return):
            batch_return = batch_return.detach().numpy()
        # calculate stats
        mean_p_loss = round(np.mean([np.sum(loss) for loss in p_losses]), 6)
        mean_v_loss = round(np.mean([np.sum(loss) for loss in v_losses]), 6)

        # Calculate the stats of an episode
        cum_ret = [np.sum(ep_rews) for ep_rews in batch_return]
        mean_ep_time = round(np.mean(time), 6) 
        mean_ep_len = round(np.mean(episode_lens), 6)

        # statistical values for return
        mean_ep_ret = round(np.mean(cum_ret), 6)
        max_ep_ret = round(np.max(cum_ret), 6)
        min_ep_ret = round(np.min(cum_ret), 6)

        # calculate standard deviation (spred of distribution)
        std_ep_rew = round(np.std(cum_ret), 6)

        # Log stats to CSV file
        self.stats_data['episodes'].append(done_so_far)
        self.stats_data['experiment'].append(exp_name)
        self.stats_data['mean episodic length'].append(mean_ep_len)
        self.stats_data['mean episodic returns'].append(mean_ep_ret)
        self.stats_data['min episodic returns'].append(min_ep_ret)
        self.stats_data['max episodic returns'].append(max_ep_ret)
        self.stats_data['mean episodic runtime'].append(mean_ep_time)
        self.stats_data['std episodic returns'].append(std_ep_rew)
        self.stats_data['eval episodes'].append(len(cum_ret))
        self.stats_data['timestep'].append(training_steps)

        # Monitoring via W&B
        wandb.log({
            'train/timesteps': training_steps,
            'train/mean policy loss': mean_p_loss,
            'train/mean value loss': mean_v_loss,
            'train/mean episode returns': mean_ep_ret,
            'train/min episode returns': min_ep_ret,
            'train/max episode returns': max_ep_ret,
            'train/std episode returns': std_ep_rew,
            'train/mean episode runtime': mean_ep_time,
            'train/mean episode length': mean_ep_len,
            'train/episodes': done_so_far,
        })

        logging.info('\n')
        logging.info(f'------------ Episode: {training_steps} --------------')
        logging.info(f"Mean return:          {mean_ep_ret}")
        logging.info(f"Mean policy loss:     {mean_p_loss}")
        logging.info(f"Mean value loss:      {mean_v_loss}")
        logging.info('--------------------------------------------')
        logging.info('\n')

####################
####################

def arg_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False, help="if toggled, capture video of run")
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, run model in training mode")
    parser.add_argument("--test", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False, help="if toggled, run model in testing mode")
    parser.add_argument("--hyperparam", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, log hyperparameters")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--project-name", type=str, default='OpenAIGym-PPO', help="the name of this project") 
    parser.add_argument("--gym-id", type=str, default="Pendulum-v1", help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000, help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    
    # Parse arguments if they are given
    args = parser.parse_args()
    return args

def make_env(env_id='Pendulum-v1', gym_wrappers=False, seed=42):
    # TODO: Needs to be parallized for parallel simulation
    env = gym.make(env_id)
    
    # gym wrapper
    if gym_wrappers:
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    
    # seed env for reproducability
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def make_vec_env(num_env=1):
    """ Create a vectorized environment for parallelized training."""
    pass

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """ Initialize the hidden layers with orthogonal initialization
        Engstrom, Ilyas, et al., (2020)
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def create_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(path, model, device='cpu'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], map_location=device)
    return model

def simulate_rollout(policy_net, env, render_video=True):
    # Rollout until user kills process
	while True:
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done:
			t += 1

			# Render environment if specified, off by default
			if render_video:
				env.render()

			# Query deterministic action from policy and run it
			action = policy_net(obs).detach().numpy()
			obs, rew, done, _ = env.step(action)

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def _log_summary(ep_len, ep_ret, ep_num):

        # Monitoring via W&B
        wandb.log({
            'test/timesteps': ep_num,
            'test/episode length': ep_len,
            'test/episode returns': ep_ret
        })

		# Print logging statements
        logging.info('\n')
        logging.info(f'------------ Episode: {ep_num} --------------')
        logging.info(f"Episodic Length: {ep_len}")
        logging.info(f"Episodic Return: {ep_ret}")
        logging.info(f"--------------------------------------------")
        logging.info('\n')

def train(env, in_dim, out_dim, total_training_steps, batch_size=512, n_rollout_steps=2048,
          n_optepochs=32, learning_rate_p=1e-4, learning_rate_v=1e-3, gae_lambda=0.95, gamma=0.99, epsilon=0.2,
          adam_epsilon=1e-8, render_steps=10, render_video=False, save_steps=10, csv_writer=None, stats_plotter=None,
          normalize_adv=False, normalize_ret=False, log_video=False, ppo_version='v2', 
          device='cpu', exp_path='./log/', exp_name='PPO-experiment'):
    """Train the policy network (actor) and the value network (critic) with PPO (clip version)"""
    agent = None
    if ppo_version == 'v2':
        agent = PPO_PolicyGradient_V2(
                    env, 
                    in_dim=in_dim, 
                    out_dim=out_dim,
                    total_training_steps=total_training_steps,
                    batch_size=batch_size,
                    n_rollout_steps=n_rollout_steps,
                    n_optepochs=n_optepochs,
                    lr_p=learning_rate_p,
                    lr_v=learning_rate_v,
                    gae_lambda = gae_lambda,
                    gamma=gamma,
                    epsilon=epsilon,
                    adam_eps=adam_epsilon,
                    normalize_adv=normalize_adv,
                    normalize_ret=normalize_ret,
                    render_steps=render_steps,
                    render_video=render_video,
                    save_model=save_steps,
                    csv_writer=csv_writer,
                    stats_plotter=stats_plotter,
                    log_video=log_video,
                    device=device,
                    exp_path=exp_path,
                    exp_name=exp_name)
    else:
        agent = PPO_PolicyGradient_V1(
                    env, 
                    in_dim=in_dim, 
                    out_dim=out_dim,
                    total_training_steps=total_training_steps,
                    batch_size=batch_size,
                    n_rollout_steps=n_rollout_steps,
                    n_optepochs=n_optepochs,
                    lr_p=learning_rate_p,
                    lr_v=learning_rate_v,
                    gae_lambda = gae_lambda,
                    gamma=gamma,
                    epsilon=epsilon,
                    adam_eps=adam_epsilon,
                    render_steps=render_steps,
                    render_video=render_video,
                    save_model=save_steps,
                    csv_writer=csv_writer,
                    stats_plotter=stats_plotter,
                    log_video=log_video,
                    device=device)
    # run training for a total amount of steps
    agent.learn()

def test(path, env, in_dim, out_dim, steps=10_000, render_video=True, log_video=False, device='cpu'):
    """Test the policy network (actor)"""
    # load model and test it
    policy_net = PolicyNet(in_dim, out_dim)
    policy_net = load_model(path, policy_net, device)
    
    for ep_num, (ep_len, ep_ret) in enumerate(simulate_rollout(policy_net, env, render_video)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

        if log_video:
            wandb.log({"test/video": wandb.Video(VIDEO_PATH, caption='episode: '+str(ep_num), fps=4, format="gif"), "step": ep_num})

def hyperparam_tuning(config=None):
    # set config
    with wandb.init(config=config):
        agent = PPO_PolicyGradient_V2(
                env,
                in_dim=config.obs_dim, 
                out_dim=config.act_dim,
                total_training_steps=config.total_training_steps,
                batch_size=config.batch_size,
                n_rollout_steps=config.n_rollout_steps,
                n_optepochs=config.n_optepochs,
                gae_lambda = config.gae_lambda,
                gamma=config.gamma,
                epsilon=config.epsilon,
                adam_eps=config.adam_epsilon,
                lr_p=config.learning_rate_p,
                lr_v=config.learning_rate_v)
    
        # run training for a total amount of steps
        agent.learn()


if __name__ == '__main__':
    
    """ Classic control gym environments 
        Find docu: https://www.gymlibrary.dev/environments/classic_control/
    """
    # parse arguments
    args = arg_parser()

    # check if path exists otherwise create
    if args.video:
        create_path(VIDEO_PATH)
    create_path(LOG_PATH)
    create_path(RESULTS_PATH)
    
    # Hyperparameter
    total_training_steps = 3_000_000     # time steps regarding batches collected and train agent
    batch_size = 512                     # max number of episode samples to be sampled per time step. 
    n_rollout_steps = 2048               # number of batches per episode, or experiences to collect per environment
    n_optepochs = 32                     # Number of epochs per time step to optimize the neural networks
    learning_rate_p = 1e-4               # learning rate for policy network
    learning_rate_v = 1e-3               # learning rate for value network
    gae_lambda = 0.95                    # factor for trade-off of bias vs variance for GAE
    gamma = 0.99                         # discount factor
    adam_epsilon = 1e-8                  # default in the PPO baseline implementation is 1e-5, the pytorch default is 1e-8 - Andrychowicz, et al. (2021)  uses 0.9
    epsilon = 0.2                        # clipping factor
    clip_range_vf = 0.2                  # clipping factor for the value loss function. Depends on reward scaling.
    env_name = 'Pendulum-v1'             # name of OpenAI gym environment other: 'Pendulum-v1' , 'MountainCarContinuous-v0', 'takeoff-aviary-v0'
    env_number = 1                       # number of actors
    seed = 42                            # seed gym, env, torch, numpy 
    normalize_adv = True                # wether to normalize the advantage estimate
    normalize_ret = False                 # wether to normalize the return function
    
    # setup for torch save models and rendering
    render_video = False
    render_steps = 10
    save_steps = 100

    # Configure logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    # seed gym, torch and numpy
    env = make_env(env_name, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # get correct device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # get dimensions of obs (what goes in?)
    # and actions (what goes out?)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    logging.info(f'env observation space: {obs_shape}')
    logging.info(f'env action space: {act_shape}')
    logging.info(f'env action upper bound: {upper_bound}')
    logging.info(f'env action lower bound: {lower_bound}')
    
    obs_dim = obs_shape[0] 
    act_dim = act_shape[0] # 2 at CartPole

    logging.info(f'env observation dim: {obs_dim}')
    logging.info(f'env action dim: {act_dim}')
    
    # upper and lower bound describing the values our obs can take
    logging.info(f'upper bound for env observation: {env.observation_space.high}')
    logging.info(f'lower bound for env observation: {env.observation_space.low}')

    # create folder for project
    exp_name = f"exp_name: {env_name}_{CURR_DATE} {args.exp_name}"
    exp_folder_name = f'{LOG_PATH}exp_{env_name}_{CURR_TIME}'
    create_path(exp_folder_name)
    # create model folder
    model_path = os.path.join(exp_folder_name, 'models')
    create_path(model_path)

    # create CSV writer
    csv_file = os.path.join(exp_folder_name, f'{env_name}_{CURR_TIME}.csv')
    png_file = os.path.join(exp_folder_name, f'{env_name}_{CURR_TIME}.png')
    csv_writer = CSVWriter(csv_file)
    stats_plotter = StatsPlotter(csv_folder_path=exp_folder_name, file_name_and_path=png_file)

    # Monitoring with W&B
    wandb.init(
            project=args.project_name,
            entity='drone-mechanics',
            sync_tensorboard=True,
            config={ # stores hyperparams in job
                'env name': env_name,
                'env number': env_number,
                'total number of steps': total_training_steps,
                'max sampled trajectories': batch_size,
                'batches per episode': n_rollout_steps,
                'number of epochs for update': n_optepochs,
                'input layer size': obs_dim,
                'output layer size': act_dim,
                'observation space': obs_shape,
                'action space': act_shape,
                'action space upper bound': upper_bound,
                'action space lower bound': lower_bound,
                'learning rate (policy net)': learning_rate_p,
                'learning rate (value net)': learning_rate_v,
                'epsilon (adam optimizer)': adam_epsilon,
                'gamma (discount)': gamma,
                'epsilon (clipping)': epsilon,
                'gae lambda (GAE)': gae_lambda,
                'normalize advantage': normalize_adv,
                'normalize return': normalize_ret,
                'seed': seed,
                'experiment path': exp_folder_name,
                'experiment name': args.exp_name
            },
            dir=os.getcwd(),
            name=exp_name, # needs flag --exp-name
            monitor_gym=True,
            save_code=True
        )

    if args.train:
        logging.info('Training model...')
        train(env,
            in_dim=obs_dim, 
            out_dim=act_dim,
            total_training_steps=total_training_steps,
            batch_size=batch_size,
            n_rollout_steps=n_rollout_steps,
            n_optepochs=n_optepochs,
            learning_rate_p=learning_rate_p, 
            learning_rate_v=learning_rate_v,
            gae_lambda = gae_lambda,
            gamma=gamma,
            epsilon=epsilon,
            adam_epsilon=adam_epsilon,
            normalize_adv=normalize_adv,
            normalize_ret=normalize_ret,
            render_steps=render_steps,
            render_video=render_video,
            save_steps=save_steps,
            csv_writer=csv_writer,
            stats_plotter=stats_plotter,
            log_video=args.video,
            device=device,
            exp_path=exp_folder_name,
            exp_name=exp_name)
    
    elif args.test:
        logging.info('Evaluation model...')
        PATH = './models/Pendulum-v1_2023-01-01_policyNet.pth' # TODO: define path
        test(PATH, env, in_dim=obs_dim, out_dim=act_dim, device=device)
    
    elif args.hyperparam:
        # hyperparameter tuning with sweeps

        logging.info('Hyperparameter tuning...')
        # sweep config
        sweep_config = {
            'method': 'bayes'
            }
        metric = {
            'name': 'mean_ep_rews',
            'goal': 'maximize'   
            }
        parameters_dict = {
            'learning_rate_p': {
                'values': [1e-5, 1e-4, 1e-3, 1e-2]
            },
            'learning_rate_v': {
                'values': [1e-5, 1e-4, 1e-3, 1e-2]
            },
            'gamma': {
                'values': [0.95, 0.96, 0.97, 0.98, 0.99]
            },
            'adam_epsilon': {
                'values': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            'epsilon': {
                'values': [0.1, 0.2, 0.25, 0.3, 0.4]
            },
            }

        # set defined parameters
        sweep_config['parameters'] = parameters_dict
        sweep_config['metric'] = metric

        # run sweep with sweep controller
        sweep_id = wandb.sweep(sweep_config, project="ppo-OpenAIGym-hyperparam-tuning")
        wandb.agent(sweep_id, hyperparam_tuning, count=5)

    else:
        assert("Needs training (--train), testing (--test) or hyperparameter tuning (--hyperparam) flag set!")

    #################
    #### Cleanup ####
    #################

    logging.info('### Done ###')

    # cleanup 
    env.close()
    wandb.run.finish() if wandb and wandb.run else None