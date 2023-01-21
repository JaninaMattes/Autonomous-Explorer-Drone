from collections import deque
import time
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
from scipy.ndimage.filters import gaussian_filter1d
from distutils.util import strtobool
import numpy as np
from datetime import datetime
import os
import argparse

from stable_baselines3.common.env_checker import check_env
# gym environment
import gym
from gym.envs.registration import register

# video logging
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

# logging python
import logging
import sys

# monitoring/logging ML
import wandb

# logging
from gym_pybullet_drones.examples.stats_logger import CSVWriter, StatsPlotter

# Paths and other constants
MODEL_PATH = './models/'
LOG_PATH = './log/'
VIDEO_PATH = './video/'
RESULTS_PATH = './results/'

# get current date and time
CURR_DATE = datetime.today().strftime('%Y-%m-%d')
CURR_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

# config logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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
        out = self.layer3(x)  # head has linear activation
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
        out = self.layer3(x)  # head has linear activation (continuous space)
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
        clip_2 = torch.clamp(ratio, 1.0 - clip_eps,
                             1.0 + clip_eps) * advantages
        # negative as Adam mins loss, but we want to max it
        policy_loss = (-torch.min(clip_1, clip_2))
        # calc clip frac
        self.clip_fraction = (abs((ratio - 1.0)) >
                              clip_eps).to(torch.float).mean()
        return policy_loss.mean()  # return mean


####################
####################

class PPO_PolicyGradient:
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
        :param max_trajectory_size: Minibatch size of collected experiences
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
                 total_training_steps=2_000_000,
                 max_trajectory_size=1024,
                 n_rollout_steps=2048,
                 n_optepochs=32,
                 learning_rate_p=1e-4,
                 learning_rate_v=1e-3,
                 gae_lambda=0.95,
                 gamma=0.99,
                 epsilon=0.2,
                 adam_eps=1e-5,
                 momentum=0.9,
                 adam=True,
                 save_model=10,
                 csv_writer=None,
                 stats_plotter=None,
                 log_video=False,
                 log_video_steps=100,
                 render_steps=10,
                 render_video=False,
                 device='cpu',
                 exp_path='./log/',
                 exp_name='PPO_V2_experiment',
                 advantage_type='gae',
                 normalize_adv=False,
                 normalize_ret=False,
                 seed=0) -> None:

        # environment
        self.env = env

        # hyperparams
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_training_steps = total_training_steps
        self.max_trajectory_size = max_trajectory_size
        self.n_rollout_steps = n_rollout_steps
        self.n_optepochs = n_optepochs
        self.learning_rate_p = learning_rate_p
        self.learning_rate_v = learning_rate_v
        self.gamma = gamma
        self.epsilon = epsilon
        self.adam_eps = adam_eps
        self.gae_lambda = gae_lambda
        self.momentum = momentum
        self.adam = adam
        self.advantage_type = advantage_type

        # logging and rendering
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
        self.log_video_steps = log_video_steps

        # keep track of rewards per episode
        self.ep_returns = deque(maxlen=max_trajectory_size)
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
        # Setup Policy Network (Actor) - (policy-based method) "How the agent behaves"
        self.policy_net = PolicyNet(self.in_dim, self.out_dim)
        # Setup Value Network (Critic) -  (value-based method) "How good the action taken is."
        self.value_net = ValueNet(self.in_dim, 1)

        # add optimizer for actor and critic
        if self.adam:
            self.policy_net_optim = Adam(self.policy_net.parameters(
            ), lr=self.learning_rate_p, eps=self.adam_eps)  # Setup Policy Network (Actor) optimizer
            self.value_net_optim = Adam(self.value_net.parameters(
            ), lr=self.learning_rate_v, eps=self.adam_eps)  # Setup Value Network (Critic) optimizer
        else:
            self.policy_net_optim = SGD(
                self.policy_net.parameters(), lr=self.learning_rate_p, momentum=self.momentum)
            self.value_net_optim = SGD(
                self.value_net.parameters(), lr=self.learning_rate_v, momentum=self.momentum)

    def get_continuous_policy(self, obs):
        """Make function to compute action distribution in continuous action space."""
        # Multivariate Normal Distribution Lecture 15.7 (Andrew Ng) https://www.youtube.com/watch?v=JjB58InuTqM
        # fixes the detection of outliers, allows to capture correlation between features
        # https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
        # 1) Use Normal distribution for continuous space
        # query Policy Network (Actor) for mean action
        action_prob = self.policy_net(obs)
        cov_matrix = torch.diag(torch.full(
            size=(self.out_dim,), fill_value=0.5))
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
                cum_returns.insert(0, reward)  # reverse it again
        cum_returns = torch.tensor(
            np.array(cum_returns), device=self.device, dtype=torch.float)
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
        for rewards in reversed(episode_rewards):  # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                # R + discount * estimated return from the next step taking action a'
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward)  # reverse it again
        cum_returns = torch.tensor(
            np.array(cum_returns), device=self.device, dtype=torch.float)
        # normalize for more stability
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        # A(s,a) = G(t)
        advantages = torch.tensor(
            np.array(cum_returns), device=self.device, dtype=torch.float)
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
        for rewards in reversed(episode_rewards):  # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                # R + discount * estimated return from the next step taking action a'
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward)  # reverse it again
        cum_returns = torch.tensor(
            np.array(cum_returns), device=self.device, dtype=torch.float)
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
                cum_returns.insert(0, reward)  # reverse it again
        cum_returns = torch.tensor(
            np.array(cum_returns), device=self.device, dtype=torch.float)
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        # TD error: A(s,a) = r + (gamma * V(s_t+1)) - V(s_t)
        last_values = values[-1]
        for i in reversed(range(len(cum_returns))):
            # TD residual of V with discount gamma
            # δ_t = r_t + γ * V(s_t+1) − V(s_t)
            delta = cum_returns[i] + (self.gamma * last_values) - values[i]
            advantages.insert(0, delta)  # reverse it again
            last_values = values[i]
        advantages = torch.tensor(
            np.array(advantages), device=self.device, dtype=torch.float)
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
        cum_returns = []
        # Step 4: Calculate returns
        for rewards in reversed(episode_rewards):  # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                # R + discount * estimated return from the next step taking action a'
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward)  # reverse it again
        cum_returns = torch.tensor(
            np.array(cum_returns), device=self.device, dtype=torch.float)
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
            prev_advantage = delta + \
                (self.gamma * self.gae_lambda * prev_advantage)
            advantages.insert(0, prev_advantage)  # reverse it again
            last_values = values[i]
        advantages = torch.tensor(
            np.array(advantages), device=self.device, dtype=torch.float)
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
        advantages = torch.tensor(
            np.array(advantages), device=self.device, dtype=torch.float)
        returns = torch.tensor(
            np.array(returns), device=self.device, dtype=torch.float)
        return advantages, returns

    def normalize_adv(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def normalize_ret(self, returns):
        eps = np.finfo(np.float32).eps.item()
        return (returns - returns.mean()) / (returns.std() + eps)

    def finish_episode(self):
        pass

    def collect_rollout(self, n_steps=2048):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)
           A typical rollout length - 2048 t0 4096
        """
        
        t_step, rewards, frames = 0, [], deque(maxlen=24) # 4 fps - 6 sec

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
        logging.info(f"Collecting trajectories for {n_steps} episodes.")
        while t_step < n_steps:
            
            # rewards collected
            rewards, done, frames = [], False, []
            obs = self.env.reset()

            # measure time elapsed for one episode
            # torch.cuda.synchronize()
            start_epoch = time.time()

            # Run episode for a fixed amount of timesteps
            # to keep rollout size fixed and episodes independent
            for t in range(0, self.max_trajectory_size):
                # render gym envs
                if self.render_video and t % self.render_steps == 0:
                    frames.append(self.env.render(mode="rgb_array"))
                
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

            episode_lens.append(t + 1) # as we started at 0
            episode_rewards.append(rewards)

        # convert trajectories to torch tensors
        obs = torch.tensor(np.array(episode_obs), device=self.device, dtype=torch.float)
        next_obs = torch.tensor(np.array(episode_nextobs), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(episode_actions), device=self.device, dtype=torch.float)
        action_log_probs = torch.tensor(np.array(episode_action_probs), device=self.device, dtype=torch.float)
        dones = torch.tensor(np.array(episode_dones), device=self.device, dtype=torch.float)

        return obs, next_obs, actions, action_log_probs, dones, episode_rewards, episode_lens, np.array(episode_time), frames
                

    def train(self, values, returns, advantages, batch_log_probs, curr_log_probs, epsilon):
        """Calculate loss and update weights of both networks."""
        # loss of the policy network
        self.policy_net_optim.zero_grad()  # reset optimizer
        policy_loss = self.policy_net.loss(
            advantages, batch_log_probs, curr_log_probs, epsilon)
        policy_loss.backward()  # backpropagation
        self.policy_net_optim.step()  # single optimization step (updates parameter)

        # loss of the value network
        self.value_net_optim.zero_grad()  # reset optimizer
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
            obs, next_obs, actions, batch_log_probs, dones, rewards, ep_lens, ep_time, frames = self.collect_rollout(n_steps=self.n_rollout_steps)

            # experiences simulated so far
            training_steps += np.sum(ep_lens)

            # STEP 4-5: Calculate cummulated reward and advantage at timestep t_step
            values, _ , _ = self.get_values(obs, actions)
            # Calculate advantage function
            # advantages, cum_returns = self.advantage_reinforce(rewards, normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.advantage_actor_critic(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.advantage_TD_actor_critic(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            advantages, cum_returns = self.generalized_advantage_estimate(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            
            # update network params 
            logging.info(f"Updating network parameter for {self.n_optepochs} epochs.")
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

                # Log to video
                if self.render_video and done_so_far % self.video_log_steps == 0:
                    filename='pendulum_v1.gif'
                    self.save_frames_as_gif(frames, self.exp_path, filename)
                    if wandb:
                        wandb.log({
                            "train/video": wandb.Video(os.path.join(self.exp_path, filename), 
                            caption='episode: '+str(done_so_far), 
                            fps=4, format="gif"), "step": done_so_far
                            })

        # Finalize and plot stats
        if self.stats_plotter:
            try: 
                df = self.stats_plotter.read_csv() # read all files in folder
                self.stats_plotter.plot_seaborn_fill(df, 
                                                    x='timestep', y='mean episodic returns', 
                                                    y_min='min episodic returns', y_max='max episodic returns',  
                                                    title=f'{env_name}', 
                                                    x_label='Episode', 
                                                    y_label='Mean Episodic Return',
                                                    smoothing=2, 
                                                    wandb=wandb)
            except:
                logging.warn('Plotting unsuccessfull...')
        if wandb:
            # save files in path
            wandb.save(os.path.join(self.exp_path, "*csv"))
            # Save any files starting with "ppo"
            wandb.save(os.path.join(wandb.run.dir, "ppo*"))

    def save_frames_as_gif(self, frames, path='./', filename='pendulum_v1.gif'):

        # Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0,
                   frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(frames), interval=50)
        video_path = os.path.join(path, filename)
        anim.save(video_path, writer='imagemagick', fps=60)


    def log_stats(self, p_losses, v_losses, batch_return, episode_lens, training_steps, time, done_so_far, exp_name='experiment', smoothing=None):
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

        # use gaussian smoothing
        if smoothing:
            max_ep_ret = gaussian_filter1d(cum_ret.to_numpy(), sigma=smoothing)

        # statistical values for return
        mean_ep_ret = round(np.mean(cum_ret), 6)
        max_ep_ret = round(np.max(cum_ret), 6)
        min_ep_ret = round(np.min(cum_ret), 6)
        std_ep_rew = round(np.std(cum_ret), 6) # standard deviation (spred of distribution)

        # Log stats to CSV file
        self.stats_data['episodes'].append(done_so_far)
        self.stats_data['experiment'].append(exp_name)
        self.stats_data['mean episodic length'].append(mean_ep_len)
        self.stats_data['mean episodic returns'].append(mean_ep_ret)
        self.stats_data['min episodic returns'].append(min_ep_ret)
        self.stats_data['max episodic returns'].append(max_ep_ret)
        self.stats_data['std episodic returns'].append(std_ep_rew)
        self.stats_data['mean episodic runtime'].append(mean_ep_time)
        self.stats_data['eval episodes'].append(len(cum_ret))
        self.stats_data['timestep'].append(training_steps)

        # Monitoring via W&B
        if wandb:
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
        logging.info(f"Max ep_return:        {max_ep_ret}")
        logging.info(f"Min ep_return:        {min_ep_ret}")
        logging.info(f"Mean ep_return:       {mean_ep_ret}")
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
    parser.add_argument("--video",          type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False, help="if toggled, capture video of run")
    parser.add_argument("--train",          type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, run model in training mode")
    parser.add_argument("--test",           type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False, help="if toggled, run model in testing mode")
    parser.add_argument("--hyperparam",     type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, log hyperparameters")
    parser.add_argument("--exp-name",       type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--project-name",   type=str, default='OpenAIGym-PPO', help="the name of this project") 
    parser.add_argument("--gym-id",         type=str, default="Pendulum-v1", help="the id of the gym environment")
    parser.add_argument("--learning-rate",  type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed",           type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000, help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda",           type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")

    # Parse arguments if they are given
    args = parser.parse_args()
    return args

def make_env(env_id='Pendulum-v1', gym_wrappers=False, seed=42):
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
    
    # check
    check_env(env,
              warn=True,
              skip_render_check=True)
    return env

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
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def register_env(id:str, entry_point:str):
    # register custom env with openAI gym
    # allows it to be detected by gym
    register(
    id=id,
    entry_point=entry_point,
)

####################
####################

class PPOTrainer:

    def __init__(self, 
                env,
                total_training_steps=10_000,
                max_trajectory_size=1024,
                n_rollout_steps=2048,
                n_optepochs=32,
                learning_rate_p=1e-4,
                learning_rate_v=1e-3,
                gae_lambda=0.95,
                gamma=0.99,
                epsilon=0.2,
                adam_eps=1e-5,
                momentum=0.9,
                adam=True,
                save_model=10,
                log_video=False,
                log_video_steps=10,
                render_steps=10,
                render_video=False,
                device='cpu',
                advantage_type='gae',
                normalize_adv=False,
                normalize_ret=True,
                deterministic=True, 
                seed=42, 
                project_name='PyBulletGym-Drone') -> None:

        self.env = env
        self.env_name = self.env.unwrapped.spec.id

        # hyperparam
        self.total_training_steps = total_training_steps
        self.max_trajectory_size = max_trajectory_size
        self.n_rollout_steps = n_rollout_steps
        self.n_optepochs = n_optepochs
        self.learning_rate_p = learning_rate_p
        self.learning_rate_v =learning_rate_v
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.epsilon = epsilon
        self.adam_eps = adam_eps
        self.momentum = momentum
        self.adam = adam
        self.advantage_type = advantage_type
        self.normalize_advantage = normalize_adv
        self.normalize_return = normalize_ret
        
        # experiment
        self.exp_name = f"exp_name: {self.env_name}_{CURR_DATE}"

        # logging
        self.project_name = project_name
        self.save_model = save_model
        self.render_video = render_video
        self.render_steps = render_steps
        self.log_video = log_video
        self.log_video_steps = log_video_steps
        
        # other
        self.deterministic = deterministic,
        self.seed = seed
        self.device = device

        # set everything up
        self.setup_env()
        self.setup_logging()
        self.setup_wb()

    def setup_env(self):
        # seeding
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True # self.deterministic

        # get dimensions of obs (what goes in?)
        # and actions (what goes out?)
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        logging.info(f'env observation space: {self.obs_shape}')
        logging.info(f'env action space: {self.act_shape}')
        logging.info(f'env action upper bound: {self.upper_bound}')
        logging.info(f'env action lower bound: {self.lower_bound}')

        self.obs_dim = self.obs_shape[0] 
        self.act_dim = self.act_shape[0]

        logging.info(f'env observation dim: {self.obs_dim}')
        logging.info(f'env action dim: {self.act_dim}')

    def setup_logging(self):
        self.exp_dir = f'{LOG_PATH}exp_{self.env_name}_{CURR_TIME}'
        self.model_dir = os.path.join(self.exp_dir, 'models')

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        csv_file = os.path.join(self.exp_dir, f'{self.env_name}_{CURR_TIME}.csv')
        png_file = os.path.join(self.exp_dir, f'{self.env_name}_{CURR_TIME}.png')

        self.csv_writer = CSVWriter(csv_file)
        self.stats_plotter = StatsPlotter(self.exp_dir, file_name_and_path=png_file)

    def setup_wb(self):
        if wandb:
            wandb.init(
                project=self.project_name,
                entity='drone-mechanics',
                sync_tensorboard=True,
                config={ # stores hyperparams in job
                        'env name': self.env_name,
                        'env number': 1, # only single env
                        'total_training_steps': self.total_training_steps,
                        'max sampled trajectories': self.max_trajectory_size,
                        'batches per episode': self.n_rollout_steps,
                        'number of epochs for update': self.n_optepochs,
                        'input layer size': self.obs_dim,
                        'output layer size': self.act_dim,
                        'observation space': self.obs_shape,
                        'action space': self.act_shape,
                        'action space upper bound': self.upper_bound,
                        'action space lower bound': self.lower_bound,
                        'learning rate (policy net)': self.learning_rate_p,
                        'learning rate (value net)': self.learning_rate_v,
                        'epsilon (adam optimizer)': self.adam_eps,
                        'gamma (discount)': self.gamma,
                        'epsilon (clip_range)': self.epsilon,
                        'gae lambda (GAE)': self.gae_lambda,
                        'normalize advantage': self.normalize_advantage,
                        'normalize return': self.normalize_return,
                        'seed': self.seed,
                        'experiment path': self.exp_dir,
                        'experiment name': self.exp_name
                    },
                    dir=os.getcwd(),
                    name=self.exp_name,
                    monitor_gym=True,
                    save_code=True
                )

    def create_ppo(self):
        agent = PPO_PolicyGradient(
                self.env,
                self.obs_dim,
                self.act_dim,
                total_training_steps=self.total_training_steps,
                max_trajectory_size=self.max_trajectory_size,
                n_rollout_steps=self.n_rollout_steps,
                n_optepochs=self.n_optepochs,
                learning_rate_p=self.learning_rate_p,
                learning_rate_v=self.learning_rate_v,
                gae_lambda=self.gae_lambda,
                gamma=self.gamma,
                epsilon=self.epsilon,
                adam_eps=self.adam_eps,
                momentum=self.momentum,
                adam=self.adam,
                save_model=self.save_model,
                csv_writer=self.csv_writer,
                stats_plotter=self.stats_plotter,
                log_video=self.log_video,
                log_video_steps=self.log_video_steps,
                render_steps=self.render_steps,
                render_video=self.render_video,
                device=self.device,
                exp_path=self.exp_dir,
                exp_name=self.exp_name,
                advantage_type=self.advantage_type,
                normalize_adv=self.normalize_advantage,
                normalize_ret=self.normalize_return)
        return agent

    def shutdown(self):
        # cleanup 
        self.env.close()
        wandb.run.finish() if wandb and wandb.run else None

    def get_policy(self):
        checkpoints = os.path.join(self.model_dir, f'{self.env_name}_policyNet.pth')
        policy_net = PolicyNet(self.obs_dim, self.act_dim)
        policy_net = load_model(checkpoints, policy_net, self.device)
        return policy_net


class PPOTuner:
    """ Class to tune PPO hyperparameter """
    def __init__(self) -> None:
        pass

    def hyperparam_tuning(self):
        # TODO
        param_dict = {
                'learning rate (policy net)': [1e-5, 1e-4, 1e-3, 1e-2],
                'learning rate (value net)': [1e-5, 1e-4, 1e-3, 1e-2],
                'gamma (discount)': [0.95, 0.96, 0.97, 0.98, 0.99],
                'epsilon (clip_range)': [0.1, 0.2, 0.22, 0.24,0.3],
                'gae lambda (GAE)': [0.9, 0.93, 0.95, 0.96, 0.97, 0.99, 1.0],
                'epsilon (adam optimizer)': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                'number of epochs for update': [8, 16, 32, 64, 128, 256],
                'max sampled trajectories': [32, 64, 128, 256, 512, 1024, 2048, 4096]
        }

