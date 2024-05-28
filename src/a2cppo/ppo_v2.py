from collections import deque
import time
from typing import Tuple
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
import numpy as np
import os

# gym environment
import gym

# logging python
import logging
import sys

# monitoring/logging ML
import wandb

# hyperparameter tuning
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

# own implementations
from network import PolicyNet, ValueNet
from wrapper.stats_logger import StatsPlotter
from wrapper.stats_logger import CSVWriter


class PPO_PolicyGradient_V2:
    """
    Proximal Policy Optimization (PPO) is an on-policy, actor-critic reinforcement learning algorithm that
    uses a clipped objective function to stabilize training and prevent large policy updates.
    PPO is an online policy gradient method that updates the policy and then discards the experience.
    This makes PPO well-suited for environments with dense reward signals, where the agent can learn
    effectively from a single trajectory.

        The clipped objective function in PPO is defined as:

        L_clip(θ) = E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ]

        where r(θ) is the ratio of the new policy to the old policy, A is the advantage function, and ε is
        a hyperparameter that controls the amount of clipping.

    The clipped objective function is a compromise between the policy gradient and the trust region method.
    The trust region method uses a KL divergence to prevent large policy updates, but it can be computationally
    expensive and difficult to tune. The clipped objective function is a more stable and efficient alternative
    to the trust region method.

    In this implementation, we use a value function estimator (VFE) to estimate the advantage function, and
    we use stochastic gradient descent (SGD) to optimize the clipped objective function. We also use a
    rolling horizon of fixed length to store the most recent trajectories, and we use mini-batches to
    improve the efficiency of the SGD updates.
    """

    # Further reading for deeper understanding:
    # PPO experiments: https://nn.labml.ai/rl/ppo/experiment.html
    #                  https://nn.labml.ai/rl/ppo/index.html
    # PPO explained:   https://huggingface.co/blog/deep-rl-ppo

    def __init__(self, 
        env, 
        in_dim, 
        out_dim,
        total_training_steps,
        max_batch_size,
        n_rollout_steps,
        noptepochs=5,
        lr_p=1e-3,
        lr_v=1e-3,
        gae_lambda=0.95,
        gamma=0.99,
        epsilon=0.22,
        adam_eps=1e-5,
        momentum=0.9,
        adam=True,
        render=10,
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
        self.max_batch_size = max_batch_size
        self.n_rollout_steps = n_rollout_steps
        self.noptepochs = noptepochs
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
        self.render_steps = render
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
        self.ep_returns = deque(maxlen=max_batch_size)
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
            }

                                                                # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim)  # Setup Policy Network (Actor) - (policy-based method) "How the agent behaves"
        self.value_net = ValueNet(self.in_dim, 1)               # Setup Value Network (Critic) -  (value-based method) "How good the action taken is."

                                                                # add optimizer for actor and critic
        if self.adam:
            self.policy_net_optim = Adam(self.policy_net.parameters(), lr=self.lr_p, eps=self.adam_eps) # Setup Policy Network (Actor) optimizer
            self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr_v, eps=self.adam_eps)  # Setup Value Network (Critic) optimizer  
        else:
            self.policy_net_optim = SGD(self.policy_net.parameters(), lr=self.lr_p, momentum=self.momentum)
            self.value_net_optim = SGD(self.value_net.parameters(), lr=self.lr_v, momentum=self.momentum)

    def get_continuous_policy(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        """
        Compute the action distribution in a continuous action space.

        This method uses a multivariate normal distribution to model the action distribution,
        with the mean action predicted by the policy network (actor) and a fixed diagonal covariance
        matrix. The use of a multivariate normal distribution allows us to capture correlations
        between features and to detect outliers more effectively.

        Args:
            obs (torch.Tensor): The observation tensor, with shape (batch_size, num_features).

        Returns:
            torch.distributions.Distribution: The action distribution, with shape (batch_size, num_actions).
        """
        action_mean = self.policy_net(obs)  # query the policy network (actor) for the mean action
        action_stddev = torch.full(size=(self.out_dim,), fill_value=0.5)  # fixed standard deviation for each action
        cov_matrix = torch.diag(action_stddev)  # fixed diagonal covariance matrix
        return torch.distributions.MultivariateNormal(loc=action_mean, covariance_matrix=cov_matrix)

    def get_action(self, dist: torch.distributions.Distribution) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the action distribution and compute its log probability and entropy.

        This method uses the action distribution computed by the `get_continuous_policy` method to
        sample an action and to compute its log probability and entropy. The log probability and entropy
        are used to compute the policy gradient and the entropy bonus, respectively.

        Args:
            dist (torch.distributions.Distribution): The action distribution, with shape (batch_size, num_actions).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors, containing the sampled action,
            its log probability, and its entropy, respectively.
        """
        action = dist.sample()  # sample an action from the action distribution
        log_prob = dist.log_prob(action)  # compute the log probability of the sampled action
        entropy = dist.entropy()  # compute the entropy of the action distribution
        return action, log_prob, entropy
    
    def get_values(self, obs, actions):
        """Make value selection function (outputs values for obs in a batch).
            - obs: observation
            - actions: actions

            Returns:
                - values: value estimates
                - log_prob: log probability of action
                - entropy: entropy of action
        """
        values = self.value_net(obs).squeeze()
        dist = self.get_continuous_policy(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def get_value(self, obs):
        """Get value estimate for observation.
            - obs: observation

            Returns:
                - value estimate
        """
        return self.value_net(obs).squeeze()

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)
            - obs: observation

            Returns:
                - action: action
                - log_prob: log probability of action
                - entropy: entropy of action
        """
        action_dist = self.get_continuous_policy(obs) 
        action, log_prob, entropy = self.get_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy(), entropy.detach().numpy()

    ################################################################################
    
    def advantage_estimate(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ Calculating advantage estimate using TD error (Temporal Difference Error).
            TD Error can be used as an estimator for Advantage function,
            - bias-variance: TD has low variance, but IS biased
            - dones: only get reward at end of episode, not disounted next state value

            A(s,a) = r + (gamma * V(s_t+1)) - V(s_t)

            Args:
            - episode_rewards: rewards per episode
            - values: value estimates
            - normalized_adv: normalize advantage
            - normalized_ret: normalize returns

            Returns:
                - advantages: advantage estimates
                - returns: cummulative rewards
        """
        # Step 4: Calculate returns
        advantages = []
        cum_returns = []
        for rewards in reversed(episode_rewards):   # reversed order
            for reward in reversed(rewards):
                cum_returns.insert(0, reward)       # reverse it again
        cum_returns = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        #  A(s,a) = r - V(s_t)
        advantages = cum_returns - values
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns

    ################################################################################

    def advantage_reinforce(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ Advantage Reinforce A(s_t, a_t) = G(t)
            The advantage function can be estimated by the total discounted reward G(t).
            The total discounted reward G(t) is the sum of all rewards from time step t to the end of the episode.
            The advantage function is the total discounted reward G(t) at time step t.
            The advantage function can be estimated by the total discounted reward G(t).

            Args:
            - episode_rewards: rewards per episode
            - values: value estimates
            - normalized_adv: normalize advantage
            - normalized_ret: normalize returns

            Returns:
                - advantages: advantage estimates
                - returns: cummulative rewards
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
        if normalized_ret:
            cum_returns = self.normalize_ret(cum_returns)
        # Step 5: Calculate advantage
        # A(s,a) = G(t)
        advantages = torch.tensor(np.array(cum_returns), device=self.device, dtype=torch.float)
        # normalize for more stability
        if normalized_adv:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns

    ################################################################################

    def advantage_a2c(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ Advantage Actor-Critic by calculating delta [G(t) - V(s_t)].
            The advantage function can be estimated by the TD error.

            A(s,a) = G(t) - V(s_t)
                    - G(t) is the total disounted reward
                    - V(s_t) is the value estimate

            Args:
            - episode_rewards: rewards per episode
            - values: value estimates
            - normalized_adv: normalize advantage
            - normalized_ret: normalize returns

            Returns:
                - advantages: advantage estimates
                - returns: cummulative rewards
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

    ################################################################################

    def advantage_td_a2c(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ Advantage TD Actor-Critic 
            The TD Error can be used as an estimator for Advantage function

            A(s,a) = r + (gamma * V(s_t+1)) - V(s_t)
                    - TD error: A(s,a) = r + (gamma * V(s_t+1)) - V(s_t)

            Args:
            - episode_rewards: rewards per episode
            - values: value estimates
            - normalized_adv: normalize advantage
            - normalized_ret: normalize returns

            Returns:
                - advantages: advantage estimates
                - returns: cummulative rewards
        """
        # Step 4: Calculate returns
        # G(t) is the total disounted reward
        # return value: G(t) = R(t) + gamma * R(t-1)
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

    ################################################################################

    def compute_gae(self, episode_rewards, values, normalized_adv=False, normalized_ret=False):
        """ The Generalized Advanatage Estimate (GAE) is a method to estimate the advantage function.
                - GAE allows to balance bias and variance through a weighted average of A_t
                - gamma (dicount factor): allows reduce variance by downweighting rewards that correspond to delayed effects

            GAE: A_t = δ_t + γ * λ * A(t+1)
                 δ_t = r_t + γ * V(s_t+1) − V(s_t)

            Args:
            - episode_rewards: rewards per episode
            - values: value estimates
            - normalized_adv: normalize advantage
            - normalized_ret: normalize returns

            Returns:
                - advantages: advantage estimates
                - returns: cummulative rewards
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


    def compute_gae2(self, obs, next_obs, episode_rewards, dones, normalized_adv=False, normalized_ret=False):
        """ The Generalized Advantage Estimate calculation. GAE defines advantage as a weighted average of A_t.
            The advantage measures if an action is better or worse than the policy's default behavior.
            Goal is to find the maximum Advantage representing the benefit of choosing a specific action.

            Args:
            - obs: observations
            - next_obs: next observations
            - episode_rewards: rewards per episode
            - dones: done flag
            - normalized_adv: normalize advantage
            - normalized_ret: normalize returns

            Returns:
                - advantages: advantage estimates
                - returns: cummulative rewards
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

    ################################################################################

    def normalize_adv(self, advantages):
        """Normalize advantages to stabilize training
           - advantages: advantage values

              Returns:
                - normalized advantages
        """
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def normalize_ret(self, returns):
        """Normalize returns to reduce variance and stabilize training
           - returns: cumulative rewards

                Returns:
                    - normalized returns  
        """
        eps = np.finfo(np.float32).eps.item()
        return (returns - returns.mean()) / (returns.std() + eps)

    def finish_episode(self):
        pass 

    ################################################################################
    
    def collect_rollout(self, n_steps=1, render=False):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)
              - n_steps: number of timesteps to simulate per batch
              - render: render the gym environment

            Returns:
                - obs: observations
                - next_obs: next observations
                - actions: actions
                - action_log_probs: log probabilities of actions
                - dones: done flag
                - episode_rewards: rewards per episode
                - episode_lens: length of episodes
                - episode_time: time elapsed per episode
        """
        
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
        logging.info("Collecting batch trajectories...")
        while t_step < n_steps:
            
            # rewards collected
            rewards, done = [], False 
            obs = self.env.reset()

            # measure time elapsed for one episode
            # torch.cuda.synchronize()
            start_epoch = time.time()

            # Run episode for a fixed amount of timesteps
            # to keep rollout size fixed and episodes independent
            for t_episode in range(0, self.max_batch_size):
                # render gym envs
                if render and t_episode % self.render_steps == 0:
                    self.env.render()
                
                t_step += 1 

                # action logic 
                # sampled via policy which defines behavioral strategy of an agent
                action, log_probability, _ = self.step(obs)
                        
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

            episode_lens.append(t_episode + 1) # as we started at 0
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
        self.policy_net_optim.zero_grad()                       # reset optimizer
        policy_loss = self.policy_net.loss(advantages, batch_log_probs, curr_log_probs, epsilon)
        policy_loss.backward()                                  # backpropagation
        self.policy_net_optim.step()                            # single optimization step (updates parameter)
                                                                # loss of the value network
        self.value_net_optim.zero_grad()                        # reset optimizer
        value_loss = self.value_net.loss(values, returns)
        value_loss.backward()
        self.value_net_optim.step()

        return policy_loss, value_loss

    def learn(self, adv_estimation='td_actor_critic'):
        """"""
        adv_config = {
            'reinforce': self.advantage_reinforce,
            'actor_critic': self.advantage_a2c,
            'td_actor_critic': self.advantage_td_a2c,
            'gae': self.compute_gae,
            'gae2': self.compute_gae2
        }

        training_steps = 0
        adv_func = adv_config[adv_estimation]                       # select advantage estimation method

        while training_steps < self.total_training_steps:
            policy_losses, value_losses = [], []
                                                                    # Collect data over one episode
                                                                    # STEP 3: simulate and collect trajectories: the following values are all per batch over one episode
            obs, next_obs, actions, batch_log_probs, dones, rewards,\
                ep_lens, ep_time = self.collect_rollout(n_steps=self.n_rollout_steps)                         
            training_steps += np.sum(ep_lens)                        # timesteps simulated so far for batch collection

                                                                    # STEP 4-5: 
            values, _ , _ = self.get_values(obs, actions)           # Calculate cummulated rewarde and advantage at timestep t_step

            # Select advantage calculation method
            advantages, cum_returns = adv_func(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)

            # advantages, cum_returns = self.advantage_reinforce(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.advantage_a2c(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.advantage_td_a2c(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            # advantages, cum_returns = self.compute_gae(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            
            # update network params 
            for _ in range(self.noptepochs):
                # STEP 6-7: calculate loss and update weights
                values, curr_log_probs, _ = self.get_values(obs, actions)
                policy_loss, value_loss = self.train(values, cum_returns, advantages, batch_log_probs, curr_log_probs, self.epsilon)
                
                policy_losses.append(policy_loss.detach().numpy())
                value_losses.append(value_loss.detach().numpy())

            # log all statistical values to CSV
            self.log_stats(policy_losses, value_losses, rewards, ep_lens, training_steps, ep_time, exp_name=self.exp_name)

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
            self.stats_plotter.plot_seaborn_fill(df, x='timestep', 
                                                 y='mean episodic returns', 
                                                y_min='min episodic returns', 
                                                y_max='max episodic returns',  
                                                title=f'{env_name}', 
                                                x_label='Timestep', 
                                                y_label='Mean Episodic Return', 
                                                color='blue', 
                                                smoothing=6, 
                                                wandb=wandb, 
                                                xlim_up=self.total_training_steps)

        # save files in path
        wandb.save(os.path.join(self.exp_path, "*csv"))
        # Save any files starting with "ppo"
        wandb.save(os.path.join(wandb.run.dir, "ppo*"))


    def log_stats(self, p_losses, v_losses, batch_return, episode_lens, training_steps, time, exp_name='experiment'):
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
            'train/std episode returns': std_ep_rew,
            'train/mean episode runtime': mean_ep_time,
            'train/mean episode length': mean_ep_len
        })

        logging.info('\n')
        logging.info(f'------------ Episode: {training_steps} --------------')
        logging.info(f"Mean return:          {mean_ep_ret}")
        logging.info(f"Mean policy loss:     {mean_p_loss}")
        logging.info(f"Mean value loss:      {mean_v_loss}")
        logging.info('--------------------------------------------')
        logging.info('\n')
