from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
import numpy as np
import datetime
import os
import argparse

# environemnts
import gym_pybullet_drones
import gym

# logging python
import logging
import sys

# monitoring/logging ML
import wandb

# own implementations
from network import PolicyNet, ValueNet

# constants
MODEL_PATH = './models/'

class PPO_PolicyGradient:
    """ Proximal Policy Optimization (PPO) is an online policy gradient method.
        As an online policy method it updates the policy and then discards the experience (no replay buffer).
        Thus the agent does well in environments with dense reward signals.
        The clipped objective function in PPO allows to keep the policy close to the policy 
        that was used to sample the data resulting in a more stable training. 
    """
    # Further reading
    # PPO experiments: https://nn.labml.ai/rl/ppo/experiment.html
    # PPO explained: https://huggingface.co/blog/deep-rl-ppo

    def __init__(self, 
        env, 
        in_dim, 
        out_dim,
        total_steps,
        max_trajectory_size,
        trajectory_iterations,
        noptepochs=5,
        lr_p=1e-3,
        lr_v=1e-3,
        gamma=0.99,
        epsilon=0.22,
        adam_eps=1e-5,
        render=1,
        save_model=10) -> None:
        
        # hyperparams
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_steps = total_steps
        self.max_trajectory_size = max_trajectory_size
        self.trajectory_iterations = trajectory_iterations
        self.noptepochs = noptepochs
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.gamma = gamma
        self.epsilon = epsilon
        self.adam_eps = adam_eps

        # environment
        self.env = env
        self.render_steps = render
        self.save_model = save_model

        # keep track of rewards per episode
        self.ep_returns = deque(maxlen=max_trajectory_size)

        # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor) - (policy-based method) "How the agent behaves"
        self.value_net = ValueNet(self.in_dim, 1) # Setup Value Network (Critic) -  (value-based method) "How good the action taken is."

        # add optimizer for actor and critic
        self.policy_net_optim = Adam(self.policy_net.parameters(), lr=self.lr_p, eps=self.adam_eps) # Setup Policy Network (Actor) optimizer
        self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr_v, eps=self.adam_eps)  # Setup Value Network (Critic) optimizer

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
    
    def get_values(self, obs, actions):
        """Make value selection function (outputs values for obs in a batch)."""
        values = self.value_net(obs).squeeze()
        dist = self.get_continuous_policy(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)"""
        action_dist = self.get_continuous_policy(obs) 
        action, log_prob, entropy = self.get_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy(), entropy.detach().numpy()

    def generalized_actor_critic_advantage(self, batch_rewards, values, done, normalized=True):
        """ Calculate advantage as a weighted average of A_t
            - advantage A_t gives information if this action is bettern than another at a state
            - done (Tensor): boolean flag for end of episode.
        """
        # general advantage estimage: https://nn.labml.ai/rl/ppo/gae.html
        advantages = []
        last_value = values[-1] # V(s_t+1)
        for rewards in reversed(batch_rewards):
            prev_advantage = 0
            for i in reversed(range(self.max_trajectory_size)):
                mask = 1.0 - done[i] # mask if episode completed after step i # TODO: Request done - true/false? 
                last_value = last_value * mask
                prev_advantage = prev_advantage * mask
                delta = rewards[i] + self.gamma * last_value - values[i]
                prev_advantage = delta + self.gamma * self.lambda_ * prev_advantage
                advantages.insert(0, prev_advantage) # we need to reverse it again
                last_value = values[i]
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages

    def actor_critic_advantage(self, rewards, values, normalized=True):
        """ Advantage calculation
            - advantage A_t gives information if this action is bettern than another at a state
        """
        cum_rewards = []
        # STEP 4: compute rewards to go
        # discounted return: G(t) = R(t) + gamma * R(t-1)
        for rewards in reversed(rewards): # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_rewards.insert(0, discounted_reward)
        returns = torch.tensor(cum_rewards, dtype=torch.float)
        # STEP 5: compute advantage estimates A_t at step t
        advantages = returns - values
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages
    
    def normalize_adv(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    def finish_episode(self):
        pass 

    def collect_rollout(self, n_step=1, render=True):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)"""
        
        step, ep_rewards = 0, []

        # collect trajectories
        trajectory_obs = []
        trajectory_actions = []
        trajectory_action_probs = []
        trajectory_dones = []
        batch_rewards = []
        batch_lens = []

        # Run Monte Carlo simulation for n timesteps per batch
        logging.info("Collecting batch trajectories...")
        while step < n_step:
            
            # rewards collected per episode
            ep_rewards, done = [], False 
            obs = self.env.reset()

            # Run episode for a fixed amount of timesteps
            # to keep rollout size fixed and episodes independent
            for ep_t in range(0, self.max_trajectory_size):
                # render gym envs
                if render and ep_t % self.render_steps == 0:
                    self.env.render(mode='human')
                
                step += 1 

                # action logic 
                # sampled via policy which defines behavioral strategy of an agent
                action, log_probability, _ = self.step(obs)
                        
                # STEP 3: collecting set of trajectories D_k by running action 
                # that was sampled from policy in environment
                __obs, reward, done, _ = self.env.step(action)

                # collection of trajectories in batches
                trajectory_obs.append(obs)
                trajectory_actions.append(action)
                trajectory_action_probs.append(log_probability)
                ep_rewards.append(reward)
                trajectory_dones.append(done)
                    
                obs = __obs

                # break out of loop if episode is terminated
                if done:
                    break
            
            batch_lens.append(ep_t + 1) # as we started at 0
            batch_rewards.append(ep_rewards)

        # convert trajectories to torch tensors
        obs = torch.tensor(np.array(trajectory_obs), dtype=torch.float)
        actions = torch.tensor(np.array(trajectory_actions), dtype=torch.float)
        log_probs = torch.tensor(np.array(trajectory_action_probs), dtype=torch.float)
        dones = torch.tensor(np.array(trajectory_dones), dtype=torch.float)
        returns = torch.tensor(np.array(batch_rewards), dtype=torch.float)

        # STEP 4: Calculate cummulated reward

        # Calculate the stats
        cum_rews = [np.sum(ep_rews) for ep_rews in batch_rewards]
        mean_ep_lens = np.mean(batch_lens)
        mean_ep_rews = np.mean(cum_rews)
        std_ep_rews = np.std(cum_rews) # calculate standard deviation (spred of distribution)
        
        # Log stats
        wandb.log({
            "train/mean episode length": mean_ep_lens,
            "train/mean episode returns": mean_ep_rews,
            "train/std episode returns": std_ep_rews,
        })

        return obs, actions, log_probs, dones, returns, batch_lens, mean_ep_rews
                

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
        value_loss = self.value_net.loss(values, returns) # TODO: discounted return 
        value_loss.backward()
        self.value_net_optim.step()

        return policy_loss, value_loss

    def learn(self):
        """"""
        steps = 0

        while steps < self.total_steps:
        
            # Collect trajectory
            # STEP 3-4: simulate and collect trajectories --> the following values are all per batch
            obs, actions, log_probs, dones, returns, batch_lens, mean_ep_rews = self.collect_rollout(n_step=self.trajectory_iterations)
            
            # timesteps simulated so far for batch collection
            steps += np.sum(batch_lens)

            # STEP 5: compute advantage estimates A_t at timestep t_step
            values, _ , _ = self.get_values(obs, actions)
            advantages = self.actor_critic_advantage(returns, values.detach())
            # advantages = self.generalized_actor_critic_advantage(batch_rewards, values.detach(), dones)
            # update network params 
            for _ in range(self.noptepochs):
                # STEP 6-7: calculate loss and update weights
                values, curr_log_probs, _ = self.get_values(obs, actions)
                policy_loss, value_loss = self.train(values, returns, advantages, log_probs, curr_log_probs, self.epsilon)

            logging.info('\n')
            logging.info('###########################################')
            logging.info(f"Mean return: {mean_ep_rews}")
            logging.info(f"Policy loss: {policy_loss}")
            logging.info(f"Value loss:  {value_loss}")
            logging.info(f"Time step:   {steps}")
            logging.info('###########################################')
            logging.info('\n')
            
            # logging for monitoring in W&B
            wandb.log({
                'train/episode': steps,
                'train/policy loss': policy_loss,
                'train/value loss': value_loss})
            
            # store model in checkpoints
            if steps % self.save_model == 0:
                env_name = self.env.unwrapped.spec.id
                torch.save({
                    'epoch': steps,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.policy_net_optim.state_dict(),
                    'loss': policy_loss,
                    }, f'{MODEL_PATH}{env_name}__policyNet')
                torch.save({
                    'epoch': steps,
                    'model_state_dict': self.value_net.state_dict(),
                    'optimizer_state_dict': self.value_net_optim.state_dict(),
                    'loss': policy_loss,
                    }, f'{MODEL_PATH}{env_name}__valueNet')
