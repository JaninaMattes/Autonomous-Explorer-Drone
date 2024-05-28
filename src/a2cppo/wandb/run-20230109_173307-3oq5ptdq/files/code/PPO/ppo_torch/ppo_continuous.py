from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from distutils.util import strtobool
import numpy as np
from datetime import datetime
import os
import argparse

# gym environment
import gym

# logging python
import logging
import sys

# monitoring/logging ML
import wandb
from wrapper.stats_logger import CSVWriter

# hyperparameter tuning
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

# Paths and other constants
MODEL_PATH = './models/'
LOG_PATH = './log/'
VIDEO_PATH = './video/'

CURR_DATE = datetime.today().strftime('%Y-%m-%d')


####################
####### TODO #######
####################

# Hint: Please if working on it mark a todo as (done) if done
# 1) Check current implementation against article: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# 3) Check calculation of advantage 

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
        """Objective function defined by mean-squared error"""
        # return 0.5 * ((rewards - values)**2).mean() # MSE loss
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
        ratio = torch.exp(curr_log_probs - batch_log_probs) # ratio between pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        clip_1 = ratio * advantages
        clip_2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = (-torch.min(clip_1, clip_2)).mean() # negative as Adam mins loss, but we want to max it
        # calc clip frac
        # self.clip_fraction = (abs((ratio - 1.0)) > clip_eps).to(torch.float).mean()
        return policy_loss


####################
####################

class PPO_PolicyGradient_V1:

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
        gae_lambda=0.95,
        gamma=0.99,
        epsilon=0.22,
        adam_eps=1e-5,
        render=10,
        save_model=10,
        csv_writer=None,
        log_video=False) -> None:
        
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
        self.gae_lambda = gae_lambda

        # environment
        self.env = env
        self.render_steps = render
        self.save_model = save_model

        # track video of gym
        self.log_video = log_video

        # keep track of rewards per episode
        self.ep_returns = deque(maxlen=max_trajectory_size)
        self.csv_writer = csv_writer
        self.stats_data = {'mean episodic length': [], 'mean episodic rewards': [], 'timestep': []}

        # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor) - (policy-based method) "How the agent behaves"
        self.value_net = ValueNet(self.in_dim, 1) # Setup Value Network (Critic) -  (value-based method) "How good the action taken is."

        # add optimizer for actor and critic
        self.policy_net_optim = Adam(self.policy_net.parameters(), lr=self.lr_p, eps=self.adam_eps) # Setup Policy Network (Actor) optimizer
        self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr_v, eps=self.adam_eps)  # Setup Value Network (Critic) optimizer


class PPO_PolicyGradient_V2:
    """ Proximal Policy Optimization (PPO) is an online policy gradient method.
        As an online policy method it updates the policy and then discards the experience (no replay buffer).
        Thus the agent does well in environments with dense reward signals.
        The clipped objective function in PPO allows to keep the policy close to the policy 
        that was used to sample the data resulting in a more stable training. 
    """
    # Further reading
    # PPO experiments: https://nn.labml.ai/rl/ppo/experiment.html
    #                  https://nn.labml.ai/rl/ppo/index.html
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
        gae_lambda=0.95,
        gamma=0.99,
        epsilon=0.22,
        adam_eps=1e-5,
        render=10,
        save_model=10,
        csv_writer=None,
        log_video=False) -> None:
        
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
        self.gae_lambda = gae_lambda

        # environment
        self.env = env
        self.render_steps = render
        self.save_model = save_model

        # track video of gym
        self.log_video = log_video

        # keep track of rewards per episode
        self.ep_returns = deque(maxlen=max_trajectory_size)
        self.csv_writer = csv_writer
        self.stats_data = {'mean episodic length': [], 'mean episodic rewards': [], 'timestep': []}

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

    def get_value(self, obs):
        return self.value_net(obs).squeeze()

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)"""
        action_dist = self.get_continuous_policy(obs) 
        action, log_prob, entropy = self.get_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy(), entropy.detach().numpy()

    def cummulative_return(self, batch_rewards, normalized=False):
        """Calculate cummulative rewards with discount factor gamma."""
        # Cumulative rewards: https://gongybable.medium.com/reinforcement-learning-introduction-609040c8be36
        # return value: G(t) = R(t) + gamma * R(t-1)
        cum_returns = []
        for rewards in reversed(batch_rewards): # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward) # reverse it again
        if normalized:
            cum_returns = (cum_returns - cum_returns.mean()) / cum_returns.std()
        return torch.tensor(cum_returns, dtype=torch.float)

    def advantage_estimate_(self, returns, values, normalized=True):
        """Calculate delta, which is defined by delta = r - v """
        advantages = returns - values # delta = r - v
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages

    def advantage_estimate(self, next_obs, obs, batch_rewards, dones, normalized=True):
        """Calculating advantage estimate using TD error.
            - done: only get reward at end of episode, not disounted next state value
        """
        advantages = []
        returns = []
        
        for rewards in reversed(batch_rewards): # reversed order
            discounted_return = 0
            for i in reversed(range(len(rewards))):
                mask = (1 - dones[i])
                discounted_return = rewards[i] + (self.gamma * discounted_return)
                advantage = discounted_return + (mask * self.gamma *  self.get_value(next_obs[i])) - self.get_value(obs[i])
                advantages.insert(0, advantage)
                returns.insert(0, discounted_return)
        advantages = np.array(advantages)
        if normalized:
            advantages = self.normalize_adv(advantages)
        return torch.tensor(advantages, dtype=torch.float), torch.tensor(np.array(returns), dtype=torch.float)

    def generalized_advantage_estimate_1(self, batch_rewards, values, normalized=True):
        """ Generalized Advantage Estimate calculation
            - GAE defines advantage as a weighted average of A_t
            - advantage measures if an action is better or worse than the policy's default behavior
            - want to find the maximum Advantage representing the benefit of choosing a specific action
        """
        # check if tensor and convert to numpy
        if torch.is_tensor(batch_rewards):
            batch_rewards = batch_rewards.detach().numpy()
        if torch.is_tensor(values):
            values = values.detach().numpy()

        # STEP 4: compute returns as G(t) = R(t) + gamma * R(t-1)
        # STEP 5: compute advantage estimates δ_t = − V(s_t) + r_t
        cum_returns = []
        advantages = []
        for rewards in reversed(batch_rewards): # reversed order
            discounted_reward = 0
            for i in reversed(range(len(rewards))):
                discounted_reward = rewards[i] + (self.gamma * discounted_reward)
                # Hinweis @Thomy Delta Könnte als advantage ausreichen
                # δ_t = − V(s_t) + r_t
                delta = discounted_reward - values[i] # delta = r - v
                advantages.insert(0, delta)
                cum_returns.insert(0, discounted_reward) # reverse it again

        # convert numpy to torch tensor
        cum_returns = torch.tensor(np.array(cum_returns), dtype=torch.float)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float)
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns


    def generalized_advantage_estimate_2(self, obs, next_obs, batch_rewards, dones, normalized=True):
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
        for rewards in reversed(batch_rewards):
            prev_advantage = 0
            returns_current = ns_values[-1]  # V(s_t+1)
            for i in reversed(range(len(rewards))):
                # STEP 5: compute advantage estimates A_t at step t
                mask = (1.0 - dones[i])
                gamma = self.gamma * mask
                td_error = rewards[i] + gamma * ns_values[i] - s_values[i]
                # A_t = δ_t + γ * λ * A(t+1)
                prev_advantage = td_error + gamma * self.gae_lambda * prev_advantage
                returns_current = rewards[i] + gamma * returns_current
                # reverse it again
                returns.insert(0, returns_current)
                advantages.insert(0, prev_advantage)
        advantages = np.array(advantages)
        if normalized:
            advantages = self.normalize_adv(advantages)
        return torch.tensor(np.array(advantages), dtype=torch.float), torch.tensor(np.array(returns), dtype=torch.float)


    def generalized_advantage_estimate_3(self, batch_rewards, values, dones, normalized=True):
        """ Calculate advantage as a weighted average of A_t
                - advantage measures if an action is better or worse than the policy's default behavior
                - GAE allows to balance bias and variance through a weighted average of A_t

                - gamma (dicount factor): allows reduce variance by downweighting rewards that correspond to delayed effects
                - done (Tensor): boolean flag for end of episode. TODO: Q&A
        """
        # general advantage estimage paper: https://arxiv.org/pdf/1506.02438.pdf
        # general advantage estimage other: https://nn.labml.ai/rl/ppo/gae.html

        advantages = []
        returns = []
        values = values.detach().numpy()
        for rewards in reversed(batch_rewards): # reversed order
            prev_advantage = 0
            discounted_reward = 0
            last_value = values[-1] # V(s_t+1)
            for i in reversed(range(len(rewards))):
                # TODO: Q&A handling of special cases GAE(γ, 0) and GAE(γ, 1)
                # bei Vetorisierung, bei kurzen Episoden (done flag)
                # mask if episode completed after step i 
                mask = 1.0 - dones[i] 
                last_value = last_value * mask
                prev_advantage = prev_advantage * mask

                # TD residual of V with discount gamma
                # δ_t = − V(s_t) + r_t + γ * V(s_t+1)
                # TODO: Delta Könnte als advantage ausreichen, r - v könnte ausreichen
                delta = - values[i] + rewards[i] + (self.gamma * last_value)
                # discounted sum of Bellman residual term
                # A_t = δ_t + γ * λ * A(t+1)
                prev_advantage = delta + self.gamma * self.gae_lambda * prev_advantage
                discounted_reward = rewards[i] + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward) # reverse it again
                advantages.insert(0, prev_advantage) # reverse it again
                # store current value as V(s_t+1)
                last_value = values[i]
        advantages = torch.tensor(np.array(advantages), dtype=torch.float)
        returns = torch.tensor(np.array(returns), dtype=torch.float)
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages, returns


    def normalize_adv(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def normalize_ret(self, returns):
        return (returns - returns.mean()) / returns.std()

    def finish_episode(self):
        pass 

    def collect_rollout(self, n_step=1, render=True):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)"""
        
        step, trajectory_rewards = 0, []

        # collect trajectories
        trajectory_obs = []
        trajectory_nextobs = []
        trajectory_actions = []
        trajectory_action_probs = []
        trajectory_dones = []
        batch_rewards = []
        batch_lens = []

        # Run Monte Carlo simulation for n timesteps per batch
        logging.info("Collecting batch trajectories...")
        while step < n_step:
            
            # rewards collected per episode
            trajectory_rewards, done = [], False 
            obs = self.env.reset()

            # Run episode for a fixed amount of timesteps
            # to keep rollout size fixed and episodes independent
            for ep_t in range(0, self.max_trajectory_size):
                # render gym envs
                if render and ep_t % self.render_steps == 0:
                    self.env.render()
                
                step += 1 

                # action logic 
                # sampled via policy which defines behavioral strategy of an agent
                action, log_probability, _ = self.step(obs)
                        
                # STEP 3: collecting set of trajectories D_k by running action 
                # that was sampled from policy in environment
                __obs, reward, done, info = self.env.step(action)

                # collection of trajectories in batches
                trajectory_obs.append(obs)
                trajectory_nextobs.append(__obs)
                trajectory_actions.append(action)
                trajectory_action_probs.append(log_probability)
                trajectory_rewards.append(reward)
                trajectory_dones.append(done)
                    
                obs = __obs

                # break out of loop if episode is terminated
                if done:
                    break
            
            batch_lens.append(ep_t + 1) # as we started at 0
            batch_rewards.append(trajectory_rewards)

        # convert trajectories to torch tensors
        obs = torch.tensor(np.array(trajectory_obs), dtype=torch.float)
        next_obs = torch.tensor(np.array(trajectory_nextobs), dtype=torch.float)
        actions = torch.tensor(np.array(trajectory_actions), dtype=torch.float)
        action_log_probs = torch.tensor(np.array(trajectory_action_probs), dtype=torch.float)
        dones = torch.tensor(np.array(trajectory_dones), dtype=torch.float)

        return obs, next_obs, actions, action_log_probs, dones, batch_rewards, batch_lens
                

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
        steps = 0

        while steps < self.total_steps:
            policy_losses, value_losses = [], []
            # Collect trajectory
            # STEP 3: simulate and collect trajectories --> the following values are all per batch
            obs, next_obs, actions, batch_log_probs, dones, rewards, batch_lens = self.collect_rollout(n_step=self.trajectory_iterations)

            # timesteps simulated so far for batch collection
            steps += np.sum(batch_lens)

            # STEP 4-5: Calculate cummulated reward and GAE at timestep t_step
            values, _ , _ = self.get_values(obs, actions)
            # cum_returns = self.cummulative_return(rewards)
            # advantages = self.advantage_estimate_(cum_returns, values.detach())
            advantages, cum_returns = self.generalized_advantage_estimate_1(rewards, values.detach())

            # update network params 
            for _ in range(self.noptepochs):
                # STEP 6-7: calculate loss and update weights
                values, curr_log_probs, _ = self.get_values(obs, actions)
                policy_loss, value_loss = self.train(values, cum_returns, advantages, batch_log_probs, curr_log_probs, self.epsilon)
                
                policy_losses.append(policy_loss.detach().numpy())
                value_losses.append(value_loss.detach().numpy())

            self.log_stats(policy_losses, value_losses, rewards, batch_lens, steps)

            # store model in checkpoints
            if steps % self.save_model == 0:
                env_name = env.unwrapped.spec.id
                policy_net_name = f'{MODEL_PATH}{env_name}_{CURR_DATE}_policyNet.pth'
                value_net_name = f'{MODEL_PATH}{env_name}_{CURR_DATE}_valueNet.pth'
                torch.save({
                    'epoch': steps,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.policy_net_optim.state_dict(),
                    'loss': policy_loss,
                    }, policy_net_name)
                torch.save({
                    'epoch': steps,
                    'model_state_dict': self.value_net.state_dict(),
                    'optimizer_state_dict': self.value_net_optim.state_dict(),
                    'loss': value_loss,
                    }, value_net_name)

                if wandb:
                    wandb.save(policy_net_name)
                    wandb.save(value_net_name)

                # Log to CSV
                if self.csv_writer is not None:
                    self.csv_writer(self.stats_data)
                    for value in self.stats_data.values():
                        del value[:]


    def log_stats(self, p_losses, v_losses, batch_return, batch_lens, steps):
        """Calculate stats and log to W&B, CSV, logger """
        if torch.is_tensor(batch_return):
            batch_return = batch_return.detach().numpy()
        # calculate stats
        mean_p_loss = np.mean([np.sum(loss) for loss in p_losses])
        mean_v_loss = np.mean([np.sum(loss) for loss in v_losses])

        # Calculate the stats of an episode
        cum_ret = [np.sum(ep_rews) for ep_rews in batch_return]
        mean_ep_lens = np.mean(batch_lens)
        mean_ep_rews = np.mean(cum_ret)
        # calculate standard deviation (spred of distribution)
        std_ep_rews = np.std(cum_ret)

        # Log stats to CSV file
        self.stats_data['mean episodic length'].append(mean_ep_lens)
        self.stats_data['mean episodic rewards'].append(mean_ep_rews)
        self.stats_data['timestep'].append(steps)

        # Monitoring via W&B
        wandb.log({
            'train/timesteps': steps,
            'train/mean policy loss': mean_p_loss,
            'train/mean value loss': mean_v_loss,
            'train/mean episode length': mean_ep_lens,
            'train/mean episode returns': mean_ep_rews,
            'train/std episode returns': std_ep_rews,
        })

        logging.info('\n')
        logging.info(f'------------ Episode: {steps} --------------')
        logging.info(f"Mean return:          {mean_ep_rews}")
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
    parser.add_argument("--video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, capture video of run")
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, run model in training mode")
    parser.add_argument("--test", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, run model in testing mode")
    parser.add_argument("--hyperparam", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, log hyperparameters")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
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

def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def simulate_rollout(policy_net, env, render=True):
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
			if render:
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

def train(env, in_dim, out_dim, total_steps, max_trajectory_size, trajectory_iterations,
          noptepochs, learning_rate_p, learning_rate_v, gae_lambda, gamma, epsilon,
          adam_epsilon, render_steps, save_steps, csv_writer, log_video=False):
    """Train the policy network (actor) and the value network (critic) with PPO"""
    agent = PPO_PolicyGradient_V2(
                env, 
                in_dim=in_dim, 
                out_dim=out_dim,
                total_steps=total_steps,
                max_trajectory_size=max_trajectory_size,
                trajectory_iterations=trajectory_iterations,
                noptepochs=noptepochs,
                lr_p=learning_rate_p,
                lr_v=learning_rate_v,
                gae_lambda = gae_lambda,
                gamma=gamma,
                epsilon=epsilon,
                adam_eps=adam_epsilon,
                render=render_steps,
                save_model=save_steps,
                csv_writer=csv_writer,
                log_video=log_video)
    
    # run training for a total amount of steps
    agent.learn()

def test(path, env, in_dim, out_dim, steps=10_000, render=True, log_video=False):
    """Test the policy network (actor)"""
    # load model and test it
    policy_net = PolicyNet(in_dim, out_dim)
    policy_net = load_model(path, policy_net)
    
    for ep_num, (ep_len, ep_ret) in enumerate(simulate_rollout(policy_net, env, render)):
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
                total_steps=config.total_steps,
                max_trajectory_size=config.max_trajectory_size,
                trajectory_iterations=config.trajectory_iterations,
                noptepochs=config.noptepochs,
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
    create_path(MODEL_PATH)
    create_path(LOG_PATH)
    
    # Hyperparameter
    total_steps = 10_000_000         # time steps regarding batches collected and train agent
    max_trajectory_size = 1000      # max number of trajectory samples to be sampled per time step. 
    trajectory_iterations = 2408    # number of batches of episodes
    noptepochs = 12                 # Number of epochs per time step to optimize the neural networks
    learning_rate_p = 1e-4          # learning rate for policy network
    learning_rate_v = 1e-3          # learning rate for value network
    gae_lambda = 0.95               # trajectory discount for the general advantage estimation (GAE)
    gamma = 0.99                    # discount factor
    adam_epsilon = 1e-8             # default in the PPO baseline implementation is 1e-5, the pytorch default is 1e-8 - Andrychowicz, et al. (2021)  uses 0.9
    epsilon = 0.2                   # clipping factor
    env_name = 'Pendulum-v1'        # name of OpenAI gym environment other: 'Pendulum-v1' , 'MountainCarContinuous-v0'
    env_number = 1                  # number of actors
    seed = 42                       # seed gym, env, torch, numpy 
    
    # setup for torch save models and rendering
    render = True
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

    # create CSV writer
    csv_writer = CSVWriter(f"{LOG_PATH}{env_name}_{CURR_DATE}.csv")

    # Monitoring with W&B
    wandb.init(
            project=f'drone-mechanics-ppo-OpenAIGym',
            entity='drone-mechanics',
            sync_tensorboard=True,
            config={ # stores hyperparams in job
            'total number of steps': total_steps,
            'max sampled trajectories': max_trajectory_size,
            'batches per episode': trajectory_iterations,
            'number of epochs for update': noptepochs,
            'input layer size': obs_dim,
            'output layer size': act_dim,
            'learning rate (policy net)': learning_rate_p,
            'learning rate (value net)': learning_rate_v,
            'epsilon (adam optimizer)': adam_epsilon,
            'gamma (discount)': gamma,
            'epsilon (clipping)': epsilon,
            'gae lambda (GAE)': gae_lambda,
            'seed': seed
        },
            name=f"exp_name: {env_name}_{CURR_DATE}",
            monitor_gym=True,
            save_code=True
        )

    if args.train:
        logging.info('Training model...')
        train(env,
            in_dim=obs_dim, 
            out_dim=act_dim,
            total_steps=total_steps,
            max_trajectory_size=max_trajectory_size,
            trajectory_iterations=trajectory_iterations,
            noptepochs=noptepochs,
            learning_rate_p=learning_rate_p,
            learning_rate_v=learning_rate_v,
            gae_lambda = gae_lambda,
            gamma=gamma,
            epsilon=epsilon,
            adam_epsilon=adam_epsilon,
            render_steps=render_steps,
            save_steps=save_steps,
            csv_writer=csv_writer,
            log_video=args.video)
    
    elif args.test:
        logging.info('Evaluation model...')
        PATH = './models/Pendulum-v1_2023-01-01_policyNet.pth'
        test(PATH, env, in_dim=obs_dim, out_dim=act_dim)
    
    elif args.hyperparam:
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
        assert("Needs training (--train) or testing (--test) flag set!")

    logging.info('### Done ###')

    # cleanup 
    env.close()
    wandb.run.finish() if wandb and wandb.run else None