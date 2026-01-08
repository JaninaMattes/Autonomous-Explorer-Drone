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
import gym
import os

import argparse

# logging python
import logging
import sys

# monitoring/logging ML
import wandb

MODEL_PATH = './models/'

####################
####### TODO #######
####################

# Hint: Please if working on it mark a todo as (done) if done
# 1) Check current implementation against article: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# 2) Check calculation of rewards --> correct mean reward over episodes? 
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
        self.clip_fraction = (abs((ratio - 1.0)) > clip_eps).to(torch.float).mean()
        return policy_loss


####################
####################


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

    def cummulative_reward(self, batch_rewards):
        """Calculate cummulative rewards with discount factor gamma."""
        # Cumulative rewards: https://gongybable.medium.com/reinforcement-learning-introduction-609040c8be36
        # advantage function: δt = G(t) + γV (st+1) − V (st)
        # return value: G(t) = R(t) + gamma * R(t-1)
        cum_rewards = []
        for rewards in reversed(batch_rewards): # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_rewards.insert(0, discounted_reward)
        return torch.tensor(cum_rewards, dtype=torch.float)

    def generalized_advantage_estimate(self, batch_rewards, values):
        """ Calculate advantage as a weighted average of A_t
            - advantage A_t gives information if this action is bettern than another at a state
        """
        # general advantage estimage: https://nn.labml.ai/rl/ppo/gae.html
        last_advantage = 0
        advantages = []
        last_value = values[-1] # V(s_t+1)
        for i in reversed(range(self.max_trajectory_size)):

            if i == self.max_trajectory_size -1:
                nextnonterminal = 1.0 - self.dones
            else:
                pass

    def advantage_estimate(self, returns, values, normalized=True):
        """ Advantage calculation
            - advantage A_t gives information if this action is bettern than another at a state
        """
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
        
        step, ep_rewards, ep_entropy = 0, [], []

        # collect trajectories
        trajectory_obs = []
        trajectory_actions = []
        trajectory_action_probs = []
        batch_entropy = []
        batch_rewards = []
        batch_lens = []

        # Run Monte Carlo simulation for n timesteps per batch
        logging.info("Collecting batch trajectories...")
        while step < n_step:
            
            # rewards collected per episode
            ep_rewards, ep_entropy, done = [], [], False 
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
                action, log_probability, entropy = self.step(obs)
                        
                # STEP 3: collecting set of trajectories D_k by running action 
                # that was sampled from policy in environment
                __obs, reward, done, _ = self.env.step(action)

                # collection of trajectories in batches
                trajectory_obs.append(obs)
                trajectory_actions.append(action)
                trajectory_action_probs.append(log_probability)
                ep_entropy.append(entropy.item())
                ep_rewards.append(reward)
                    
                obs = __obs

                # break out of loop if episode is terminated
                if done:
                    break
            
            batch_lens.append(ep_t + 1) # as we started at 0
            batch_rewards.append(ep_rewards)
            batch_entropy.append(ep_entropy)

        # convert trajectories to torch tensors
        obs = torch.tensor(np.array(trajectory_obs), dtype=torch.float)
        actions = torch.tensor(np.array(trajectory_actions), dtype=torch.float)
        log_probs = torch.tensor(np.array(trajectory_action_probs), dtype=torch.float)
        
        # STEP 4: Calculate cummulated reward
        cummulative_reward = self.cummulative_reward(batch_rewards)
        cummulative_reward = torch.tensor(np.array(cummulative_reward), dtype=torch.float)

        # Calculate the stats
        cum_rews = [np.sum(ep_rews) for ep_rews in batch_rewards]
        mean_ep_lens = np.mean(batch_lens)
        mean_ep_rews = np.mean(cum_rews)
        std_ep_rews = np.std(cum_rews) # calculate standard deviation (spred of distribution)
        mean_ep_entropy = np.mean(np.sum(ep_entr) for ep_entr in batch_entropy)
        
        # Log stats
        wandb.log({
            "train/mean episode length": mean_ep_lens,
            "train/mean episode returns": mean_ep_rews,
            "train/std episode returns": std_ep_rews,
            "train/mean episode entropy": mean_ep_entropy, # should decrease over time - agent is more "sure"
        })

        return obs, actions, log_probs, cummulative_reward, batch_lens, mean_ep_rews
                

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
            obs, actions, log_probs, cum_return, batch_lens, mean_reward = self.collect_rollout(n_step=self.trajectory_iterations)
            
            # timesteps simulated so far for batch collection
            steps += np.sum(batch_lens)

            # STEP 5: compute advantage estimates A_t at timestep t_step
            values, _ , _ = self.get_values(obs, actions)
            advantages = self.advantage_estimate(cum_return, values.detach())

            # update network params 
            for _ in range(self.noptepochs):
                # STEP 6-7: calculate loss and update weights
                values, curr_log_probs, entropy = self.get_values(obs, actions)
                entropy_bonus = np.mean(entropy)
                policy_loss, value_loss = self.train(values, cum_return, advantages, log_probs, curr_log_probs, self.epsilon)

            logging.info('\n')
            logging.info('###########################################')
            logging.info(f"Mean return: {mean_reward}")
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
                env_name = env.unwrapped.spec.id
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
                best_mean_reward = mean_reward

####################
####################

def arg_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser = argparse.ArgumentParser()
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

def train():
    # TODO Add Checkpoints to load model 
    pass

def test():
    pass

if __name__ == '__main__':
    
    """ Classic control gym environments 
        Find docu: https://www.gymlibrary.dev/environments/classic_control/
    """
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    args = arg_parser()
    
    # Hyperparameter
    unity_file_name = ''            # name of unity environment
    total_steps = 30000000          # time steps to train agent
    max_trajectory_size = 1000      # max number of trajectory samples to be sampled per time step. 
    trajectory_iterations = 4600    # number of batches of episodes
    noptepochs = 5                  # Number of epochs per time step to optimize the neural networks
    learning_rate_p = 1e-4          # learning rate for policy network
    learning_rate_v = 1e-3          # learning rate for value network
    gae_lambda = 0.95               # lambda coef for the general advantage estimation
    gamma = 0.99                    # discount factor
    adam_epsilon = 1e-8             # default in the PPO baseline implementation is 1e-5, the pytorch default is 1e-8 - Andrychowicz, et al. (2021)  uses 0.9
    epsilon = 0.2                   # clipping factor
    env_name = 'Pendulum-v1'        # name of OpenAI gym environment other: 'Pendulum-v1' , 'MountainCarContinuous-v0'
    env_number = 8                  # number of actors
    seed = 42                       # seed gym, env, torch, numpy 
    
    # setup for torch save models and rendering
    render_steps = 10
    save_steps = 10

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

    logging.info(f'env observation space: {obs_shape}')
    logging.info(f'env action space: {act_shape}')
    
    obs_dim = obs_shape[0] 
    act_dim = act_shape[0] # 2 at CartPole

    logging.info(f'env observation dim: {obs_dim}')
    logging.info(f'env action dim: {act_dim}')
    
    # upper and lower bound describing the values our obs can take
    logging.info(f'upper bound for env observation: {env.observation_space.high}')
    logging.info(f'lower bound for env observation: {env.observation_space.low}')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   
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
            'seerd': seed
        },
    name=f"{env_name}__{current_time}",
    monitor_gym=True,
    save_code=True,
    )

    agent = PPO_PolicyGradient(
                env, 
                in_dim=obs_dim, 
                out_dim=act_dim,
                total_steps=total_steps,
                max_trajectory_size=max_trajectory_size,
                trajectory_iterations=trajectory_iterations,
                noptepochs=noptepochs,
                lr_p=learning_rate_p,
                lr_v=learning_rate_v,
                gamma=gamma,
                epsilon=epsilon,
                adam_eps=adam_epsilon,
                render=render_steps,
                save_model=save_steps)
    
    # run training for a total amount of steps
    agent.learn()
    logging.info('### Done ###')

    # cleanup 
    env.close()
    wandb.run.finish() if wandb and wandb.run else None