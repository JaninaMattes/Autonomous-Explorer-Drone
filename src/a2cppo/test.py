import os
import torch
from torch import nn
import sys
import gym

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

import argparse
from distutils.util import strtobool

# logging and monitoring
import logging
import wandb


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """ Initialize the hidden layers with orthogonal initialization
        Engstrom, Ilyas, et al., (2020)
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def load_model(path, model, device='cpu'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def simulate_rollout(policy_net, env, vid_render=True):
    frames=[]
    while True: 
        obs, done = env.reset(), False
        t_so_far, ep_lengths, ep_returns = 0, 0, 0
        while not done:
            t_so_far += 1
			# Query deterministic action from policy and run it
            action = policy_net(obs).detach().numpy()
            obs, reward, done, _ = env.step(action)

            if vid_render:
                frames.append(env.render(mode="rgb_array")) 
            
            ep_returns += reward
            ep_lengths = t_so_far

        yield ep_lengths, ep_returns, frames


def save_frames_as_gif(frames, path='./', filename='pendulum_v1.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    video_path = os.path.join(path, filename)
    anim.save(video_path, writer='imagemagick', fps=60)

def test(model_path, env, in_dim, out_dim, n_steps=10, device='cpu', log_video=True, filename='pendulum_v1.gif'):
    """Test the policy network (actor)"""
    logging.info('Evaluation model...')
    # split to base path
    base_path, _ = os.path.split(model_path)
    video_path = os.path.join(base_path, filename)

    # load model and test it
    policy_net = PolicyNet(in_dim, out_dim)
    policy_net = load_model(model_path, policy_net, device)
    
    for ep_num, (ep_lengths, ep_returns, frames) in enumerate(simulate_rollout(policy_net, env, log_video)):
        _log_summary(ep_len=ep_lengths, ep_ret=ep_returns, ep_num=ep_num)
        # save frames
        save_frames_as_gif(frames, base_path, filename)
        # log video
        if log_video:
            wandb.log({"test/video": wandb.Video(video_path, caption='episode: '+str(ep_num), fps=4, format="gif"), "step": ep_num})
        
        if ep_num == n_steps:
            break
    
    # cleanup
    env.close()
    wandb.run.finish() if wandb and wandb.run else None
    
    
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

def arg_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False, help="if toggled, capture video of run")
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, run model in training mode")
    parser.add_argument("--test", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False, help="if toggled, run model in testing mode")
    parser.add_argument("--hyperparam", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, log hyperparameters")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--model-path", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of the model path.")
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

def make_env(env_id='Pendulum-v1', gym_wrappers=False, gym_monitor=False, monitor_path='./', seed=42):
    # TODO: Needs to be parallized for parallel simulation
    env = gym.make(env_id)
    
    # gym wrapper
    if gym_wrappers:
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    if gym_monitor:
        env = gym.wrappers.Monitor(env, monitor_path, force = True)
    # seed env for reproducability
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

if __name__ == '__main__':

        # Hyperparameter
    total_training_steps = 3_000_000     # time steps regarding batches collected and train agent
    batch_size = 512                     # max number of episode samples to be sampled per time step. 
    n_rollout_steps = 2048               # number of batches per episode, or experiences to collect per environment
    n_optepochs = 31                     # Number of epochs per time step to optimize the neural networks
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
    normalize_adv = True                 # wether to normalize the advantage estimate
    normalize_ret = False                # wether to normalize the return function
    
    # setup for torch save models and rendering
    render = False
    render_steps = 10
    save_steps = 100

    # parse arguments
    args = arg_parser()

    # Configure logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    base_path, _ = os.path.split(args.model_path)

    # seed gym, torch and numpy
    env = make_env(env_name, gym_monitor=True, monitor_path=base_path, seed=seed)
    
    # seed
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
                'experiment model path': args.model_path,
                'experiment name': args.exp_name
            },
            dir=os.getcwd(),
            name=args.exp_name, # needs flag --exp-name
            monitor_gym=True,
            save_code=True
        )

    test(args.model_path, env, in_dim=obs_dim, out_dim=act_dim, n_steps=10, device=device)