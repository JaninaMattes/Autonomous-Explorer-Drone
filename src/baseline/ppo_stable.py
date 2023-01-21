import gym

# requires gym==0.21.0 & pyglet==1.5.27
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
import datetime


# TODO: Hyperparameters should be the same in baseline and our own implementation

# # Parameters
# num_total_steps = 25e3          # Total number of time steps to run the training
# learning_rate_policy = 1e-3     # Learning rate for optimizing the neural networks
# learning_rate_value = 1e-3
# num_epochs = 5                  # Number of epochs per time step to optimize the neural networks
# epsilon = 0.2                   # Epsilon value in the PPO algorithm
# max_trajectory_size = 10000     # max number of trajectory samples to be sampled per time step. 
# trajectory_iterations = 16      # number of batches of episodes
# input_length_net = 4            # input layer size
# policy_output_size = 2          # policy output layer size
# discount_factor = 0.99
# env_name = "Pendulum-v1"        # LunarLander-v2 or MountainCar-v0 or CartPole-v1 or Pendulum-v1


##### Parameters baseline for PPO based on ppo.tf.py implementation
policy = "MlpPolicy"            # The policy model to use (MlpPolicy, CnnPolicy, …)
env = "Pendulum-v1"             # The environment to learn from (if registered in Gym, can be str)
n_envs = 1                      # amount of envs used simultaneously
learning_rate = 1e-3            # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
n_steps = 25e3                  # The number of steps to run for each environment per update (greater than 1)
batch_size = 10                 # Minibatch size
n_epochs = 5                    # Number of epoch when optimizing the surrogate loss
gamma = 0.99                    # Discount factor
#gae_lambda =                   # Factor for trade-off of bias vs variance for
#clip_range =                   # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0)
#clip_range_vf =                # Clipping parameter for the value function, it can be a function of the current progress remaining (from 1 to 0)
#normalize_advantage            # (bool) – Whether to normalize or not the advantage
#ent_coef =                     # Entropy coefficient for the loss calculation
#vf_coef =                      # Value function coefficient for the loss calculation
#tensorboard_log =              # The log location for tensorboard
#verbose =                      # Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
seed = 42                       # Seed for the pseudo random generators
total_timesteps = 25e3          # The total number of samples (env steps) to train on


##### Parameters for Weights & Biases
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

wandb.init(
    project=f'drone-mechanics-ppo',
    entity='drone-mechanics',
    sync_tensorboard=True,
    config={ # stores hyperparams in job
        "policy" : policy,
        "env": env,
        "n_envs": n_envs,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "seed": seed,
        "total_timesteps": total_timesteps
    },
    name=f"{env}__{current_time}",
    # monitor_gym=True,
    save_code=True,
)

# Parallel environments
env = make_vec_env(env, n_envs=n_envs) # TODO: @Ardian Check stable_baseline3 library 

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    gamma=gamma,
    # Using https://proceedings.mlr.press/v164/raffin22a.html
    use_sde=True,
    sde_sample_freq=4,
    learning_rate=learning_rate,
    verbose=1,
    tensorboard_log=f"runs/{run.id}"
)

# Train the agent
model.learn(
    total_timesteps=int(total_timesteps),
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

    # TODO: Define values to track --> look into PPO
    # TODO: Track the value in W&B (ppo_tf.py is already implemented) @Ardian
    # TODO: What is the maximum reward we can reach --> 500 

    # Log into tensorboard & Wandb
    wandb.log({
        'time steps': num_passed_timesteps, 
        'policy loss': policy_loss, 
        'value loss': value_loss, 
        'mean return': mean_return})
