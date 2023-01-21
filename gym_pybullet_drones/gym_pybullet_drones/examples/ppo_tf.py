import datetime
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
import tensorflow_probability as tfp
import numpy as np
# from mlagents_envs.environment import UnityEnvironment
import gym # requires gym==0.21.0
import wandb

tfd = tfp.distributions
#
#
#
#
#
#
#

# Parameters
unity_file_name = ""            # Unity environment name
num_total_steps = 1          # Total number of epochs to run the training
learning_rate_policy = 3e-4     # Learning rate for optimizing the neural networks
learning_rate_value = 1e-3
num_epochs = 20                 # Number of epochs per time step to optimize the neural networks
epsilon = 0.2                   # Epsilon value in the PPO algorithm
max_trajectory_size = 1000     # max number of trajectory samples to be sampled per time step. 
trajectory_iterations = 10      # number of batches of episodes
input_length_net = 12            # input layer size
policy_output_size = 4          # policy output layer size
discount_factor = 0.99
env_name = "takeoff-aviary-v0"     # LunarLander-v2 or MountainCar-v0 or CartPole-v1 or Pendulum-v1
continous = True                # Whether the action space is cont.

print(f"Tensorflow version: {tf.__version__}")

#
#
#
#
#
#
#

# Define the policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = Flatten()
        self.dense1 = Dense(units=input_length_net, activation='tanh')
        self.dense2 = Dense(units=64, activation='tanh')
        self.dense3 = Dense(units=64, activation='tanh')
        self.out = Dense(units=policy_output_size, activation='softmax') # 'linear' if the action space is continous
        if continous:
            self.out = Dense(units=policy_output_size, activation='tanh')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)

# Define the value network
class ValueNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = Flatten()
        self.dense1 = Dense(units=input_length_net, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=64, activation='relu')
        self.out = Dense(units=1, activation='linear')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)

#
#
#
#
#
#
#

# Setup the actor/critic networks
policy_net = PolicyNetwork()
value_net = ValueNetwork()
#policy_net = tf.keras.models.load_model(f"{env_name}_policy_model")
#value_net = tf.keras.models.load_model(f"{env_name}_value_model")

#
#
#
#
#
#
#

# This is a non-blocking call that only loads the environment.
#env = UnityEnvironment(file_name=unity_file_name, seed=42, side_channels=[])
# Start interacting with the environment.a
#env.reset()
#behavior_names = env.behavior_specs.keys()

#
#
#
#
#
#
#

# Setup training properties
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)

env = gym.make(env_name)
env.observation_space.seed(888)
observation = env.reset()
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space shape: {env.action_space.shape}")
print(env.action_space)

#
#
#
#
#
#
#

# Training loop
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logging
train_log_dir = f'logs/gradient_tape/{env_name}' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
wandb.init(
    project=f'drone-mechanics-ppo',
    entity='drone-mechanics',
    sync_tensorboard=True,
    config={ # stores hyperparams in job
            'total epochs': num_epochs,
            'total steps': num_total_steps,
            'batches per episode': trajectory_iterations,
            'input layer size': input_length_net,
            'output layer size': policy_output_size,
            'lr policyNet': learning_rate_policy,
            'lr valueNet': learning_rate_value,
            'epsilon': epsilon,
            'discount': discount_factor
    },
    name=f"{env_name}__{current_time}",
    # monitor_gym=True,
    save_code=True,
)



def train():
    '''
    Main training loop.

    The agent is trained for @num_total_steps times.
    '''

    num_passed_timesteps = 0
    sum_rewards = 0
    num_episodes = 1
    last_mean_reward = -2000

    for epochs in range(num_total_steps):

        trajectory_observations = []
        trajectory_rewards = []
        trajectory_action_probs = []
        trajectory_advantages = np.array([])
        trajectory_actions = []
        values = []
        total_reward = 0
        observation = env.reset()

        total_reward = 0
        mean_return = 0
        num_episodes = 0

        '''
            The trajectories are collected in batches and will be saved to memory.
            The information is used for training the policy and value networks.
        '''
        for iter in range(trajectory_iterations):
            while True:
                trajectory_observations.append(observation)

                # Sample action of the agent
                current_action_prob = policy_net(observation.reshape(1,input_length_net))
                current_action_dist = tfd.Categorical(probs=current_action_prob)
                if continous:
                    action_std = tf.ones_like(current_action_prob)
                    current_action_dist = tfd.MultivariateNormalDiag(current_action_prob, action_std)
                    
                current_action = current_action_dist.sample(seed=42).numpy()[0]
                trajectory_actions.append(current_action)

                # Sample new state from environment with the current action
                #print(env.step(current_action))
                observation, reward, terminated, info = env.step(current_action)
                env.render()
                num_passed_timesteps += 1
                sum_rewards += reward
                total_reward += reward

                # Collect trajectory sample
                trajectory_rewards.append(reward)
                trajectory_action_probs.append(current_action_dist.prob(current_action))
                value = value_net(observation.reshape((1,input_length_net)))
                values.append(value)
                    
                if terminated:
                    observation = env.reset()

                    # Compute advantages at the end of the trajectory
                    new_adv = np.array(total_reward, dtype=np.float32) - np.array(values, dtype=np.float32)
                    new_adv = np.squeeze(new_adv)
                    trajectory_advantages = np.append(trajectory_advantages, new_adv)
                    trajectory_advantages = trajectory_advantages.flatten()

                    num_episodes += 1
                    total_reward = 0
                    values = []
                    print("done")
                    break

        # Compute the mean cumulative reward.
        mean_return = sum_rewards / num_episodes
        sum_rewards = 0
        print(f"Mean cumulative reward: {mean_return}", flush=True)

        trajectory_observations  = tf.constant(np.array(trajectory_observations), dtype=tf.float32)
        trajectory_action_probs  = tf.squeeze(tf.constant(np.array(trajectory_action_probs), dtype=tf.float32))
        trajectory_rewards       = tf.constant(np.array(trajectory_rewards), dtype=tf.float32)

        # Normalize advantages
        trajectory_advantages    = tf.constant(trajectory_advantages, dtype=tf.float32)
        trajectory_advantages    = tf.squeeze(trajectory_advantages)
        trajectory_advantages    = (trajectory_advantages - np.mean(trajectory_advantages)) / (np.std(trajectory_advantages) + 1e-8)

        # Update the network loop
        for epoch in range(num_epochs):

            with tf.GradientTape() as policy_tape:
                policy_dist             = policy_net(trajectory_observations)
                dist                    = tfd.Categorical(probs=policy_dist)
                if continous:
                    action_std = tf.ones_like(policy_dist)
                    dist = tfd.MultivariateNormalDiag(policy_dist, action_std)
                
                policy_action_prob      = dist.prob(trajectory_actions)
                
                # Policy loss update
                ratios                  = tf.divide(policy_action_prob, trajectory_action_probs)
                clip_1                  = tf.multiply(ratios, trajectory_advantages)
                clip                    = tf.clip_by_value(ratios, 1.0 - epsilon, 1.0 + epsilon)
                clip_2                  = tf.multiply(clip, trajectory_advantages)
                min                     = tf.minimum(clip_1, clip_2)
                policy_loss             = tf.math.negative(tf.reduce_mean(min))

            policy_gradients = policy_tape.gradient(policy_loss, policy_net.trainable_variables)
            policy_optimizer.apply_gradients(zip(policy_gradients, policy_net.trainable_variables))

            with tf.GradientTape() as value_tape:
                value_out  = tf.squeeze(value_net(trajectory_observations))
                # Value loss update
                value_loss = tf.keras.losses.MSE(value_out, trajectory_advantages)
                
            value_gradients = value_tape.gradient(value_loss, value_net.trainable_variables)
            value_optimizer.apply_gradients(zip(value_gradients, value_net.trainable_variables))

        print(f"Epoch: {epoch}, Policy loss: {policy_loss}", flush=True)
        print(f"Epoch: {epoch}, Value loss: {value_loss}", flush=True)
        print(f"Total time steps: {num_passed_timesteps}", flush=True)

        # Make sure the best model is saved.
        if mean_return > last_mean_reward:
            # Save the policy and value networks for further training/tests
            policy_net.save(f"{env_name}_policy_model")
            value_net.save(f"{env_name}_value_model")
            last_mean_reward = mean_return

        # Log into tensorboard & Wandb
        wandb.log({
            'time steps': num_passed_timesteps, 
            'policy loss': policy_loss, 
            'value loss': value_loss, 
            'mean return': mean_return})

        with train_summary_writer.as_default():
            tf.summary.scalar('policy loss', policy_loss, step=num_passed_timesteps)
            tf.summary.scalar('value loss', value_loss, step=num_passed_timesteps)
            tf.summary.scalar('mean return', mean_return, step=num_passed_timesteps)
    
    env.close()
    wandb.run.finish() if wandb and wandb.run else None

    # Save the policy and value networks for further training/tests
    policy_net.save(f"{env_name}_policy_model_{num_passed_timesteps}")
    value_net.save(f"{env_name}_value_model_{num_passed_timesteps}")