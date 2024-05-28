import logging
import numpy as np
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



# Set up logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

# Parallel environments
env = make_vec_env("Pendulum-v0", n_envs=1)

# Initialize PPO agent
agent = PPO("MlpPolicy", env, verbose=1)

# Set up wandb
wandb.init(project="PPO-Baseline", entity="your-entity-name")


# Define the learn function
def learn(agent, total_training_steps):
    """
    Train the PPO agent for a specified number of timesteps.

    Args:
    agent (PPO): The PPO agent to train.
    total_training_steps (int): The total number of timesteps to train for.
    """
    for training_steps in range(1, total_training_steps + 1):
        # Collect data over one episode
        obs, next_obs, actions, batch_log_probs, dones, rewards, ep_lens, ep_time = agent.collect_rollout(
            n_steps=agent.n_rollout_steps, render=agent.render)

        # experiences simulated so far
        training_steps += np.sum(ep_lens)

        # STEP 4-5: Calculate cummulated reward and advantage at timestep t_step
        values, _, _ = agent.get_values(obs, actions)
        advantages, cum_returns = agent.generalized_advantage_estimate(rewards, values.detach(),
                                                                         normalized_adv=agent.normalize_advantage,
                                                                         normalized_ret=agent.normalize_return)

        # update network params
        for _ in range(agent.noptepochs):
            # STEP 6-7: calculate loss and update weights
            values, curr_log_probs, _ = agent.get_values(obs, actions)
            policy_loss, value_loss = agent.train(values, cum_returns, advantages, batch_log_probs, curr_log_probs,
                                                   agent.epsilon)

            # Log policy and value losses to wandb
            wandb.log({
                'train/policy_loss': policy_loss,
                'train/value_loss': value_loss
            })

        # Log statistics to wandb
        wandb.log({
            'train/timesteps': training_steps,
            'train/mean_episode_return': np.mean(cum_returns),
            'train/mean_episode_length': np.mean(ep_lens),
            'train/mean_episode_time': np.mean(ep_time)
        })

        # Log statistics to console
        logging.info(f'Episode: {training_steps}')
        logging.info(f'Mean episode return: {np.mean(cum_returns):.2f}')
        logging.info(f'Mean episode length: {np.mean(ep_lens):.2f}')
        logging.info(f'Mean episode time: {np.mean(ep_time):.2f}')

    # Save the final model
    agent.save("ppo_baseline")

# Train the agent
learn(agent, total_training_steps=10000)
