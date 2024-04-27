import gym
from stable_baselines3 import PPO

import numpy as np
import torch
import wandb
import os

from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("Pendulum-v0", n_envs=1)
agent = PPO("MlpPolicy", env, verbose=1)

def learn(self):
        """"""
        
        for training_steps in range(0, self.total_training_steps):
            policy_losses, value_losses = [], []

            # Collect data over one episode
            obs, next_obs, actions, batch_log_probs, dones, rewards, ep_lens, ep_time = self.collect_rollout(n_steps=self.n_rollout_steps, render=self.render)

            # experiences simulated so far
            training_steps += np.sum(ep_lens)

            # STEP 4-5: Calculate cummulated reward and advantage at timestep t_step
            values, _ , _ = self.get_values(obs, actions)
            advantages, cum_returns = self.generalized_advantage_estimate(rewards, values.detach(), normalized_adv=self.normalize_advantage, normalized_ret=self.normalize_return)
            
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
            self.stats_plotter.plot_seaborn_fill(df, x='timestep', y='mean episodic returns', 
                                                y_min='min episodic returns', y_max='max episodic returns',  
                                                title=f'{env_name}', x_label='Timestep', y_label='Mean Episodic Return', 
                                                color='blue', smoothing=6, wandb=wandb, xlim_up=self.total_training_steps)

            # self.stats_plotter.plot_box(df, x='timestep', y='mean episodic runtime', 
            #                             title='title', x_label='Timestep', y_label='Mean Episodic Time', wandb=wandb)

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