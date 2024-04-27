import numpy as np

# logging
import wandb

# stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


class TrainingCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.n_ep_rewards = []
        self.max_batch_size = 512
        self.n_rollout_steps = 2048
        self.t_episode = 0
        self.step = 0
        self.mean_rewards = 0
        self.over_flag = False
    
    def _on_step(self):
        if(self.locals.get('n_steps')!= self.n_rollout_steps-1):
            if(self.locals.get('dones')):
                self.n_ep_rewards.append(self.locals.get('infos')[0].get('episode').get('r'))
                if(self.over_flag):
                    self.mean_rewards = np.mean(self.n_ep_rewards)
                    print('------', self.step)
                    print(self.mean_rewards)
                    wandb.log({
                        'mean_episodes_rewards': self.mean_rewards,

                    }, step=self.step)
                    self.n_ep_rewards = []
                    self.over_flag = False
                    
        elif(self.locals.get('dones')):
            self.n_ep_rewards.append(self.locals.get('infos')[0].get('episode').get('r'))
            self.mean_rewards = np.mean(self.n_ep_rewards)
            print('------', self.step)
            print(self.mean_rewards)
            wandb.log({
                'mean_episodes_rewards': self.mean_rewards,

            }, step=self.step)
            self.n_ep_rewards = []
        else:
            self.over_flag = True

        # print(self.locals)
        self.step += 1
        return True  # returns True, training continues.


tmp_path = "tmp/"
# set up logger
new_logger = configure(tmp_path, ["csv"])

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "Pendulum-v1",
    "n_steps": 2048,
}

run = wandb.init(
    project="log_test_zhengjie",
    config=config,
    entity='drone-mechanics',
    name='stable_baselines3'
)

model = PPO(config["policy_type"], config["env_name"], n_steps=config["n_steps"], verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=config["total_timesteps"], callback=TrainingCallback())

run.finish()

