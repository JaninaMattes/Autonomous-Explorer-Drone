"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import gym
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import ray
from ray.tune import register_env
# from ray.rllib.agents import ppo

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

# import own modules
import gym_pybullet_drones.examples.ppo as ppo

DEFAULT_ENV = 'takeoff' #"takeoff", "hover"
DEFAULT_RLLIB = True
DEFAULT_ALGO = 'ppo_v2'
DEFAULT_GUI = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_SEED = 42

def make_env(env_id, seed=42):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def run(env_id=DEFAULT_ENV, 
        rllib=DEFAULT_RLLIB, 
        algo=DEFAULT_ALGO, 
        output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI,
        plot=True, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        seed=DEFAULT_SEED):
    
    #####################
    #### Check the environment's spaces ########################
    #####################
    if not env_id in ['takeoff', 'hover']: 
        print("[ERROR] 1D action space is only compatible with Takeoff and HoverAviary")
        exit()

    env_id = env_id + "-aviary-v0"
    env = make_env(env_id, seed=seed)
    
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    
    check_env(env, warn=True, skip_render_check=True)

    print(env.action_space.sample())

    ############################################################
    #### Train the model #######################################
    ############################################################

    if algo == 'ppo_sb3':
        # stable baseline3
        model = A2C(MlpPolicy,
                    env,
                    verbose=1
                    )
        model.learn(total_timesteps=10_000) # Typically not enough
    
    elif algo == 'ppo_v2':
        # custom ppo-v2
        # get PPOTrainer
        trainer = ppo.PPOTrainer(
                    env, 
                    total_training_steps=1_000_000, # 1_000_000, shorter just for testing
                    seed=seed) 
        # train PPO
        agent = trainer.create_ppo()
        agent.learn()
        # get trained policy
        policy = trainer.get_policy()
        # cleanup
        trainer.shutdown()

    elif algo == 'ppo_raylib':
        # use ray-lib ppo
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        register_env(env_id, lambda _: TakeoffAviary())
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_workers"] = 2
        config["framework"] = "torch"
        config["env"] = env_id
        agent = ppo.PPOTrainer(config)
        for i in range(3): # Typically not enough
            results = agent.train()
            print("[INFO] {:d}: episode_reward max {:f} min {:f} mean {:f}".format(i,
                                                                                   results["episode_reward_max"],
                                                                                   results["episode_reward_min"],
                                                                                   results["episode_reward_mean"]
                                                                                   )
            )
        policy = agent.get_policy()
        ray.shutdown()

    ############################################################
    #### Show (and record a video of) the model's performance ##
    ############################################################

    if env_id == 'takeoff':
        env = TakeoffAviary(gui=gui,
                            record=record_video
                            )
    elif env_id == 'hover':
        env = HoverAviary(gui=gui,
                            record=record_video
                            )
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )
    obs = env.reset()
    start = time.time()
    for i in range(30000*env.SIM_FREQ):
        if not rllib:
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
        else:
            action = policy(obs).detach().numpy()
        
        obs, reward, done, info = env.step(action)
        print(f'reward {reward}')
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()

    if plot:
        logger.plot()


#######################################
#######################################


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--rllib',              default=DEFAULT_RLLIB,          type=str2bool,       help='Whether to use RLlib PPO in place of stable-baselines A2C (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,            type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,   type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,  type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--algo',               default=DEFAULT_ALGO,           type=str,           help='Select an algorithm to be used, either custom ppo or stable-baseline3 (ppo_v2, ppo_sb3)')
    parser.add_argument('--env_id',             default=DEFAULT_ENV,            type=str,           help='Select an environment to train on (hover, takeoff)')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,          type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
