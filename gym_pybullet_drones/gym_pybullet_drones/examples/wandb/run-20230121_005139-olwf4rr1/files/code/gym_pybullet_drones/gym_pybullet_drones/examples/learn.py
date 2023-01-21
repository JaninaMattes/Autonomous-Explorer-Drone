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
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
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
from gym_pybullet_drones.examples.ppo import PPOTrainer

#######################################
#######################################

DEFAULT_VISION = False
DEFAULT_ENV = 'takeoff' #"takeoff", "hover", 'flythrugate'
DEFAULT_RLLIB = True
DEFAULT_ALGO = 'ppo_v2'
# drones
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
# simulation of env
DEFAULT_AGGREGATE = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
# gui and logging
DEFAULT_GUI = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_RECORD_VIDEO = True
DEFAULT_PLOT = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_SEED = 42
DEFAULT_TRAINING_STEPS = 100_000 # just for testing, too short otherwise

#######################################
#######################################


def run(env_id=DEFAULT_ENV, 
        rllib=DEFAULT_RLLIB, 
        algo=DEFAULT_ALGO,
        train_steps=DEFAULT_TRAINING_STEPS,
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        aggregate=DEFAULT_AGGREGATE,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        seed=DEFAULT_SEED):
    
    #############################################################
    #### Initialize spaces ########################
    #############################################################

    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1


    #############################################################
    #### Check the environment's spaces ########################
    #############################################################
    
    if not env_id in ['takeoff', 'hover', 'flythrugate']: 
        print("[ERROR] 1D action space is only compatible with Takeoff- and HoverAviary")
        exit()

    _env_id = env_id + "-aviary-v0"
    print("[INFO] You selected env:", env_id)
    # make env
    env = make_env(_env_id, seed=seed)
    
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
        model.learn(total_timesteps=train_steps)
    
    elif algo == 'ppo_v2':
        # custom ppo-v2
        # create PPOTrainer
        trainer = PPOTrainer(
                    env, 
                    total_training_steps=train_steps, # shorter just for testing
                    n_optepochs=64,
                    epsilon=0.22,
                    gae_lambda=0.95,
                    gamma=0.99,
                    adam_eps=1e-7,
                    seed=seed) 
        # train PPO
        agent = trainer.create_ppo()
        agent.learn()
        # get trained policy
        policy = trainer.get_policy()
        # cleanup
        trainer.shutdown()

    elif algo == 'ppo_rllib':
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
                    results["episode_reward_mean"])
                )
        policy = agent.get_policy()
        ray.shutdown()

    ############################################################
    #### Show (and record a video of) the model's performance ##
    ############################################################

    if env_id == 'takeoff':
        env = TakeoffAviary(drone_model=drone,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            freq=simulation_freq_hz,
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            gui=gui,
                            physics=physics,
                            record=record_video
                        )
    elif env_id == 'hover':
        env = HoverAviary(drone_model=drone,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            freq=simulation_freq_hz,
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            gui=gui,
                            physics=physics,
                            record=record_video
                        )
    elif env_id == 'flythrugate':
        env = FlyThruGateAviary(drone_model=drone,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            freq=simulation_freq_hz,
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            gui=gui,
                            physics=physics,
                            record=record_video
                        )
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    obs = env.reset()
    start = time.time()
    for i in range(30000*env.SIM_FREQ):
        
        # query policy for action
        if not rllib and algo == 'ppo_sb3':
            action, _states = model.predict(obs, deterministic=True)
        elif algo == 'ppo_v2':
            action = policy(obs).detach().numpy()
        
        # update environment
        obs, reward, done, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)

            if vision:
                for j in range(num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"]))
        
        #### Sync the simulation ###################################
        if gui:
            sync(i, start, env.TIMESTEP)
        
        if done:
            obs = env.reset()
    
    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


#######################################
#######################################

def make_env(env_id, seed=42):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

#######################################
#######################################


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary or HoverAviary')
    parser.add_argument('--rllib',              default=DEFAULT_RLLIB,              type=str2bool,       help='Whether to use RLlib PPO in place of stable-baselines A2C (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,                type=str2bool,       help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,       type=str2bool,       help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,      type=str,            help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--algo',               default=DEFAULT_ALGO,               type=str,            help='Select an algorithm to be used, either custom ppo or stable-baseline3 (ppo_v2, ppo_sb3, ppo_rllib)')
    parser.add_argument('--env_id',             default=DEFAULT_ENV,                type=str,            help='Select an environment to train on (hover, takeoff, flythrugate)')
    parser.add_argument('--train_steps',        default=DEFAULT_TRAINING_STEPS,     type=int,            help='Select the amount of training steps')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,              type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
