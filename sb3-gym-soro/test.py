"""Test script using sb3-gym-template

  Examples:
    
    [Trunk]     python test.py --test_env trunk-v0 --test_episodes 1 --seed 0 --algo ppo --offline --load_path ./example-results/trunk/RFDROPO/2023_02_28_20_31_32_trunk-v0_ppo_t2000000_seed2_login027851592_TM84F --test_render
    [TrunkCube] python test.py --test_env trunkcube-v0 --test_episodes 1 --seed 0 --algo ppo --offline --load_path ./example-results/trunkcube/RFDROPO/2023_07_10_11_34_58_trunkcube-v0_ppo_t2000000_seed1_7901a3c94a22_G0QXG --test_render
    [TrunkWall] python test.py --test_env trunkwall-v0 --test_episodes 1 --seed 0 --algo ppo --offline --load_path ./example-results/trunkwall/2023_02_26_20_46_59_trunkwall-v0_ppo_t2000000_seed3_mn011935323_R922D --test_render
    [Multigait] python test.py --test_env multigaitrobot-v0 --test_episodes 1 --seed 0 --algo ppo --offline --load_path ./example-results/multigait/2023_02_07_08_37_02_multigaitrobot-v0_ppo_t341000_seed1_hactarlogin358482_X54NP --test_render


"""
from pprint import pprint
import argparse
import pdb
import sys
import socket
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym
import torch
import wandb
from stable_baselines3.common.env_util import make_vec_env

# import random_envs
from envs.RandomVecEnv import RandomSubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.utils import *
from policy.policy import Policy

from sofagym import *

def main():

    assert args.test_env is not None and args.load_path is not None
    if args.test_render:
        assert args.now == 1 # visualize the render is meaningful only with one worker

    pprint(vars(args))
    set_seed(args.seed)

    random_string = get_random_string(5)

    wandb.init(config=vars(args),
             project="SoRo-RL",
             group=(args.test_env+'_test' if args.group is None else args.group),
             #name='SoRo'+('InfOnly' if args.inference_only else '')+'_'+args.algo+'_seed'+str(args.seed)+'_'+random_string,
             name=args.algo+'_seed'+str(args.seed)+'_'+random_string,
             save_code=True,
             tags=None,
             notes=args.notes,
             # mode=('online' if not args.offline else 'disabled'))
             mode='disabled')

    run_path = args.load_path+"/test/"+get_current_date()+"_"+socket.gethostname()+"_"+random_string+"/"
    create_dirs(run_path)

    test_env = gym.make(args.test_env)
    test_env.config["save_path"] = run_path

    wandb.config.path = run_path
    wandb.config.hostname = socket.gethostname()

    if args.test_rand_dynamics:
        assert args.bounds_path is not None and args.distribution_type is not None
        print("\nLoading bounds from ", args.bounds_path)
        bounds = list(np.load(args.bounds_path))
        print(pretty_print_bounds(test_env, bounds),'\n')

        
        test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv)

        if args.distribution_type == "truncnorm":
            wandb.run.summary["gaussian_dr"] = bounds
            test_env.set_dr_distribution(dr_type='truncnorm', distr=bounds)
        elif args.distribution_type == "uniform":
            wandb.run.summary["uniform_dr"] = bounds
            test_env.set_dr_distribution(dr_type='uniform', distr=bounds)
        else:
            raise ValueError('Not implemented distribution')

        test_env.set_dr_training(True)

    """Evaluation on target domain"""
    print('\n\n--- TARGET DOMAIN EVALUATION ---')
    #test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=SubprocVecEnv) # , vec_env_cls=RandomSubprocVecEnv
    load_path  = os.path.join(args.load_path, "best_model.zip")
    assert os.path.exists(load_path), "best_model.zip hasn't been saved because too few evaluations have been performed. Check --eval_freq and -t in train.py"
    
    size_layer=[]
    for _ in range(args.n_layers):
        size_layer.append(args.n_neurons)
    
    policy = Policy(algo=args.algo,
                    env=test_env,
                    device=args.device,
                    seed=args.seed,
                    lr=args.lr, 
                    batch_size=args.batch_size,
                    size_layer=size_layer,
                    load_from_pathname=load_path)
    print("Best model loaded.\nStart testing.")

    if args.test_eval:
        mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes, render=args.test_render)
        print('Target reward and stdev:', mean_reward, std_reward)
    else:
        policy.test(n_test_episodes=args.test_episodes, render=args.test_render, 
                    save_dataset=args.save_dataset, save_results=run_path, random_dynamics=args.test_rand_dynamics, seed=args.seed)
    # wandb.run.summary["target_mean_reward"] = mean_reward
    # wandb.run.summary["target_std_reward"] = std_reward

    # policy.test(n_test_episodes=args.test_episodes, render=args.test_render)

    wandb.finish()

def pretty_print_bounds(sim_env, phi):
		assert (
				sim_env is not None
				and isinstance(sim_env.dynamics_indexes, dict)
			   )

		return '\n'.join([str(sim_env.dynamics_indexes[i])+':\t'+str(round(phi[i*2],5))+', '+str(round(phi[i*2+1],5)) for i in range(len(phi)//2)])




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_env', default='trunkcube-v0', type=str, help='Test gym env')
    parser.add_argument('--group', default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo (ppo, sac)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--n_layers', default=2, type=int, help='Network number of layers')
    parser.add_argument('--n_neurons', default=128, type=int, help='Network neurons in each layer')
    parser.add_argument('--now', default=1, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--test_episodes', default=1, type=int, help='# episodes for test evaluations')
    parser.add_argument('--test_render', default=False, action='store_true', help='Render test episodes')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', default='cpu', type=str, help='<cpu,cuda>')
    parser.add_argument('--verbose', default=0, type=int, help='0,1,2')
    parser.add_argument('--notes', default=None, type=str, help='Wandb notes')
    parser.add_argument('--offline', default=False, action='store_true', help='Offline run without wandb')
    parser.add_argument('--load_path', default='./example-results/trunkcube/RFDROPO/2023_07_10_11_34_58_trunkcube-v0_ppo_t2000000_seed1_7901a3c94a22_G0QXG', type=str, help='Path of the model to load')
    parser.add_argument('--test_eval', default=False, action='store_true', help='Use of evaluate_policy')
    parser.add_argument('--save_dataset', default=False, action='store_true', help='Save dataset of observations, actions during each test episode')
    parser.add_argument('--test_rand_dynamics', default=False, action='store_true', help='Test on randomized dynamics')
    parser.add_argument('--distribution_type', default=None, type=str, help='Type of distribution: uniform, truncnorm')
    parser.add_argument('--bounds_path', default=None, type=str, help='Path for bounds to be loaded')

    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    main()
