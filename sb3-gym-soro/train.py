"""Train script using sb3-gym-template, without using Domain Randomization

  Examples:
    
    [Training] python train.py --env trunkcube-v0 --algo ppo --now 1 --seed 0 -t 2000000 --run_path ./runs/no-DR --wandb_mode disabled

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
import re
from stable_baselines3.common.env_util import make_vec_env

# import random_envs
# from envs.RandomVecEnv import RandomSubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.utils import *
from policy.policy import Policy

from sofagym import *

def main():

    assert args.env is not None
    assert not(args.resume ^ (args.resume_path is not None and args.resume_wandb is not None))
    if args.test_env is None:
        args.test_env = args.env
    
    torch.set_num_threads(args.now)   

    pprint(vars(args))
    set_seed(args.seed)
    resume_string = re.findall("_([^_]+)?$", args.resume_path)[0] if args.resume_path is not None else None #take the random string of the resume_path
    random_string = get_random_string(5) if not args.resume else resume_string 
    run_id = args.resume_wandb if (args.resume and args.wandb_mode == "online") else wandb.util.generate_id()

    if args.run_path is not None:
        run_path = args.run_path+"/runs/"+str(args.env)+"/"+get_run_name(args)+"_"+random_string+"/"
    else:
        run_path = "runs/"+str(args.env)+"/"+get_run_name(args)+"_"+random_string+"/"
    create_dirs(run_path)

    wandb.init(config=vars(args),
             id = run_id,
             dir = run_path,
             project="SoRo-RL",
             group=(args.env+'_train' if args.group is None else args.group),
             #name='SoRo'+('InfOnly' if args.inference_only else '')+'_'+args.algo+'_seed'+str(args.seed)+'_'+random_string,
             name=args.algo+'_seed'+str(args.seed)+'_'+random_string,
             save_code=True,
             tags=None,
             notes=args.notes,
             mode=(args.wandb_mode),
             resume="allow",
             )

    # env = gym.make(args.env)
    # test_env = gym.make(args.test_env)

    if not args.resume:
        wandb.config.path = run_path
        wandb.config.hostname = socket.gethostname()

    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=SubprocVecEnv) #, vec_env_cls=RandomSubprocVecEnv
    
    size_layer=[]
    for _ in range(args.n_layers):
        size_layer.append(args.n_neurons)

    if args.resume:
        ckpt = os.listdir(os.path.join(args.resume_path, "logs"))[0]
        load_path  = os.path.join(args.resume_path, "logs", ckpt)
        assert os.path.exists(load_path), "model_ckpt_*_steps.zip hasn't been found"
        policy = Policy(algo=args.algo, 
                        env=env, 
                        lr=args.lr, 
                        batch_size=args.batch_size,
                        size_layer=size_layer,
                        device=args.device, 
                        seed=args.seed, 
                        load_from_pathname=load_path,
                        reset_num_timesteps=False)
        n_previous_steps = policy.model.num_timesteps
        policy.model.num_timesteps = n_previous_steps - args.eval_freq
        print(f"Checkpoint model loaded.\nResume training from step {policy.model.num_timesteps}.")
        print(f"Training for other {args.timesteps - policy.model.num_timesteps} steps.")
        timesteps_effective = args.timesteps - policy.model.num_timesteps
        if timesteps_effective <= 0:
            print("\nTraining finished. Exit.")
            exit
    else:
        policy = Policy(algo=args.algo,
                        env=env,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        size_layer=size_layer,
                        device=args.device,
                        seed=args.seed)
        timesteps_effective = args.timesteps


    print('--- Policy training start ---')
    mean_reward, std_reward, best_policy, which_one = policy.train(timesteps=timesteps_effective,
                                                                   stopAtRewardThreshold=args.reward_threshold,
                                                                   n_eval_episodes=args.eval_episodes,
                                                                   eval_freq=args.eval_freq,
                                                                   best_model_save_path=run_path,
                                                                   return_best_model=True,
                                                                   save_freq=(int(args.save_freq) if args.save_freq is not None else None),
                                                                   keep_prev_ckpt=args.keep_prev_ckpt
                                                                   )

    policy.save_state_dict(run_path+"final_model.pth")
    policy.save_full_state(run_path+"final_full_state.zip")
    print('--- Policy training done ----')

    print('\n\nMean reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["train_mean_reward"] = mean_reward
    wandb.run.summary["train_std_reward"] = std_reward
    wandb.run.summary["which_best_model"] = which_one

    torch.save(best_policy, run_path+"overall_best.pth")
    wandb.save(run_path+"overall_best.pth")


    """Evaluation on test domain"""
    print('\n\n--- TEST DOMAIN EVALUATION ---')
    #now = 1 when render
    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=SubprocVecEnv) # , vec_env_cls=RandomSubprocVecEnv
    policy = Policy(algo=args.algo,
                    env=test_env,
                    device=args.device,
                    seed=args.seed,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    size_layer=size_layer
                    )
    policy.load_state_dict(best_policy)

    mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes, render=args.test_render)
    print('Test reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["test_mean_reward"] = mean_reward
    wandb.run.summary["test_std_reward"] = std_reward

    wandb.finish()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='trunkcube-v0', type=str, help='Train gym env')
    parser.add_argument('--test_env', default=None, type=str, help='Test gym env')
    parser.add_argument('--group', default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo (ppo, sac)')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=144, type=int, help='Batch size')
    parser.add_argument('--n_layers', default=3, type=int, help='Network number of layers')
    parser.add_argument('--n_neurons', default=512, type=int, help='Network neurons in each layer')
    parser.add_argument('--now', default=1, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--timesteps', '-t', default=2000000, type=int, help='Training timesteps')
    parser.add_argument('--reward_threshold', default=False, action='store_true', help='Stop at reward threshold')
    parser.add_argument('--eval_freq', default=10000, type=int, help='timesteps frequency for training evaluations')
    parser.add_argument('--eval_episodes', default=50, type=int, help='# episodes for training evaluations')
    parser.add_argument('--test_episodes', default=100, type=int, help='# episodes for test evaluations')
    parser.add_argument('--test_render', default=False, action='store_true', help='Render test episodes')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', default='cpu', type=str, help='<cpu,cuda>')
    parser.add_argument('--verbose', default=0, type=int, help='0,1,2')
    parser.add_argument('--notes', default=None, type=str, help='Wandb notes')
    parser.add_argument('--wandb_mode', default='online', type=str, help='Wandb mode: online (default), offline, disabled')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume from previous training ckpt')
    parser.add_argument('--resume_path', default=None, type=str, help='Path for the ckpt training')
    parser.add_argument('--resume_wandb', default=None, type=str, help='Run ID of wandb previous run (e.g., \'wandb/run-date_time-ID\')')
    parser.add_argument('--run_path', default=None, type=str, help='Path for saving run results')
    parser.add_argument('--save_freq', default=None, type=str, help='timesteps frequency for savining ckpt')
    parser.add_argument('--keep_prev_ckpt', default=False, action='store_true', help='Keep all the ckpt saving files')

    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    main()