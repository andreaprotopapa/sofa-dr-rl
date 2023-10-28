"""Uniform Domain Distribution (UDR) on Random_envs: 

  Example: 

    [Training] python train_udr.py --env trunkcube-v0 --algo ppo --seed 0 --now 1 -t 2000000 --run_path ./runs/UDR --wandb_mode disabled

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
from envs.RandomVecEnv import RandomSubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.utils import *
from policy.policy import Policy

from sofagym import *

def main():

    assert args.env is not None
    assert not(args.resume ^ (args.resume_path is not None and args.resume_wandb is not None))
    if args.test_env is None:
        args.test_env = args.env
    if args.unmodeled and args.env != "trunkcube-v0":
        raise ValueError(f"Unmodeled setting implemented only for trunkcube-v0")

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

    unmodeled = 'Unmodeled_' if args.unmodeled else ''

    wandb.init(config=vars(args),
             id = run_id,
             dir = run_path,
             project="SoRo-RL",
             group=('UDR_'+unmodeled+args.env+'_train' if args.group is None else args.group),
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

    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, env_kwargs={'unmodeled': args.unmodeled}) #, vec_env_cls=SubprocVecEnv

    unmodeled_str=""

    if unmodeled:
        for p in env.get_attr('config')[0]["unmodeled_param"]:
            unmodeled_str += f'{p}: {env.get_attr("config")[0][p+"_init"]}\n'
        wandb.config.unmodeled_param = unmodeled_str
        print("Unmodeled params:")
        print("\t"+unmodeled_str)
    
    # >>>>>>>>>> UDR 
    min_space, max_space = env.get_task_search_bounds()
    bound_min = np.random.uniform(min_space, max_space, len(min_space))
    bound_max = np.random.uniform(min_space, max_space, len(min_space))

    bound_final = []
    for bm, bM in zip(bound_min, bound_max):
         c = bm
         bm = bm if bm < bM else bM
         bM = bM if c < bM else c 
         bound_final.append(bm)
         bound_final.append(bM)
    distributions_bounds = np.array(bound_final)

    env.set_dr_distribution(dr_type='uniform', distr=distributions_bounds)
    env.set_dr_training(True)
    dynamics_names = env.get_attr('dynamics_indexes')[0]

    wandb.config.update({"train_dynamics_names": dynamics_names}, allow_val_change=True) 
    wandb.config.update({"train_space_bounds": [ [i, j] for i,j  in zip(min_space, max_space) ]}, allow_val_change=True) 
    wandb.config.update({"train_uniform_distr": [ [distributions_bounds[i*2], distributions_bounds[i*2+1]] for i in range(len(distributions_bounds)//2) ]}, allow_val_change=True) 

    print('\n\nSampled uniform DR:\n', distributions_bounds, '\n\n')
    save_uniform_distr(dynamics_names, distributions_bounds, run_path)

    # <<<<<<<<<<

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
        print(f"Checkpoint model loaded.\nResume training from step {n_previous_steps}.")
    else:
        policy = Policy(algo=args.algo,
                        env=env,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        size_layer=size_layer,
                        device=args.device,
                        seed=args.seed)


    print('--- Policy training start ---')
    mean_reward, std_reward, best_policy, which_one = policy.train(timesteps=args.timesteps,
                                                                   stopAtRewardThreshold=args.reward_threshold,
                                                                   n_eval_episodes=args.eval_episodes,
                                                                   eval_freq=args.eval_freq,
                                                                   best_model_save_path=run_path,
                                                                   return_best_model=True)

    policy.save_state_dict(run_path+"final_model.pth")
    policy.save_full_state(run_path+"final_full_state.zip")
    print('--- Policy training done ----')

    print('\n\nMean reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["train_mean_reward"] = mean_reward
    wandb.run.summary["train_std_reward"] = std_reward
    wandb.run.summary["which_best_model"] = which_one

    torch.save(best_policy, run_path+"overall_best.pth")
    wandb.save(run_path+"overall_best.pth")
    env.set_dr_training(False)

    """Evaluation on test domain"""
    print('\n\n--- TEST DOMAIN EVALUATION ---')
    #now = 1 when render
    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv) # , vec_env_cls=SubprocVecEnv
    wandb.config.test_dynamics = test_env.get_task()
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

def save_uniform_distr(dynamics_names, uniform_dr, path, comment=''):
    with open(os.path.join(path, '')+'udr.bounds', 'w', encoding='utf-8') as file:
        bound_min = np.empty((uniform_dr.shape[0]//2))
        bound_max = np.empty((uniform_dr.shape[0]//2))
        for i in range(len(uniform_dr)//2):
            bound_min[i] = uniform_dr[i*2]
            bound_max[i] = uniform_dr[i*2 + 1]

        print('uniform', file=file)
        for i,(b_min, b_max) in enumerate(zip(bound_min, bound_max)):
            print(dynamics_names[i], file=file)
            if i != (len(bound_min)-1):
                print('\t'+str(b_min)+', ', file=file, end='')
                print('\t'+str(b_max)+'\n', file=file, end='')
            else:
                print('\t'+str(b_min)+', ', file=file, end='')
                print('\t'+str(b_max), file=file)

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

    parser.add_argument('--unmodeled', default=False, action='store_true', help='Unmodeled setting (implemented only for trunkcube env)')


    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    main()