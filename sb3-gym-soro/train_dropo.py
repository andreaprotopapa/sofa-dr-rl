"""RF-DROPO on Random_envs

  Example:

    [Inference] python train_dropo.py --env trunkcube-v0 --test_env trunkcube-v0 -n 1 --budget 5000 --now 1 --data custom --data_path ./Dataset/TrunkCube/20230208-091408_1episodes.npy --inference_only --run_path ./runs/RFDROPO --seed 0 -eps 1.0e-4 --wandb_mode disabled
    [Training]  python train_dropo.py --env trunkcube-v0 --test_env trunkcube-v0 --seed 0 --now 1 -t 2000000 --run_path ./runs/RFDROPO --training_only --bounds_path ./BestBounds/TrunkCube/RFDROPO/bounds_A1S0X.npy --wandb_mode disabled

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
from stable_baselines3.common.evaluation import evaluate_policy
import re
import json


# import random_envs
from rfdropo import Dropo
from envs.RandomVecEnv import RandomSubprocVecEnv
from utils.utils import *
from policy.policy import Policy

from sofagym import *
import denormalize



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
    
    if not args.no_output:
        create_dirs(run_path)
    
    print('\n ===== RUN NAME:', random_string, f' ({run_path}) ===== \n')


    infonly = 'InfOnly_' if args.inference_only else ''
    unmodeled = 'Unmodeled_' if args.unmodeled else ''

    if args.resume:
        with open(os.path.join(args.resume_path,'wandb_config.json'), 'r') as file:
            wandb_config = json.load(file)
        wandb_config.update(vars(args))

    wandb.init(config=vars(args) if not args.resume else wandb_config,
             id = run_id,
             dir = run_path,
             project="SoRo-RL",
             group=('DROPO_'+unmodeled+args.env+'_train' if args.group is None else args.group),
             name= infonly+args.algo+'_seed'+str(args.seed)+'_'+random_string,
             save_code=True,
             tags=None,
             notes=args.notes,
             mode=(args.wandb_mode),
             resume="allow"
             )
    
    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, env_kwargs={'unmodeled': args.unmodeled})

    unmodeled_str=""

    if unmodeled:
        for p in env.get_attr('config')[0]["unmodeled_param"]:
            unmodeled_str += f'{p}: {env.get_attr("config")[0][p+"_init"]}\n'
        wandb.config.unmodeled_param = unmodeled_str
        print("Unmodeled params:")
        print("\t"+unmodeled_str)

    test_env = gym.make(args.test_env)
    target_task = env.get_task()
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Target dynamics:', target_task)

    if not args.resume:
        wandb.config.path = run_path
        wandb.config.hostname = socket.gethostname()
        wandb.config.target_task = target_task

    if not args.training_only:

        if args.data == 'random': # Collect data randomly
            # T = collect_offline_data(env=test_env, n=args.n_trajectories)
            T = collect_offline_data_clipping(env=test_env, n=args.n_trajectories, clipping=args.clipping, env_name=args.test_env, save_dataset=True)

        elif args.data == 'off_policy': # Collect data with external policy
            # policy = Policy(algo=args.algo, env=test_env, load_from_pathname=args.off_policy, device=args.device, seed=args.seed)
            policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed)
            policy.load_state_dict(args.off_policy)
            # T = collect_offline_data(env=test_env, n=args.n_trajectories, policy=policy)
            T = collect_offline_data_clipping(env=test_env, n=args.n_trajectories, policy=policy, clipping=args.clipping, env_name=args.test_env)

        elif args.data == 'custom': # Custom offline dataset
            if os.path.isfile(args.data_path):
                T = load_data_from_file(args.data_path)
            else:
                raise ValueError(f"{args.data_path}: file is not correct")

        else:
            raise ValueError(f"Unsupported args.data parameter: {args.data}")
        if T['actions'].ndim == 1:
            T['actions'] = T['actions'].reshape((len(T['actions']),1))
        wandb.config.n_transitions = T['observations'].shape[0]


        dropo = Dropo(sim_env=env,
                  t_length=None,
                  scaling=args.scaling,
                  seed=args.seed,
                  clip_state=args.clip_state
                  )

        # Load target offline dataset
        dropo.set_offline_dataset(T, n=args.n_trajectories)

        # Run DROPO
        (best_phi,
        best_score,
        elapsed) = dropo.optimize_dynamics_distribution_resetfree(opt='cma',
                                                                budget=args.budget,
                                                                additive_variance=args.additive_variance,
                                                                epsilon=args.epsilon,
                                                                sample_size=args.sample_size,
                                                                now=args.now,
                                                                learn_epsilon=False,
                                                                normalize=args.normalize,
                                                                logstdevs=args.logstdevs,
                                                                clip_episode_length=args.clip_episode_length,
                                                                temperatureRegularization=args.temperatureRegularization,
                                                                wandb=wandb,
                                                                run_path=run_path,
                                                                unmodeled=unmodeled, 
                                                                resume=args.resume, 
                                                                resume_path=args.resume_path)
        
        
        """
            OUTPUT RESULTS
        """

        output_results(dropo, args, best_phi, env.get_task(), best_score, elapsed)

        if not args.no_output:  # Output results to file
            # make_dir(args.output_dir)

            filename = 'RF-DROPO_'+str(args.env)+'_n'+str(args.n_trajectories)+'_'+datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.txt'
            with open(os.path.join(run_path, filename), 'a', encoding='utf-8') as file:
                output_results(dropo, args, best_phi, env.get_task(), best_score, elapsed, file=file)

        
        np.save(run_path+unmodeled+"best_phi.npy", best_phi)
    else:
        assert args.bounds_path is not None
        print("\nLoading best bounds from ", args.bounds_path)
        best_phi = list(np.load(args.bounds_path))

        if not args.load_denormalized_bounds:
            print('Normalized means and st.devs:\n---------------')
            print(denormalize.pretty_print_bounds(env, best_phi),'\n')
            best_phi = denormalize.denormalize_bounds(env, args.logstdevs, best_phi)
            print('Denormalized means and st.devs:\n---------------')
            print(denormalize.pretty_print_bounds(env, best_phi),'\n')
        else:
            print('Denormalized means and st.devs:\n---------------')
            print(denormalize.pretty_print_bounds(env, best_phi),'\n')

    wandb.run.summary["best_phi"] = best_phi

    if not args.inference_only:
        env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, env_kwargs={'unmodeled': args.unmodeled})
        env.set_dr_distribution(dr_type='truncnorm', distr=best_phi)
        env.set_dr_training(True)
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

        env.set_dr_training(False)

        policy.save_state_dict(run_path+"final_model.pth")
        policy.save_full_state(run_path+"final_full_state.zip")
        print('--- Policy training done ----')

        print('\n\nMean reward and stdev:', mean_reward, std_reward)

        wandb.run.summary["train_mean_reward"] = mean_reward
        wandb.run.summary["train_std_reward"] = std_reward
        wandb.run.summary["which_best_model"] = which_one

        torch.save(best_policy, run_path+"overall_best.pth")
        wandb.save(run_path+"overall_best.pth")


        """Evaluation on target domain"""
        print('\n\n--- TARGET DOMAIN EVALUATION ---')
        test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv)
        policy = Policy(algo=args.algo,
                    env=test_env,
                    device=args.device,
                    seed=args.seed,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    size_layer=size_layer
                    )
        policy.load_state_dict(best_policy)

        mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes)
        print('Target reward and stdev:', mean_reward, std_reward)

        wandb.run.summary["target_mean_reward"] = mean_reward
        wandb.run.summary["target_std_reward"] = std_reward


    wandb.finish()

def output_results(dropo, args, best_phi, gt, best_score, elapsed, file=None):
    print('\n-----------', file=file)
    print('RESULTS\n', file=file)
    print('ARGS:', vars(args), '\n\n', file=file)

    print('GROUND TRUTH dynamics parameters:', gt, '\n', file=file)

    print('Best means and st.devs:\n---------------', file=file)
    print(dropo.pretty_print_bounds(best_phi),'\n', file=file)
    print('Best score (log likelihood):', best_score, file=file)

    # print('MSE:', dropo.MSE_trajectories(dropo.get_means(best_phi)), file=file)
    print('Elapsed:', round(elapsed/60, 4), 'min', file=file)


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
    parser.add_argument('--now', default=1, type=int, help='Number of cpus for parallelization (Default: 1 => no parallelization)')
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

    parser.add_argument("--n_trajectories", "-n", type=int, default=None, help="Number of target trajectories in the target dataset to consider")
    parser.add_argument('--data', default='custom', type=str, help='Offline data collection method [random, off_policy, custom]')
    parser.add_argument('--data_path', default=None, type=str, help='Path to custom offline dataset')
    parser.add_argument('--off_policy', default=None, type=str, help='Path to model for data collection off-policy')
    parser.add_argument("--budget", type=int, default=5000, help="Number of evaluations in the opt. problem (Default: 1000)")
    parser.add_argument('--inference_only', default=False, action='store_true', help='Avoid policy training')
    parser.add_argument('--clipping', default=None, type=int, help='Clipping the real-world rollout at <clipping> state-transitions')
    parser.add_argument('--initial_info_only', default=True, action='store_true', help='Reset state only once while computing obj. function. - Always set to TRUE')
    parser.add_argument('--training_only', default=False, action='store_true', help='Avoid dynamics inference')
    parser.add_argument('--bounds_path', default=None, type=str, help='Path for best bounds to be loaded')

    parser.add_argument('--unmodeled', default=False, action='store_true', help='Unmodeled setting (implemented only for trunkcube env)')

    parser.add_argument('--no_output', default=False, action='store_true', help='DO NOT save output results of the optimization')

    parser.add_argument('--additive_variance', default=True, action='store_true', help='RECOMMENDED. Add value --epsilon to the diagonal of the cov_matrix to regularize the next-state distribution inference')
    parser.add_argument('--normalize', default=True, action='store_true', help='RECOMMENDED. Normalize dynamics search space to [0,4] as a regularization for CMA-ES.')
    parser.add_argument('--load_denormalized_bounds', default=False, action='store_true', help='The loaded bounds are already denormalized for training phase.')
    parser.add_argument('--logstdevs', default=True, action='store_true', help='RECOMMENDED. Optimize stdevs in log space')
    parser.add_argument("--epsilon", "-eps", type=float, default=1.0e-5 , help="Epsilon hyperparameter. Valid only when additive_variance is set to True (default: 1e-3)")
    parser.add_argument('--scaling', default=False, action='store_true', help='Scaling each state dimension')
    parser.add_argument("--sample_size", "-ss", type=int, default=100, help="Number of observations to sample to estimate the next-state distribution (Default: 100)")
    parser.add_argument("--clip_episode_length", type=int, default=100, help="If set, clip episode length to this value.")
    parser.add_argument('--temperatureRegularization', default=True, action='store_true', help='Start with a single transition, then consider a longer-time horizon as CMA converges.')
    parser.add_argument('--clip_state', default=None, type=int, help='Clipping the dimensionality of the observation space (for Trunk and TrunkCube envs)')


    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    main()

