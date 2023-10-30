"""Test of Reset-Free DROPO (RF-DROPO)

For the RandomHopperMass environment, a dataset has been collected offline
from with a semi-converged policy and made available in datasets/.

This repo needs the random_envs package installed (https://github.com/gabrieletiboni/random-envs)

Examples:
    
    [debug]
        python test_resetfree_dropo.py config=[resetfree,debug] budget=50 seed=42

    [Vanilla]
        python test_resetfree_dropo.py config=[resetfree] budget=1000 now=10 seed=42

    [Noisy]
        python test_resetfree_dropo.py config=[resetfree] dataset=datasets/hoppernoisy1e-510000 budget=1000 now=10 seed=42

    [Unmodeled environment test]
        python test_resetfree_dropo.py config[resetfree] env=RandomHopperUnmodeled-v0 dataset=datasets/hopper10000 budget=1000 now=10 seed=42
"""
import glob
import os
import sys
import pdb
from datetime import datetime
import argparse
import socket

import numpy as np
import gym
import wandb
from stable_baselines3.common.env_util import make_vec_env
from dropo_dev import Dropo
import random_envs

from RandomVecEnv import RandomSubprocVecEnv
from args import *

def main():
    random_str = get_random_string(5)
    set_seed(args.seed)

    run_name = random_str+('_'+args.name if args.name is not None else '')+'-S'+str(args.seed)
    save_dir = os.path.join((args.output_dir if not args.debug else 'debug_runs'), run_name)

    if not args.no_output:
        create_dirs(save_dir)
        save_config(args, save_dir)

    print('\n ===== RUN NAME:', run_name, f' ({save_dir}) ===== \n')
    print(pformat_dict(args, indent=0))

    wandb.init(config=to_dict(args),
               project="rf-dropo-dev",
               name=run_name,
               group=args.group,
               save_code=True,
               notes=args.notes,
               mode=('online' if not args.debug else 'disabled'))
    wandb.config.path = save_dir
    wandb.config.hostname = socket.gethostname()

    # foo = gym.vector.make(args.env, num_envs=5)
    sim_env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv)  # A VecEnv is required and assumed, even if --now=1
    target_env = gym.make(args.env)
    # sim_env.reset()
    # act = np.zeros((args.now, sim_env.action_space.shape[0]))
    # sim_env.step(act)

    print('Action space:', sim_env.action_space)
    print('State space:', sim_env.observation_space)
    print('Target dynamics:', target_env.get_task())
    # print('\nARGS:', vars(args))

    observations = np.load(glob.glob(os.path.join(args.dataset, '*_observations.npy'))[0])
    next_observations = np.load(glob.glob(os.path.join(args.dataset, '*_nextobservations.npy'))[0])
    actions = np.load(glob.glob(os.path.join(args.dataset, '*_actions.npy'))[0])
    terminals = np.load(glob.glob(os.path.join(args.dataset, '*_terminals.npy'))[0])

    T = {'observations': observations, 'next_observations': next_observations, 'actions': actions, 'terminals': terminals }

    # Initialize RF-DROPO
    dropo = Dropo(sim_env=sim_env,
                  t_length=None,
                  scaling=args.scaling,
                  clip_state=args.clip_state,
                  seed=args.seed)


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
                                                               normalize=args.normalize,
                                                               logstdevs=args.logstdevs,
                                                               clip_episode_length=args.clip_episode_length,
                                                               temperatureRegularization=args.temperatureRegularization,
                                                               wandb=wandb)
    
    
    """
        OUTPUT RESULTS
    """
    wandb.run.summary["best_phi"] = best_phi

    output_results(dropo, args, best_phi, target_env.get_task(), best_score, elapsed)

    if not args.no_output:  # Output results to file
        # make_dir(args.output_dir)

        filename = 'RF-DROPO_'+str(args.env)+'_n'+str(args.n_trajectories)+'_'+datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.txt'
        with open(os.path.join(save_dir, filename), 'a', encoding='utf-8') as file:
            output_results(dropo, args, best_phi, target_env.get_task(), best_score, elapsed, file=file)

    wandb.finish()

    return


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


def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)


# def make_dir(dir_path):
#     try:
#         os.mkdir(dir_path)
#     except OSError:
#         pass

#     return

if __name__ == '__main__':
    main()