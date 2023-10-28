"""Bayes-Sim (https://github.com/rafaelpossas/bayes_sim) on Random_envs: 

  Example:

    [Inference] python train_bayessim.py --run_path ./runs/BayesSim --data custom --data_path ./Dataset/TrunkCube/20230208-091408_1episodes.npy -n 1 --seed 0 --inference_only --model MDN --wandb_mode disabled
    [Training]  python train_bayessim.py --env trunkcube-v0 --test_env trunkcube-v0 --seed 0 --now 1 -t 2000000 --run_path ./runs/BayesSim/ --training_only --bounds_path ./BestBounds/TrunkCube/BayesSim/bounds_N6TW2.txt --wandb_mode disabled

"""

#%% Imports
from bayes_sim.src.data.sofa_data_generator import SofaDataGenerator
from bayes_sim.src.utils.param_inference import *
from pprint import pprint
import os
import argparse
import gym
from utils.utils import *
from policy.policy import Policy
import pdb
import wandb
import re

from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import torch
from envs.RandomVecEnv import RandomSubprocVecEnv

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

        cur_root_dir = os.getcwd()
        print("Current Directory: {}".format(cur_root_dir))

        infonly = 'InfOnly_' if args.inference_only else ''
        unmodeled = 'Unmodeled_' if args.unmodeled else ''

        wandb.init(config=vars(args),
             id = run_id,
             dir = run_path,
             project="SoRo-RL",
             group=('BayesSim_'+unmodeled+args.env+'_train' if args.group is None else args.group),
             name= infonly+args.algo+'_seed'+str(args.seed)+'_'+random_string,
             save_code=True,
             tags=None,
             notes=args.notes,
             mode=(args.wandb_mode),
             resume="allow",
             )

        env = gym.make(args.env, unmodeled=args.unmodeled)

        unmodeled_str=""

        if unmodeled:
                for p in env.config["unmodeled_param"]:
                        unmodeled_str += f'{p}: {env.config[p+"_init"]}\n'
                wandb.config.unmodeled_param = unmodeled_str
                print("Unmodeled params:")
                print("\t"+unmodeled_str)

        test_env = gym.make(args.test_env)
        target_task = env.get_task()
        print('Target task:', target_task)

        if not args.resume:
                wandb.config.path = run_path
                wandb.config.hostname = socket.gethostname()
                wandb.config.target_task = target_task

        if not args.training_only:
                #%% Load Policy

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

                g = SofaDataGenerator(env=env, dataset=T, load_from_file=False,
                        max_data=args.dlimit, noise_var=args.noise_var, assets_path=run_path)

                #%% Load data shape
                params, stats = g.gen(1)
                shapes = {"params": params.shape[1], "data": stats.shape[1]}
                print("Total data size: {}".format(params.shape[0]))
                # pdb.set_trace()
                #%% Train model
                log_mdn, inf_mdn = train(epochs=args.epochs, batch_size=100, params_dim=g.param_dim,
                                        stats_dim=g.feature_dim, num_sampled_points=args.n_points,
                                        generator=g, model=args.model, n_components=args.components,
                                        eps=args.eps, seed=args.seed, quasi_random=False, wandb=wandb, save_path=run_path)
               
                print(f"log_mdn: {log_mdn}\ninf_mdn: {inf_mdn}")
                # pdb.set_trace()
                """
                        OUTPUT RESULTS
                """
                #%% Plot Results for mass and length specific params
                true_obs = target_task.reshape(1, -1)

                gen_posterior, expert_posterior = get_results_from_true_obs(env_params=list(env.dynamics_indexes.values()),
                        true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                        p_lower=g.env.min_task, p_upper=g.env.max_task, env_name=env.config['scene'], save_path=run_path)

                gen_posterior.to_file(run_path+"gen_posterior.bounds")
                if expert_posterior is not None:
                        expert_posterior.to_file(run_path+"expert_posterior.bounds")


                if args.true_dist:
                        g.env.load_dr_distribution_from_file(args.true_dist)
                        true_obs = g.env.mean_task.reshape(1, -1)
                        dist_posterior = get_results_from_current_dist(env_params=list(env.dynamics_indexes.values()),
                                true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
                                p_lower=g.env.min_task, p_upper=g.env.max_task, env_name="hopper_wrong_mass",
                                dyn_samples=200, save_path=run_path)
                        dist_posterior.to_file(run_path+"dist_posterior.bounds")
        else:
                assert args.bounds_path is not None
                print("\nLoading best bounds from ", args.bounds_path)
                best_phi = create_bounds_from_file(args.bounds_path, args.components)
                wandb.run.summary["best_phi"] = best_phi
                np.save(run_path+unmodeled+"best_phi.npy", best_phi)

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

def create_bounds_from_file(file, n_components):
        bounds = []
        dict_p = {}
        with open(file) as f:
                i = 0
                f = list(f)
                for line in f[1:]:
                        dict_c = {}
                        c = line.split(": ")[0]
                        for p in line.split(": ")[1].split("; "):
                                k = p.split( " = ")[0]
                                v = p.split( " = ")[1]
                                dict_c[k] = float(v)
                        dict_p[c] = dict_c

                        i=i+1
                        if i ==  n_components:
                                i = 0
                                bounds.append(dict_p)
                                dict_p = {}
        final_bounds = []
        for p in bounds:
                mean = 0
                var = 0
                for k in p.keys():
                        mean += p[k]['mean'] * p[k]['mixture weight'] 
                        var  += p [k]['variance'] * p[k]['mixture weight'] 
                final_bounds.extend([mean, var])
        
        return final_bounds

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', default='trunkcube-v0', type=str, help='Train gym env')
        parser.add_argument('--test_env', default='trunkcube-v0', type=str, help='Test gym env')
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

        
        parser.add_argument("--eps", type=float, default=0.0, help="Value added to covariance diagonal")
        parser.add_argument("--noise-var", type=float, default=0.0, help="Noise variance added to observations")
        parser.add_argument("--dlimit", type=int, default=50, help="Number of data samples")
        parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
        parser.add_argument("--components", type=int, default=5, help="Number of Gaussian mixture model components")
        parser.add_argument("--model", type=str, default="MDRFF",
                choices=["MDN", "MDRFF", "MDLSTM", "MDRFFLSTM"], help="NN model to use")
        parser.add_argument("--true-dist", type=str, default=None)
        parser.add_argument("--n_points", type=int, default=1000, help="Number of data points generated before training")

        parser.add_argument("--n_trajectories", "-n", type=int, default=None, help="Number of target trajectories for running DROID. if --sparse-mode or --data random is selected, this parameter refers to the number of single TRANSITIONS instead.")
        parser.add_argument('--data', default='custom', type=str, help='Offline data collection method [random, off_policy, custom]')
        parser.add_argument('--data_path', default='/home/andrea/MySofa/sb3-gym-soro/Dataset/TrunkCube/20230208-091408_1episodes.npy', type=str, help='Path to custom offline dataset')
        parser.add_argument('--off_policy', default=None, type=str, help='Path to model for data collection off-policy')
        parser.add_argument('--clipping', default=None, type=int, help='Clipping the real-world rollout at <clipping> state-transitions')
        parser.add_argument('--inference_only', default=False, action='store_true', help='Avoid policy training')
        parser.add_argument('--training_only', default=False, action='store_true', help='Avoid dynamics inference')
        parser.add_argument('--bounds_path', default=None, type=str, help='Path for best bounds to be loaded')

        parser.add_argument('--unmodeled', default=False, action='store_true', help='Unmodeled setting (implemented only for trunkcube env)')


        return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    main()