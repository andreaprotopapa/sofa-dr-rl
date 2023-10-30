#%% Imports
from src.data.hopper_wrongm_data_generator import HopperWrongMassDataGenerator
from src.utils.param_inference import *
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--eps", type=float, default=0.0, help="Value added to covariance diagonal")
parser.add_argument("--noise-var", type=float, default=0.0, help="Noise variance added to observations")
parser.add_argument("--dlimit", type=int, default=50, help="Number of data samples")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=123, help="Random seed value")
parser.add_argument("--components", type=int, default=5, help="Number of Gaussian mixture model components")
parser.add_argument("--model", type=str, default="MDRFF",
        choices=["MDN", "MDRFF", "MDLSTM", "MDRFFLSTM"], help="NN model to use")
args = parser.parse_args()

cur_root_dir = os.getcwd()
print("Current Directory: {}".format(cur_root_dir))
#%% Load Policy

g = HopperWrongMassDataGenerator(demo_file=args.dataset, load_from_file=False,
        max_data=args.dlimit, noise_var=args.noise_var)

#%% Load data shape
params, stats = g.gen(1)
shapes = {"params": params.shape[1], "data": stats.shape[1]}
print("Total data size: {}".format(params.shape[0]))

#%% Train model
log_mdn, inf_mdn = train(epochs=args.epochs, batch_size=100, params_dim=g.param_dim,
                         stats_dim=g.feature_dim, num_sampled_points=1000,
                         generator=g, model=args.model, n_components=args.components,
                         eps=args.eps, seed=args.seed)

#%% Plot Results for mass and length specific params
true_obs = g.env.original_masses.reshape(1, -1)

gen_posterior, expert_posterior = get_results_from_true_obs(env_params=[f"m{i}" for i in range(4)],
        true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
        p_lower=g.env.min_task, p_upper=g.env.max_task, env_name="hopper_wrong_mass")

gen_posterior.to_file("gen_posterior.bounds")
if expert_posterior is not None:
    expert_posterior.to_file("expert_posterior.bounds")


