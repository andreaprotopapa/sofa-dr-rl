#%% Imports
from src.data.panda_data_generator import PandaDataGenerator
from src.utils.param_inference import *
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--eps", type=float, default=0.0, help="Value added to covariance diagonal")
parser.add_argument("--gt-dist", type=str, default=None, help="File with true .bounds")
parser.add_argument("--panda-env", type=str, default="PandaPushFixedStart-PosCtrl-v0", help="Panda env to use")
parser.add_argument("--noise-var", type=float, default=0.0, help="Noise variance added to observations")
parser.add_argument("--dlimit", type=int, default=1000, help="Number of data samples")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=123, help="Random seed value")
parser.add_argument("--components", type=int, default=5, help="Number of Gaussian mixture model components")
parser.add_argument("--model", type=str, default="MDRFF",
        choices=["MDN", "MDRFF", "MDLSTM", "MDRFFLSTM"], help="NN model to use")
parser.add_argument("--true-dist", type=str, default=None)
args = parser.parse_args()

cur_root_dir = os.getcwd()
print("Current Directory: {}".format(cur_root_dir))
#%% Load Policy

g = PandaDataGenerator(demo_file=args.dataset, load_from_file=False,
        max_data=args.dlimit, noise_var=args.noise_var,
        env_name=args.panda_env)

#%% Load data shape
params, stats = g.gen(1)
shapes = {"params": params.shape[1], "data": stats.shape[1]}
print("Total data size: {}".format(params.shape[0]))

#%% Plot Results for mass and length specific params
#true_obs = g.env.original_masses.reshape(1, -1)
if args.gt_dist is not None:
    with open(args.gt_dist) as f:
        lines = f.readlines()
    dist_type = lines[0].replace("\n", "")
    assert dist_type in ["truncnorm", "gaussian"]

    params = np.array(list(map(float, lines[1].replace("\n", "").split(","))))
    means = params[::2]
    stds = params[1::2]
    true_obs = means.reshape(1, -1)
else:
    true_obs = g.env.min_task.reshape(1, -1)

#%% Train model
log_mdn, inf_mdn = train(epochs=args.epochs, batch_size=100, params_dim=g.param_dim,
                         stats_dim=g.feature_dim, num_sampled_points=10,
                         generator=g, model=args.model, n_components=args.components,
                         eps=args.eps, seed=args.seed, quasi_random=False)

gen_posterior, expert_posterior = get_results_from_true_obs(env_params=g.params,
        true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
        p_lower=g.env.min_task, p_upper=g.env.max_task, env_name="panda")

gen_posterior.to_file("gen_posterior.bounds")
if expert_posterior is not None:
    expert_posterior.to_file("expert_posterior.bounds")


if args.true_dist:
    g.env.load_dr_distribution_from_file(args.true_dist)
    true_obs = g.env.mean_task.reshape(1, -1)
    dist_posterior = get_results_from_current_dist(env_params=g.params,
        true_obs=true_obs, generator=g, inf=inf_mdn, shapes=shapes,
        p_lower=g.env.min_task, p_upper=g.env.max_task, env_name="panda",
        dyn_samples=200)
    dist_posterior.to_file("dist_posterior.bounds")

