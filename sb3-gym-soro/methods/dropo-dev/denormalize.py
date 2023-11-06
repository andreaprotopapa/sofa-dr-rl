"""Denormalize bounds

"""
import glob
import sys
import pdb
from datetime import datetime
import argparse

import numpy as np
import gym

from utils import *
from dropo_dev import Dropo

from sofagym import *

import nevergrad as ng


def main():

	args = parse_args()

	sim_env = gym.make(args.env)
	unmodeled = 'Unmodeled_' if args.unmodeled else ''
	phi = np.load(args.run_path+'/'+unmodeled+'best_phi.npy')
	min_task, max_task = sim_env.get_task_search_bounds()
	normalized_width = 4
	
	new_phi = denormalize_bounds(sim_env, args.logstdevs, phi)

	"""
		OUTPUT RESULTS
	"""

	print('\n-----------')
	print('RESULTS\n')
	
	print('ARGS:', vars(args), '\n\n')

	print('Original means and st.devs:\n---------------')
	print(pretty_print_bounds(sim_env, phi),'\n')

	print('Denormalized means and st.devs:\n---------------')
	print(pretty_print_bounds(sim_env, new_phi),'\n')

	np.save(args.run_path+'/'+unmodeled+'denorm_phi.npy', new_phi)

	# phi_normalized = []
	# for i, (mean, stdev) in enumerate(zip(get_means(new_phi), get_stdevs(new_phi))):
	# 	width = max_task[i]-min_task[i] # Search interval for this parameter
	# 	# MEAN
	# 	phi_normalized.append(ng.p.Scalar(init=mean).set_bounds(lower=0, upper=normalized_width))
	# 	# STD
	# 	initial_std = width/8	# This may sometimes lead to a stdev smaller than the lower threshold of 0.00001, so take the minimum
	# 	stdev_lower_bound = np.min([0.00001, initial_std-1e-5])
	# 	stdev_upper_bound = width/4
	# 	phi_normalized.append(ng.p.Scalar(init=stdev).set_bounds(lower=0, upper=normalized_width))
	
	# phi_normalized = [i.value for i in phi_normalized]
	# print('Renormalized once means and st.devs:\n---------------')
	# print(pretty_print_bounds(sim_env, phi_normalized),'\n')

	phi_normalized = []
	
	phi_normalized = normalize_bounds(sim_env, args.logstdevs, phi)

	print('Renormalized twice means and st.devs:\n---------------')
	print(pretty_print_bounds(sim_env, phi_normalized),'\n')

	phi_denormalized_again = denormalize_bounds(sim_env, args.logstdevs, phi_normalized)

	print('Denormalized again means and st.devs:\n---------------')
	print(pretty_print_bounds(sim_env, phi_denormalized_again),'\n')

	return

def get_means(phi):
	return np.array(phi)[::2]

def get_stdevs(phi):
	return np.array(phi)[1::2]

def pretty_print_bounds(sim_env, phi):
		sim_reset = sim_env.reset()
		if isinstance(sim_env.reset(), np.ndarray) and sim_reset.ndim == 2:
			index_to_name = sim_env.get_attr('dynamics_indexes')[0]
		else:
			index_to_name = sim_env.dynamics_indexes

		return '\n'.join([str(index_to_name[i])+':\t'+str(round(phi[i*2],5))+', '+str(round(phi[i*2+1],5)) for i in range(len(phi)//2)])

def denormalize_bounds(sim_env, logstdevs, phi):
	sim_reset = sim_env.reset()
	if isinstance(sim_env.reset(), np.ndarray) and sim_reset.ndim == 2:
		dim_task = len(sim_env.get_task()[0])
	else:
		dim_task = len(sim_env.get_task())

	parameter_bounds = np.empty((dim_task, 2, 2), float)
	normalized_width = 4
	logstdevs = logstdevs
	
	min_task, max_task = sim_env.get_task_search_bounds()

	for i in range(dim_task):  # Initialize each dynamics parameter dimension
		width = max_task[i]-min_task[i] # Search interval for this parameter

		# MEAN
		parameter_bounds[i, 0, 0] = min_task[i]
		parameter_bounds[i, 0, 1] = max_task[i]


		# STANDARD DEVIATION
		initial_std = width/8	# This may sometimes lead to a stdev smaller than the lower threshold of 0.00001, so take the minimum
		stdev_lower_bound = np.min([0.00001, initial_std-1e-5])
		stdev_upper_bound = width/4

		parameter_bounds[i, 1, 0] = stdev_lower_bound
		parameter_bounds[i, 1, 1] = stdev_upper_bound

		new_phi = []

		for i in range(len(phi)//2):
			norm_mean = phi[i*2]
			norm_std = phi[i*2 + 1]

			mean = (norm_mean * (parameter_bounds[i,0,1]-parameter_bounds[i,0,0]))/normalized_width + parameter_bounds[i,0,0]
			
			if not logstdevs:
				std = (norm_std * (parameter_bounds[i,1,1]-parameter_bounds[i,1,0]))/normalized_width + parameter_bounds[i,1,0]
			else:
				std = parameter_bounds[i,1,0] * ((parameter_bounds[i,1,1]/parameter_bounds[i,1,0])**(norm_std/normalized_width)) # a × (b/a)^(x/10) ≥ 0.

			new_phi.append(mean)
			new_phi.append(std)
	return new_phi
		
def normalize_bounds(sim_env, logstdevs, phi):
    sim_reset = sim_env.reset()
    if isinstance(sim_env.reset(), np.ndarray) and sim_reset.ndim == 2:
        dim_task = len(sim_env.get_task()[0])
    else:
        dim_task = len(sim_env.get_task())

    parameter_bounds = np.empty((dim_task, 2, 2), float)
    normalized_width = 4
    logstdevs = logstdevs

    min_task, max_task = sim_env.get_task_search_bounds()

    for i in range(dim_task):
        width = max_task[i] - min_task[i]

        # MEAN
        parameter_bounds[i, 0, 0] = min_task[i]
        parameter_bounds[i, 0, 1] = max_task[i]

        # STANDARD DEVIATION
        initial_std = width / 8
        stdev_lower_bound = np.min([0.00001, initial_std - 1e-5])
        stdev_upper_bound = width / 4

        parameter_bounds[i, 1, 0] = stdev_lower_bound
        parameter_bounds[i, 1, 1] = stdev_upper_bound

    new_phi = []

    for i in range(len(phi) // 2):
        mean = phi[i * 2]
        std = phi[i * 2 + 1]

        norm_mean = (mean - parameter_bounds[i, 0, 0]) / (parameter_bounds[i, 0, 1] - parameter_bounds[i, 0, 0]) * normalized_width
        if not logstdevs:
            norm_std = (std - parameter_bounds[i, 1, 0]) / (parameter_bounds[i, 1, 1] - parameter_bounds[i, 1, 0]) * normalized_width
        else:
           norm_std = (normalized_width / np.log(parameter_bounds[i, 1, 1] / parameter_bounds[i, 1, 0])) * np.log(std / parameter_bounds[i, 1, 0])

        new_phi.append(norm_mean)
        new_phi.append(norm_std)

    return new_phi







def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='trunkcube-v0', type=str, help='Train gym env')
    parser.add_argument('--run_path', default=None, type=str, help='Path for saving run results')

    parser.add_argument('--unmodeled', default=False, action='store_true', help='Unmodeled setting (implemented only for trunkcube env)')

    parser.add_argument('--logstdevs', default=True, action='store_true', help='RECOMMENDED. Optimize stdevs in log space')

    return parser.parse_args()

if __name__ == '__main__':
	main()