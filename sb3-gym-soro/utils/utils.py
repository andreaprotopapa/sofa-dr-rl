import numpy as np
import random
import string
import socket
import os
import pathlib
import glob
import pdb
from datetime import datetime

import gym

def get_current_date():
	return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

def get_run_name(args):
	current_date = get_current_date()
	return str(current_date)+"_"+str(args.env)+"_"+str(args.algo)+"_t"+str(args.timesteps)+"_seed"+str(args.seed)+"_"+socket.gethostname()

def get_random_string(n=5):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

def set_seed(seed):
	if seed > 0:
		np.random.seed(seed)

def create_dir(path):
	try:
		os.mkdir(os.path.join(path))
	except OSError as error:
		# print('Dir esiste già:', path)
		pass

def create_dirs(path):
	try:
		os.makedirs(os.path.join(path))
	except OSError as error:
		pass

def collect_offline_data(env, n, policy=None, shuffle=False):
	states = []
	actions = []
	next_states = []
	terminals = []

	state = env.reset()

	n_traj = 0
	while n_traj < n:
		
		if policy is not None:
			action, _ = policy.predict(state, deterministic=True)
		else:
			action = env.action_space.sample() # action = np.random.randn((env.action_space.shape[0]))

		states.append(state)
		actions.append(action)

		state, reward, done, info = env.step(action)

		next_states.append(state)

		if done:
			terminals.append(True)
			state = env.reset()
			n_traj += 1
		else:
			terminals.append(False)


	ds = []
	for state, next_state, action, terminal in zip(states, next_states, actions, terminals):
		ds.append((state,next_state,action,terminal))
	
	if shuffle:
		random.shuffle(ds)

	T = {'observations': None, 'next_observations': None, 'actions': None, 'terminals': None}
	T['observations'] = np.empty((0, len(states[0])), float)
	T['next_observations'] = np.empty((0, len(states[0])), float)
	T['actions'] = np.empty((0, len(actions[0])), float)
	T['terminals'] = np.empty((0,), bool)

	for state, next_state, action, terminal in (ds):
		T['observations'] = np.append(T['observations'], np.array([state]), axis=0)
		T['next_observations'] = np.append(T['next_observations'], np.array([next_state]), axis=0)
		T['actions'] = np.append(T['actions'], np.array([action]), axis=0)
		T['terminals'] = np.append(T['terminals'], np.array([terminal]), axis=0)
	
	T['terminals'][-1] = True

	return T


def collect_offline_data_clipping(env, n, policy=None, shuffle=False, clipping=None, env_name=None, save_dataset=False):
	states = []
	actions = []
	next_states = []
	terminals = []

	if 'Noisy' in env_name:
		noisy = True
		noise_level = env.noise_level
		standard_env_name = env_name.replace("Noisy", "")
		env = gym.make(standard_env_name)
	else:
		noisy = False

	assert noisy or 'noisy' not in env_name.lower()

	env.set_endless(True)

	state = env.reset()

	n_traj = 0
	n_transitions = 1
	while n_traj < n:
		if n_transitions == 1:
			print("Collect trajectory n. ", n_traj)
		if policy is not None:
			if noisy:
				action, _ = policy.predict(state + np.sqrt(noise_level)*np.random.randn(state.shape[0]), deterministic=True)
			else:
				action, _ = policy.predict(state, deterministic=True)
		else:
			action = env.action_space.sample() # action = np.random.randn((env.action_space.shape[0]))

		states.append(state)
		actions.append(action)

		state, reward, done, info = env.step(action)

		if noisy:
			next_states.append(state + np.sqrt(noise_level)*np.random.randn(state.shape[0]))
		else:
			next_states.append(state)

		if clipping is not None and n_transitions >= clipping:
			terminals.append(True)
			state = env.reset()
			n_traj += 1
			n_transitions = 1
			continue
		else:
			terminals.append(done)

		if done:
			state = env.reset()
			n_traj += 1

		n_transitions += 1

	env.set_endless(False)

	ds = []
	for state, next_state, action, terminal in zip(states, next_states, actions, terminals):
		ds.append((state,next_state,action,terminal))
	
	if shuffle:
		random.shuffle(ds)
	T = {'observations': None, 'next_observations': None, 'actions': None, 'terminals': None}
	T['observations'] = np.empty((0, len(states[0])), float)
	T['next_observations'] = np.empty((0, len(states[0])), float)
	T['actions'] = np.empty((0, len(actions)), float)
	T['terminals'] = np.empty((0,), bool)

	for state, next_state, action, terminal in (ds):
		T['observations'] = np.append(T['observations'], np.array([state]), axis=0)
		T['next_observations'] = np.append(T['next_observations'], np.array([next_state]), axis=0)
		T['actions'] = np.append(T['actions'], np.array([action]))
		T['terminals'] = np.append(T['terminals'], np.array([terminal]), axis=0)
	
	T['terminals'][-1] = True

	if save_dataset:
		save_path_dataset = str(pathlib.Path(__file__).parent.absolute())+"/../Dataset"
		env_name = env.__getattr__("config")["scene"]
		os.makedirs(save_path_dataset, exist_ok=True)
		save_path_dataset_env = save_path_dataset+"/"+env_name
		os.makedirs(save_path_dataset_env, exist_ok=True)
		np.save(save_path_dataset_env+"/"+datetime.now().strftime('%Y%m%d-%H%M%S')+"_"+str(n)+"trajectories.npy", T)


	return T


def get_ordered_n_trajectories(T, n=None):
	"""Returns observations and actions of n trajectories
	sampled from T"""

	terminals = T['terminals']

	arr = np.where(terminals==True)[0]

	arr = np.insert(-1, 1, arr)   # Insert first trajectory
	arr = arr[:-1]   # Remove last terminal state (no trajectory after it)
	arr = arr+1   # Starting state is the one after the previous episode has finished

	if n is not None:
		ts = arr[:n]
	else:
		ts = list(arr)

	transitions = []
	states = []
	actions = []

	for t in ts:
		duration = np.argmax(T['terminals'][t:])

		c_states = []
		c_actions = []

		c_states.append(T['observations'][t]) # Starting state
		# c_actions.append(T['actions'][t])

		for toadd in range(t, t+duration+1):
			transitions.append(toadd)

			c_states.append(T['next_observations'][toadd])
			c_actions.append(T['actions'][toadd])

			# pdb.set_trace()

		# c_actions.append(T['actions'][toadd])
		states.append(np.array(c_states))
		actions.append(np.array(c_actions))
	
	return states, actions


def load_data_from_file(path):
	"""Load offline dataset from path"""
	T = np.load(path, allow_pickle=True).item()
	return T


def save_dataset(T, path, prefix=''):
	try:
		os.makedirs(os.path.join(path), exist_ok=True)
	except OSError as error:
		raise ValueError(f"Dataset dir already exists: {path}")
	
	np.save(os.path.join(path, prefix+'_observations.npy'), T['observations'])
	np.save(os.path.join(path, prefix+'_nextobservations.npy'), T['next_observations'])
	np.save(os.path.join(path, prefix+'_actions.npy'), T['actions'])
	np.save(os.path.join(path, prefix+'_terminals.npy'), T['terminals'])