import sys
import time
import pdb
import math
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import nevergrad as ng


class Dropo(object):
	"""
	Domain Randomization Off-Policy Optimization (DROPO)

	Official implementation of DROPO as in the paper "DROPO: Sim-to-Real
	Transfer with Offline Domain Randomization". View the file test_dropo.py
	for a sample usage of the class.
	Public repo at: https://github.com/gabrieletiboni/dropo
	
	Main methods
	-------
	optimize_dynamics_distribution(...)
	    Starts the main DROPO optimization problem

	set_offline_dataset(...)
		Sets the offline dataset of transitions used for running DROPO

	MSE(means), MSE_trajectories(means)
		Compute the MSE in state space with <means> as dynamics parameters
		(respectively for --sparse-mode and trajectory mode)
	"""

	def __init__(self,
				 sim_env,
				 t_length=None,
				 seed=0,
				 scaling=False,
				 sync_parall=True,
				 clip_state=None):
		"""		
		Parameters
		----------
		sim_env : gym simulated environment object.

		t_length : int,
			Lambda hyperparameter as in our paper. Specifies how many
			consecutive actions are executed for each likelihood evaluation.

		seed : int, optional

		scaling : boolean, optional
			If True, each state observation dimension is rescaled to get similar
			scaling across different dimensions.

		sync_parall : boolean, optional
			If True, explicitly adjust the number of evaluations in the opt.
			problem	to match CMA's population size w.r.t. the number of
			parallel workers used.
		"""

		assert (t_length is None or t_length > 0)

		self.sim_env = sim_env
		self.sim_env.reset()

		if hasattr(self.sim_env, 'get_sim_state'):
			self._raw_mjstate = deepcopy(self.sim_env.get_sim_state())  # Save fresh full mjstate for resetting the env at each target state visited
			if self.is_vectorized(self.sim_env):
				self._raw_mjstate = self._raw_mjstate[0]
		else:
			self._raw_mjstate = None

		self.t_length = t_length

		self.scaling = scaling
		self.scaler = (StandardScaler(copy=True) if self.scaling else None)

		self.T = None
		self.seed = seed
		self.sync_parall = sync_parall

		self.clip_state = clip_state

		return
	

	def set_offline_dataset(self, T, indexes=None, n=None, sparse_mode=False):
		"""Sets the offline state transitions used for running DROPO.

		In general, we can select a subset of all of the transitions contained
		in the target dataset `T`, to speed up the opt. problem or
		for debugging. Specify the value `n` to subselect a number of
		trajectories.
		
		Parameters
		----------
		T : dict,
			Offline dataset with keys: ['observations',
										'next_observations',
										'actions',
										'terminals'
									   ]
			T['observations'] : ndarray,
				2D array (t, n) containing the current state information
				for each timestep `t`
			T['next_observations'] : ndarray
      			2D array (t, n) containing the next-state information
      			for each timestep `t`
      		T['actions'] : ndarray
      			2D array (t, a) containing the action commanded to the agent
      			at the current timestep `t`
			T['terminals'] : ndarray
     			1D array (t,) of booleans indicating whether or not the
     			current state transition is terminal (ends the episode)

		indexes : list, optional
			List of integers indicating the subset of transitions used for
			running DROPO. If None, transitions are automatically selected
			based on `n` and `sparse_mode`. (default: none)

		n : int, optional
			Number of trajectories sampled from `T`, if `indexes` is not
			explicitly specified.
			NOTE: if --sparse-mode is selected, then `n` refers to number of
			single sparse transitions instead.

		sparse_mode : boolean, optional
			if True, DROPO is run on random sparse transitions, rather than
			full episodes. In this mode, `n` is treated as the number of transitions.
		"""

		assert ('observations' in T
				and 'next_observations' in T
				and 'actions' in T
				and 'terminals' in T)

		self.T = T

		self.sparse_mode = sparse_mode

		if indexes is None:
			if self.sparse_mode:
				if n is None:	# Use all transitions in `T`
					self.transitions = list(range(len(self.T['observations'])-self.t_length))

				else:	# Get a subset of `n` sparse transitions randomly sampled in `T`
					self.transitions = self._get_subset_sparse_transitions(n)

			else:	# Subsample the first `n` trajectories from `T`
				self.transitions = self._get_ordered_n_trajectories(n)

		else:
			self.transitions = indexes

		if self.scaling:	# Fit scaler
			self.scaler.fit(self.T['next_observations'])

		return


	def get_means(self, phi):
		return np.array(phi)[::2]

	def get_stdevs(self, phi):
		return np.array(phi)[1::2]

	def pretty_print_bounds(self, phi):
		if self.is_vectorized(self.sim_env):
			index_to_name = self.sim_env.get_attr('dynamics_indexes')[0]
		else:
			index_to_name = self.sim_env.dynamics_indexes

		return '\n'.join([str(index_to_name[i])+':\t'+str(round(phi[i*2],5))+', '+str(round(phi[i*2+1],5)) for i in range(len(phi)//2)])




	def optimize_dynamics_distribution(self, opt,
									   budget=1000,
									   additive_variance=False,
									   epsilon=1e-3,
									   sample_size=100,
									   now=1,
									   learn_epsilon=False,
									   normalize=False,
									   logstdevs=False):
		"""Starts the main DROPO optimization problem
		
		Parameters
		----------
		budget : int,
			Number of objective function evaluations for CMA-ES

		additive_variance : boolean,
			if True, add --epsilon to the diagonal of the cov_matrix to regularize the next-state distribution inference

		epsilon : float

		sample_size : int,
			Number of dynamics parameters sampled from the domain randomization distribution

		now : int,
			number of parallel workers

		learn_epsilon : boolean,
			if True, learn the --epsilon parameter by adding it as a parameter to the opt. problem

		normalize : boolean,
			if True, normalize mean and st.devs. in the search space to the interval [0, 4] (recommended)

		logstdevs : boolean,
			if True, denormalize st.devs. for objective function evaluation in log-space
		"""

		dim_task = len(self.sim_env.get_task())

		search_space = []
		search_space_bounds = []

		self.parameter_bounds = np.empty((dim_task, 2, 2), float)
		self.normalized_width = 4

		self.logstdevs = logstdevs

		assert hasattr(self.sim_env, 'set_task_search_bounds')
		self.sim_env.set_task_search_bounds()

		for i in range(dim_task):
			width = self.sim_env.max_task[i]-self.sim_env.min_task[i] # Search interval for this parameter


			# MEAN
			initial_mean = (self.sim_env.min_task[i]+width/4) + np.random.rand()*((self.sim_env.max_task[i]-width/4)-(self.sim_env.min_task[i]+width/4))  # Initialize it somewhat around the center
			if normalize:	# Normalize parameter mean to interval [0, 4]
				search_space.append(ng.p.Scalar(init=self.normalized_width*0.5).set_bounds(lower=0, upper=self.normalized_width))
			else:
				search_space.append(ng.p.Scalar(init=initial_mean).set_bounds(lower=self.sim_env.min_task[i], upper=self.sim_env.max_task[i]))
			
			self.parameter_bounds[i, 0, 0] = self.sim_env.min_task[i]
			self.parameter_bounds[i, 0, 1] = self.sim_env.max_task[i]


			# STANDARD DEVIATION
			initial_std = width/8	# This may sometimes lead to a stdev smaller than the lower threshold of 0.00001, so take the minimum
			stdev_lower_bound = np.min([0.00001, initial_std-1e-5])
			stdev_upper_bound = width/4

			if normalize: # Normalize parameter stdev to interval [0, 4]
				if self.logstdevs:	# Recommended: optimize stdevs in log-space
					search_space.append(ng.p.Scalar(init=self.normalized_width/2).set_bounds(lower=0, upper=self.normalized_width))

				else:	# Linearly optimize stdevs
					search_space.append(ng.p.Scalar(init=self.normalized_width * (initial_std-stdev_lower_bound) / (stdev_upper_bound - stdev_lower_bound) ).set_bounds(lower=0, upper=self.normalized_width))
			
			else:	# Optimize parameters in their original scale (not recommended when using CMA-ES with the identity matrix as starting cov_matrix)
				search_space.append(ng.p.Scalar(init=initial_std).set_bounds(lower=stdev_lower_bound, upper=stdev_upper_bound))

			self.parameter_bounds[i, 1, 0] = stdev_lower_bound
			self.parameter_bounds[i, 1, 1] = stdev_upper_bound


			search_space_bounds.append(self.sim_env.min_task[i])
			search_space_bounds.append(self.sim_env.max_task[i])


		if learn_epsilon:
			search_space.append( ng.p.Log(init=1e-3).set_bounds(lower=1e-15, upper=1e-1) )
			epsilon = None

		params = ng.p.Tuple(*search_space)

		instru = ng.p.Instrumentation(bounds=params,
									  sample_size=sample_size,
									  epsilon=epsilon,
									  additive_variance=additive_variance,
									  learn_epsilon=learn_epsilon,										  
									  normalize=normalize)

		Optimizer = self.__get_ng_optimizer(opt)

		optim = Optimizer(parametrization=instru, budget=budget, num_workers=now)

		start = time.time()

		if not self.sparse_mode:
			loss_function = self._L_target_given_phi_trajectories
			loss_function_parallel = self._L_target_given_phi_trajectories_parallel
		else:
			loss_function = self._L_target_given_phi
			loss_function_parallel = self._L_target_given_phi_parallel


		# Run optimization problem
		if now == 1:
			recommendation = optim.minimize(loss_function)
		else:
			print('Parallelization with num workers:', optim.num_workers)

			if self.sync_parall:

				budget_used = 0
				while budget_used < budget:
					fit, X = [], []

					while len(X) < optim.es.popsize:
						solutions = []

						remaining = optim.es.popsize - len(X)
						curr_now = np.min([now, remaining])

						for nw in range(curr_now):
							solutions.append(optim.ask())
							X.append(solutions[-1])

						f_args = zip(range(now), [dict(item.kwargs) for item in solutions])

						pool = Pool(processes=curr_now)
						res = pool.map(loss_function_parallel, f_args)
						pool.close()
						pool.join()

						for r in res:
							fit.append(r)

					for x, r in zip(X, fit):
						optim.tell(x, r)

					budget_used += optim.es.popsize

				recommendation = optim.recommend() # Get final minimum found
			else:
				for u in range(budget // now):
					xs = []
					for i in range(now):
						xs.append(optim.ask())

					f_args = zip(range(now), [dict(item.kwargs) for item in xs])

					pool = Pool(processes=now)
					res = pool.map(loss_function_parallel, f_args)
					pool.close()
					pool.join()

					for x, r in zip(xs, res):
						optim.tell(x, r)
				
				recommendation = optim.recommend() # Get final minimum found

		end = time.time()
		elapsed = end-start

		if normalize:
			if learn_epsilon:
				return self._denormalize_bounds(recommendation.value[1]['bounds'][:-1]), loss_function(**recommendation.kwargs), elapsed, recommendation.value[1]['bounds'][-1]
			else:
				return self._denormalize_bounds(recommendation.value[1]['bounds']), loss_function(**recommendation.kwargs), elapsed, None
		else:
			if learn_epsilon:
				return recommendation.value[1]['bounds'][:-1], loss_function(**recommendation.kwargs), elapsed, recommendation.value[1]['bounds'][-1]
			else:
				return recommendation.value[1]['bounds'], loss_function(**recommendation.kwargs), elapsed, None

	def is_vectorized(self, env):
		if env.reset().ndim == 2:
			return True
		else:
			return False

	def optimize_dynamics_distribution_resetfree(self,
												 opt,
									   			 budget=1000,
									   			 additive_variance=False,
									   			 epsilon=1e-3,
											     sample_size=100,
											     now=1,
											     learn_epsilon=False,
											     normalize=False,
											     logstdevs=False,
											     clip_episode_length=None,
											     temperatureRegularization=False,
											     wandb=None):
		"""Starts the main DROPO optimization problem

			Uses Reset-Free DROPO version. This way, the simulator
			is not reset at every intermediate state transition, allowing
			DROPO to be used on partially-observable episodes where
			only the initial state configuration is fully known.
		
		Parameters
		----------
		budget : int,
			Number of objective function evaluations for CMA-ES

		additive_variance : boolean,
			if True, add --epsilon to the diagonal of the cov_matrix to regularize the next-state distribution inference

		epsilon : float

		sample_size : int,
			Number of dynamics parameters sampled from the domain randomization distribution

		now : int,
			number of parallel workers

		learn_epsilon : boolean,
			if True, learn the --epsilon parameter by adding it as a parameter to the opt. problem

		normalize : boolean,
			if True, normalize mean and st.devs. in the search space to the interval [0, 4] (recommended)

		logstdevs : boolean,
			if True, denormalize st.devs. for objective function evaluation in log-space
		"""
		assert self.is_vectorized(self.sim_env), f'The source environment is expected to be a Vectorized gym environment for RF-DROPO. See test_resetfree_dropo.py'
		assert sample_size % now == 0, f'Hard constraint violated: --sample_size ({sample_size}) needs to be a multiple of --now ({now}) for RF-DROPO.'
		assert self.sim_env.num_envs == now, 'Sanity check: the number of parallel environments should be indicated by the --now parameter'
		assert hasattr(self.sim_env, 'get_task_search_bounds'), 'The environment needs to have a get_task_search_bounds method returning the opt. search space for each dynamics parameter being optimized.'
		assert np.where(self.T['terminals'][self.transitions]==True)[0].shape[0] == 1, 'WARNING! The current implementation assumes a single trajectory is used. Make sure -n=1' 

		if clip_episode_length is not None:
			self.transitions = self.transitions[:clip_episode_length]
			self.T['terminals'][self.transitions[-1]] = True
			print(f'WARNING! The clip_episode_length is being used. Actual episode length is {len(self.transitions)}')

		if not hasattr(self.sim_env, 'set_sim_state'):
			print(f'WARNING! The current environment used does not have the set_sim_state attribute for resetting the sim configuration. Make sure the initial state distribution is deterministic and fixed.')
		episode_length = len(self.transitions)
		dim_task = len(self.sim_env.get_task()[0])

		search_space = []
		search_space_bounds = []

		self.parameter_bounds = np.empty((dim_task, 2, 2), float)
		self.normalized_width = 4
		self.logstdevs = logstdevs
		
		min_task, max_task = self.sim_env.get_task_search_bounds()

		for i in range(dim_task):  # Initialize each dynamics parameter dimension
			width = max_task[i]-min_task[i] # Search interval for this parameter

			# MEAN
			initial_mean = (min_task[i]+width/4) + np.random.rand()*((max_task[i]-width/4)-(min_task[i]+width/4))  # Initialize it somewhat around the center
			if normalize:	# Normalize parameter mean to interval [0, 4]
				search_space.append(ng.p.Scalar(init=self.normalized_width*0.5).set_bounds(lower=0, upper=self.normalized_width))
			else:
				search_space.append(ng.p.Scalar(init=initial_mean).set_bounds(lower=min_task[i], upper=max_task[i]))
			
			self.parameter_bounds[i, 0, 0] = min_task[i]
			self.parameter_bounds[i, 0, 1] = max_task[i]


			# STANDARD DEVIATION
			initial_std = width/8	# This may sometimes lead to a stdev smaller than the lower threshold of 0.00001, so take the minimum
			stdev_lower_bound = np.min([0.00001, initial_std-1e-5])
			stdev_upper_bound = width/4

			if normalize: # Normalize parameter stdev to interval [0, 4]
				if self.logstdevs:	# Recommended: optimize stdevs in log-space
					search_space.append(ng.p.Scalar(init=self.normalized_width/2).set_bounds(lower=0, upper=self.normalized_width))

				else:	# Linearly optimize stdevs
					search_space.append(ng.p.Scalar(init=self.normalized_width * (initial_std-stdev_lower_bound) / (stdev_upper_bound - stdev_lower_bound) ).set_bounds(lower=0, upper=self.normalized_width))
			
			else:	# Optimize parameters in their original scale (not recommended when using CMA-ES with the identity matrix as starting cov_matrix)
				search_space.append(ng.p.Scalar(init=initial_std).set_bounds(lower=stdev_lower_bound, upper=stdev_upper_bound))

			self.parameter_bounds[i, 1, 0] = stdev_lower_bound
			self.parameter_bounds[i, 1, 1] = stdev_upper_bound

			search_space_bounds.append(min_task[i])
			search_space_bounds.append(max_task[i])


		params = ng.p.Tuple(*search_space)
		instru = ng.p.Instrumentation(phi=params,
									  sample_size=sample_size,
									  epsilon=epsilon,
									  additive_variance=additive_variance,								  
									  normalize=normalize)

		Optimizer = self.__get_ng_optimizer(opt)
		optim = Optimizer(parametrization=instru, budget=budget, num_workers=now)

		# Run optimization problem
		start = time.time()
		time_per_iter = 0

		print('CMA population size:', optim.es.popsize)
		wandb.run.summary["CMA_pop_size"] = optim.es.popsize
		tot_opt_iterations = math.ceil(budget / optim.es.popsize)
		temperature = episode_length

		for i in range(tot_opt_iterations):
			print(f'Opt. iteration {i+1}/{tot_opt_iterations}. Est. time left: {(self._seconds_to_human_time(time_per_iter*(tot_opt_iterations-i)) if time_per_iter != 0 else "-")}')
			wandb.run.summary["opt_iteration_tot"] = tot_opt_iterations
			wandb.run.summary["opt_iteration_i"] = i+1
			wandb.run.summary["est_time_left"] = (self._seconds_to_human_time(time_per_iter*(tot_opt_iterations-i)) if time_per_iter != 0 else "-")
			if temperatureRegularization:  # Start considering a shorter-time horizon, then grow.
				temperature = self._get_temperature_regularization(i, tot_opt_iterations, episode_length)				

			fit, X = [], []
			while len(X) < optim.es.popsize:  # One obj-fun eval at a time (parallelization is performed inside the evaluation)
				x = optim.ask()
				fit_next = self._L_target_given_phi_resetfree(**{'temperature': temperature, **x.kwargs})

				X.append(x)
				fit.append(fit_next)

			for x, r in zip(X, fit):
				optim.tell(x, r)

			time_per_iter = (time.time() - start)/(i+1)

			if wandb is not None:
				self._plot_phi_on_wandb(wandb, optim.recommend().value[1]['phi'])
				wandb.log({"temperature": temperature})

		elapsed = time.time()-start

		recommendation = optim.recommend() # Get final minimum found

		if normalize:
			return self._denormalize_bounds(recommendation.value[1]['phi']), self._L_target_given_phi_resetfree(**recommendation.kwargs), elapsed
		else:
			return recommendation.value[1]['phi'], self._L_target_given_phi_resetfree(**recommendation.kwargs), elapsed


	def _get_temperature_regularization(self, i, K, n):
		"""Temperature parameter T that clips the episode length.
			
			T starts at 1 and decays exponentially
			to episode_length. At half the number of iterations,
			T is at 90% the episode_length	
		
		Args:
			i: int, current iteration
			K: int, tot number of iterations
			n: int, episode length
		"""
		c = 2/K*np.log(10)  # T is at 90% the episode length at K/2
		temperature = n*(1 - np.exp(-c*i))
		return max(1, int(temperature))

	def _L_target_given_phi_resetfree(self,
									  phi,  # bounds
									  sample_size,
									  epsilon,
									  additive_variance,
									  normalize,
									  temperature=None):

		if normalize:
			phi = self._denormalize_bounds(phi)

		n_cycles = sample_size // self.sim_env.num_envs

		# n_episodes = np.where(self.T['terminals'][self.transitions] == True)[0].shape[0]
		if temperature is not None:
			transitions = self.transitions[:temperature]
		else:
			transitions = self.transitions
		episode_length = len(transitions)

		obs_space_dim = self.sim_env.reset()[0].shape[0]
		source_s_prime_samples_per_transition = np.zeros((len(transitions), sample_size, obs_space_dim), float)
		real_s_prime_per_transition = np.zeros((len(transitions), obs_space_dim), float)

		### Collect Monte-Carlo samples
		for k in range(n_cycles):
			self.sim_env.reset()

			if hasattr(self.sim_env, 'set_sim_state'):  # Reset env to initial state configuration
				self.sim_env.set_sim_state(self.sim_env.get_initial_mjstate(self.T['observations'][0], self._raw_mjstate))

			# Sample tasks from phi
			sampled_tasks = self.sample_truncnormal(phi, self.sim_env.num_envs )
			self.sim_env.set_task(sampled_tasks)

			for t in transitions:
				real_s_prime = self.T['next_observations'][t]
				action = self.T['actions'][t]
				vecAction = np.repeat(action[np.newaxis, :], self.sim_env.num_envs, axis=-2)
				s_prime, reward, done, _ = self.sim_env.step(vecAction)

				if self.scaling:
					raise NotImplementedError
					# real_s_prime = self.scaler.transform(real_s_prime.reshape(1,-1))[0]
					# mapped_sample = self.scaler.transform(mapped_sample.reshape(1, -1))[0]

				source_s_prime_samples_per_transition[t, (k*self.sim_env.num_envs):((k+1)*self.sim_env.num_envs), :] = s_prime[:, :]
				real_s_prime_per_transition[t, :] = real_s_prime

		### Estimate likelihood of real s_prime
		likelihood = 0
		for t in transitions:
			s_prime_samples = source_s_prime_samples_per_transition[t]
			real_s_prime = real_s_prime_per_transition[t]

			# Infer next-state distribution parameters
			cov_matrix = np.cov(s_prime_samples, rowvar=0)
			mean = np.mean(s_prime_samples, axis=0)

			if additive_variance:
				cov_matrix = cov_matrix + np.diag(np.repeat(epsilon, mean.shape[0]))

			multi_normal = multivariate_normal(mean=mean, cov=cov_matrix, allow_singular=True)

			logdensity = multi_normal.logpdf(real_s_prime)
			likelihood += logdensity

		if np.isinf(likelihood):
			raise ValueError('WARNING: infinite likelihood encountered.')

		return -1*likelihood


	def _L_target_given_phi_parallel(self, args):
		i, args = args
		np.random.seed(i+self.seed)

		return self._L_target_given_phi(**args)

	def _L_target_given_phi_trajectories_parallel(self, args):
		i, args = args
		np.random.seed(i+self.seed)

		return self._L_target_given_phi_trajectories(**args)


	def _L_target_given_phi(self,
						   bounds,
						   sample_size=100,
						   epsilon=1e-3,
						   additive_variance=False,
						   learn_epsilon=False,
						   normalize=False):
		"""Objective function evaluation for --sparse-mode"""

		likelihood = 0

		if learn_epsilon:
			epsilon = bounds[-1]
			bounds = bounds[:-1]

		if normalize:
			bounds = self._denormalize_bounds(bounds)

		sample = self.sample_truncnormal(bounds, sample_size*len(self.transitions))

		t_length = self.t_length

		# For each transition, map the sample to the state space,
		# estimate the next-state distribution, and compute the likelihood
		# of the real next state.
		for k, t in enumerate(self.transitions):

			ob = self.T['observations'][t]
			target_ob_prime = self.T['next_observations'][t+t_length-1]

			mapped_sample = []

			for ss in range(sample_size):
				r = self.sim_env.reset()

				task = sample[k*sample_size + ss]
				self.sim_env.set_task(*task)

				self.sim_env.set_sim_state(self.sim_env.get_full_mjstate(ob, self._raw_mjstate))

				if hasattr(self.sim_env.sim, 'forward'):
					self.sim_env.sim.forward()
				else:
					raise ValueError('No forward() method found. This environment is not supported.')

				for j in range(t, t+t_length):
					action = self.T['actions'][j]
					s_prime, reward, done, _ = self.sim_env.step(action)

				mapped_sample.append(s_prime)

			mapped_sample = np.array(mapped_sample)

			if self.scaling:
				target_ob_prime = self.scaler.transform(target_ob_prime.reshape(1,-1))[0]
				mapped_sample = self.scaler.transform(mapped_sample)

			# Infer covariance matrix and mean
			cov_matrix = np.cov(mapped_sample, rowvar=0)
			mean = np.mean(mapped_sample, axis=0)

			if additive_variance:
				cov_matrix = cov_matrix + np.diag(np.repeat(epsilon, mean.shape[0]))

			multi_normal = multivariate_normal(mean=mean, cov=cov_matrix, allow_singular=True)

			logdensity = multi_normal.logpdf(target_ob_prime)
			likelihood += logdensity

		if np.isinf(likelihood):
			print('WARNING: infinite likelihood encountered.')

		return -1*likelihood

	
	def _L_target_given_phi_trajectories(self,
										bounds,
										sample_size=100,
										additive_variance=False,
										epsilon=1e-3,
										normalize=False,
										learn_epsilon=False):
		"""Objective function evaluation for standard trajectory mode"""


		if learn_epsilon:
			epsilon = bounds[-1]
			bounds = bounds[:-1]

		if normalize:
			bounds = self._denormalize_bounds(bounds)


		sample = self.sample_truncnormal(bounds, sample_size)

		r = self.sim_env.reset()

		mapped_sample_per_transition = np.zeros((len(self.transitions), sample_size, r.shape[0]), float)
		target_ob_prime_per_transition = np.zeros((len(self.transitions), r.shape[0]), float)

		lambda_steps = self.t_length

		effective_transitions = []
		first_pass = True

		for i, ss in enumerate(range(sample_size)):

			task = sample[ss]
			self.sim_env.set_task(*task)

			reset_next = True
			lambda_count = -1

			# Reproduce trajectories with this task from the phi
			for k, t in enumerate(self.transitions):

				lambda_count += 1
				if lambda_count < 0 or lambda_count%lambda_steps != 0:
					continue
				
				# Check if any of the next lambda_steps transitions are ending states, including current one
				for l in range(k, k+lambda_steps):
					if self.T['terminals'][self.transitions[l]] == True:
						reset_next = True
						lambda_count = -1
						break
				if lambda_count == -1:
					continue
				if first_pass:
					effective_transitions.append(k)

				ob = self.T['observations'][t]
				target_ob_prime = self.T['next_observations'][t+lambda_steps-1]

				if reset_next:	# Initialize simulator at the beginning of the episode
					r = self.sim_env.reset()
					self.sim_env.set_sim_state(self.sim_env.get_initial_mjstate(ob, self._raw_mjstate))

					if hasattr(self.sim_env.sim, 'forward'):
						self.sim_env.sim.forward()
					else:
						raise ValueError('No forward() method found. This environment is not supported.')

					reset_next = False

				else:	# Reset simulator after last transition
					self.sim_env.set_sim_state(self.sim_env.get_full_mjstate(ob, self.sim_env.get_sim_state()))

					if hasattr(self.sim_env.sim, 'forward'):
						self.sim_env.sim.forward()
					else:
						raise ValueError('No forward() method found. This environment is not supported.')

				
				for j in range(t, t+lambda_steps):
					action = self.T['actions'][j]
					s_prime, reward, done, _ = self.sim_env.step(action)

				mapped_sample = np.array(s_prime)

				if self.scaling:
					target_ob_prime = self.scaler.transform(target_ob_prime.reshape(1,-1))[0]
					mapped_sample = self.scaler.transform(mapped_sample.reshape(1, -1))[0]

				mapped_sample_per_transition[k, i, :] = mapped_sample
				target_ob_prime_per_transition[k, :] = target_ob_prime

			first_pass = False

		likelihood = 0
		
		for i, k in enumerate(effective_transitions):

			mapped_sample = mapped_sample_per_transition[k]
			target_ob_prime = target_ob_prime_per_transition[k]

			# if hasattr(self.sim_env, 'set_static_goal'):
			if self.clip_state is not None:
				mapped_sample = mapped_sample[:, :self.clip_state]
				target_ob_prime = target_ob_prime[:self.clip_state]

			# Infer next-state distribution parameters
			cov_matrix = np.cov(mapped_sample, rowvar=0)
			mean = np.mean(mapped_sample, axis=0)
			
			if additive_variance:
				cov_matrix = cov_matrix + np.diag(np.repeat(epsilon, mean.shape[0]))

			multi_normal = multivariate_normal(mean=mean, cov=cov_matrix, allow_singular=True)

			logdensity = multi_normal.logpdf(target_ob_prime)
			likelihood += logdensity

		if np.isinf(likelihood):
			print('WARNING: infinite likelihood encountered.')

		return -1*likelihood


	def _denormalize_bounds(self, phi):
		"""Denormalize means and stdevs in phi back to their original space
		for evaluating the likelihood."""

		new_phi = []

		for i in range(len(phi)//2):
			norm_mean = phi[i*2]
			norm_std = phi[i*2 + 1]

			mean = (norm_mean * (self.parameter_bounds[i,0,1]-self.parameter_bounds[i,0,0]))/self.normalized_width + self.parameter_bounds[i,0,0]
			
			if not self.logstdevs:
				std = (norm_std * (self.parameter_bounds[i,1,1]-self.parameter_bounds[i,1,0]))/self.normalized_width + self.parameter_bounds[i,1,0]
			else:
				std = self.parameter_bounds[i,1,0] * ((self.parameter_bounds[i,1,1]/self.parameter_bounds[i,1,0])**(norm_std/self.normalized_width)) # a × (b/a)^(x/10) ≥ 0.

			new_phi.append(mean)
			new_phi.append(std)

		return new_phi


	def MSE(self, means):
		"""Compute the MSE in state space with means as dynamics parameters (--sparse-mode).
		
		Refer to our paper (Section IV.A) for a detailed explanation on how
		the MSE is computed.
		"""
		distance = 0
		
		task = np.array(means)
		self.sim_env.set_task(*task)

		for t in self.transitions:

			ob = self.T['observations'][t]
			action = self.T['actions'][t]
			target_ob_prime = self.T['observations'][t+1]

			mapped_sample = []

			self.sim_env.set_sim_state(self.sim_env.get_full_mjstate(ob, self._raw_mjstate))
			s_prime, reward, done, _ = self.sim_env.step(action)

			mapped_sample.append(list(s_prime))
			mapped_sample = np.array(mapped_sample)

			if self.scaling:
				mapped_sample = self.scaler.transform(mapped_sample)
				target_ob_prime = self.scaler.transform(target_ob_prime.reshape(1,-1))[0]

			mapped_sample = mapped_sample[0,:]

			distance += np.linalg.norm(target_ob_prime-mapped_sample)**2

		mean_distance = distance / len(self.transitions)

		return mean_distance



	def MSE_trajectories(self, means):
		"""Compute the MSE in state space with means as dynamics parameters.
		
		Refer to our paper (Section IV.A) for a detailed explanation on how
		the MSE is computed.
		"""
		distance = []

		task = np.array(means)
		self.sim_env.set_task(*task)

		reset_next = True

		for k, t in enumerate(self.transitions):

			if self.T['terminals'][t] == True:
				reset_next = True
				continue

			target_s = self.T['observations'][t]
			target_s_prime = self.T['observations'][t+1]

			if reset_next:
				r = self.sim_env.reset()
				self.sim_env.set_sim_state(self.sim_env.get_initial_mjstate(target_s, self._raw_mjstate))
				if hasattr(self.sim_env.sim, 'forward'):
					self.sim_env.sim.forward()
				elif hasattr(self.sim_env.sim.env.sim, 'forward'):
					self.sim_env.sim.env.sim.forward()
				else:
					raise ValueError('No forward() method found. This environment is not supported.')

				reset_next = False
			else:
				self.sim_env.set_sim_state(self.sim_env.get_full_mjstate(target_s, self.sim_env.get_sim_state()))

			action = self.T['actions'][t]
			sim_s_prime, reward, done, _ = self.sim_env.step(action)

			sim_s_prime = np.array(sim_s_prime)

			if self.scaling:
				sim_s_prime = self.scaler.transform(sim_s_prime.reshape(1, -1))[0]
				target_s_prime = self.scaler.transform(target_s_prime.reshape(1, -1))[0]

			distance.append(np.linalg.norm(sim_s_prime - target_s_prime)**2)

		return np.mean(distance)


	def sample_truncnormal(self, phi, size=1):
		"""Sample <size> observations from the dynamics distribution parameterized by <phi>.
		
		A truncnormal density function is used, truncating values more than
		2 standard deviations away => happens around 5% of the time otherwise.
		"""
		a,b = -2, 2
		sample = []

		for i in range(len(phi)//2):
			mean = phi[i*2]
			std = phi[i*2 + 1]

			if hasattr(self.sim_env, 'get_task_lower_bound'):
				lower_bound = self.sim_env.get_task_lower_bound(i)
			else:
				lower_bound = 0.0001

			if hasattr(self.sim_env, 'get_task_upper_bound'):
				upper_bound = self.sim_env.get_task_upper_bound(i)
			else:
				upper_bound = 1000000000

			
			# Make sure all samples belong to [lower_bound, upper_bound]
			attempts = 0
			obs = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
			while np.any((obs<lower_bound) | (obs>upper_bound)):

				obs[((obs < lower_bound) | (obs > upper_bound))] = truncnorm.rvs(a, b, loc=mean, scale=std, size=len(obs[((obs < lower_bound) | (obs > upper_bound))]))

				attempts += 1
				if attempts > 20:
					obs[obs < lower_bound] = lower_bound
					obs[obs > upper_bound] = upper_bound
					print(f"Warning - Not all samples were above >= {lower_bound} or below {upper_bound} after 20 attempts. Setting them to their min/max bound values, respectively.")

			sample.append(obs)

		return np.array(sample).T


	def _distance(self, target, sim_state):

		if self.scaling:
			d = np.linalg.norm(
							   	self.scaler.transform(target.reshape(1,-1))
							    - self.scaler.transform(sim_state.reshape(1,-1))
							  )**2
		else:
			d = np.linalg.norm(target - sim_state)**2

		return d


	def _get_trajectories_indexes(self, n=None):
		"""Returns starting index of each trajectory"""

		terminals = self.T['terminals']

		arr = np.where(terminals==True)[0]

		arr = np.insert(-1, 1, arr) # Insert first trajectory
		arr = arr[:-1] # Remove last terminal state (no trajectory after it)
		arr = arr+1 # Starting state is the one after the previous episode has finished

		if n is not None:
			ts = np.random.choice(arr, size=n, replace=False)
		else:
			ts = list(arr)

		return ts

	def _get_ordered_n_trajectories(self, n=None):
		"""Returns indexes of n trajectories
		randomly sampled from self.T"""

		terminals = self.T['terminals']

		arr = np.where(terminals==True)[0]

		arr = np.insert(-1, 1, arr)   # Insert first trajectory
		arr = arr[:-1]   # Remove last terminal state (no trajectory after it)
		arr = arr+1   # Starting state is the one after the previous episode has finished

		if n is not None:
			# ts = np.random.choice(arr, size=n, replace=False)
			ts = arr[:n]
		else:
			ts = list(arr)


		transitions = []
		for t in ts:
			duration = np.argmax(self.T['terminals'][t:])

			for toadd in range(t, t+duration+1):
				transitions.append(toadd)

		return transitions

	def _get_subset_sparse_transitions(self, n):

		if self.t_length < 1:
			raise ValueError('Invalid lambda value')

		if n < 1:
			raise ValueError('Invalid number of transitions')

		c = 0
		valid_ts = []
		size = len(self.T['observations'])

		while c < n:
			t = np.random.randint(0, size-self.t_length)

			valid = True
			for i in range(t, t+self.t_length):
				if self.T['terminals'][i]:
					valid = False
					break

			if not valid:
				continue

			valid_ts.append(t)
			c+=1

		return valid_ts

	def _seconds_to_human_time(self, seconds):
		if seconds < 60:
			return str(round(seconds))+'s'
		else:
			minutes = seconds/60
			hours = int(minutes//60)
			minutes = int(minutes%60)

			hours_str = str(hours)+"h " if hours > 0 else "";
			minutes_str = str(minutes)+"m"

			return hours_str+minutes_str
	
	def __get_ng_optimizer(self, opt_string):
		"""Get Nevergrad optimizer
		
		https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
		"""

		opts = {
			'oneplusone': ng.optimizers.OnePlusOne, # simple robust method for continuous parameters with num_workers < 8.
			'bayesian': ng.optimizers.BO, # Bayesian optimization
			'twopointsde': ng.optimizers.TwoPointsDE,  # excellent in many cases, including very high num_workers
			'pso': ng.optimizers.PSO, # excellent in terms of robustness, high num_workers ok
			'tbpsa': ng.optimizers.TBPSA, # excellent for problems corrupted by noise, in particular overparameterized (neural) ones; very high num_workers ok).
			'random': ng.optimizers.RandomSearch, # the classical random search baseline; don’t use softmax with this optimizer.
			'meta': ng.optimizers.NGOpt, # “meta”-optimizer which adapts to the provided settings (budget, number of workers, parametrization) and should therefore be a good default.
			'cma': ng.optimizers.CMA # CMA-ES (https://en.wikipedia.org/wiki/CMA-ES)
		}

		if opt_string not in opts:
			raise NotImplementedError('Optimizer not found')

		return opts[opt_string]


	def _plot_phi_on_wandb(self, wandb, phi):
		"""Plots distribution phi on wandb"""
		if self.is_vectorized(self.sim_env):
			index_to_name = self.sim_env.get_attr('dynamics_indexes')[0]
		else:
			index_to_name = self.sim_env.dynamics_indexes

		phi = self._denormalize_bounds(phi)
		for i, (mean, stdev) in enumerate(zip(self.get_means(phi), self.get_stdevs(phi))):
			wandb.log({index_to_name[i]+"_MEAN": mean})
			wandb.log({index_to_name[i]+"_STDEV": stdev})


		
