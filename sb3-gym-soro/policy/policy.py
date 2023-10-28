"""
	Wrapper class for policy training and evaluation
"""
import sys
import numpy as np
import torch
import pdb
import os
import pathlib
import datetime
import time

from .callbacks import WandbRecorderCallback, CheckpointLastCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.vec_env import VecEnv

class Policy:
	
	def __init__(self,
				 algo=None,
				 env=None,
				 lr=0.0003,
				 batch_size=64,
				 size_layer=[128, 128],
				 device='cpu',
				 seed=None,
				 load_from_pathname=None,
				 reset_num_timesteps=True):

		# assert isinstance(env, VecEnv)
		# else: env = make_vec_env(env, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)

		self.seed = seed
		self.device = device
		self.env = env
		self.algo = algo
		self.lr=lr
		self.batch_size=batch_size
		self.size_layer=size_layer
		self.reset_num_timesteps=reset_num_timesteps

		if load_from_pathname is None:
			self.model = self.create_model(algo, lr=lr, batch_size=batch_size, size_layer=size_layer)
		else:
			self.model = self.load_model(algo, load_from_pathname)

		return
	
	def create_model(self, algo, lr, batch_size, size_layer):
		if algo == 'ppo':
			policy_kwargs = dict(activation_fn=torch.nn.Tanh,
			                     net_arch=[dict(pi=size_layer, vf=size_layer)])
			model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs,
						learning_rate=lr, batch_size=batch_size, verbose=0, seed=self.seed, device=self.device)

		elif algo == 'sac':
			policy_kwargs = dict(activation_fn=torch.nn.Tanh,
			                     net_arch=dict(pi=size_layer, qf=size_layer))
			model = SAC("MlpPolicy", self.env, policy_kwargs=policy_kwargs,
						learning_rate=lr, batch_size=batch_size, verbose=0, seed=self.seed, device=self.device)
		else:
			raise ValueError(f"RL Algo not supported: {algo}")

		return model

	def load_model(self, algo, pathname):
		if algo == 'ppo':
			model = PPO.load(pathname, env=self.env, device=self.device,
							 lr=self.lr, batch_size=self.batch_size, size_layer=self.size_layer,
							 custom_objects={'action_space': self.env.action_space, 'observation_space': self.env.observation_space} )
		elif algo == 'sac':
			model = SAC.load(pathname, env=self.env, device=self.device,
							 lr=self.lr, batch_size=self.batch_size, size_layer=self.size_layer)
		else:
			raise ValueError(f"RL Algo not supported: {algo}")
		return model

	def train(self,
			  timesteps=2000,
			  stopAtRewardThreshold=True,
			  n_eval_episodes=50,
			  eval_freq=1000,
			  best_model_save_path=None,
			  return_best_model=True,
			  wandb_loss_suffix="",
			  verbose=0,
			  save_freq=None,
			  keep_prev_ckpt=False):
		"""Train a model

			1. Setup callbacks
			2. Train model
			3. Find best model and return it
		"""
		if stopAtRewardThreshold and self.model.get_env().env_method('get_reward_threshold')[0] is not None:
			stop_at_reward_threshold = StopTrainingOnRewardThreshold(reward_threshold=self.model.get_env().env_method('get_reward_threshold')[0], verbose=1)
		else:
			stop_at_reward_threshold = None

		wandb_recorder = WandbRecorderCallback(eval_freq= max(eval_freq // self.env.num_envs, 1), wandb_loss_suffix=wandb_loss_suffix) # Plot stuff on wandb
		n_eval_episodes = n_eval_episodes
		eval_callback = EvalCallback(self.env,
		                             best_model_save_path=best_model_save_path,
		                             # log_path='./logs/',
		                             eval_freq= max(eval_freq // self.env.num_envs, 1),
		                             n_eval_episodes=n_eval_episodes,
		                             deterministic=True,
		                             callback_after_eval=wandb_recorder,
		                             callback_on_new_best=stop_at_reward_threshold,
		                             verbose=verbose,
		                             render=False)
		if save_freq is None:
			save_freq = eval_freq
		checkpoint_callback = CheckpointLastCallback(
									save_freq = max(save_freq // self.env.num_envs, 1),
									save_path=best_model_save_path+"logs/",
									name_prefix="model_ckpt",
									save_replay_buffer=True,
									save_vecnormalize=True,
									verbose=2,
									keep_prev=keep_prev_ckpt
									)
									
		callback_list = CallbackList([eval_callback, checkpoint_callback])
		self.model.learn(total_timesteps=timesteps, callback=callback_list, reset_num_timesteps=self.reset_num_timesteps)

		if return_best_model:   # Find best model
			reward_final, std_reward_final = self.eval(n_eval_episodes=n_eval_episodes)

			assert os.path.exists(os.path.join(best_model_save_path, "best_model.zip")), "best_model.zip hasn't been saved because too few evaluations have been performed. Check --eval_freq and -t"
			best_model = self.load_model(self.algo, os.path.join(best_model_save_path, "best_model.zip"))
			reward_best, std_reward_best = evaluate_policy(best_model, best_model.get_env(), n_eval_episodes=n_eval_episodes)

			if reward_final > reward_best:
				best_policy = self.state_dict()
				best_mean_reward, best_std_reward = reward_final, std_reward_final
				which_one = 'final'
			else:
				best_policy = best_model.policy.state_dict()
				best_mean_reward, best_std_reward = reward_best, std_reward_best
				which_one = 'best'

			return best_mean_reward, best_std_reward, best_policy, which_one
		else:
			return self.eval(n_eval_episodes)

	def eval(self, n_eval_episodes=50, render=False):
		mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=n_eval_episodes, render=render)
		return mean_reward, std_reward
	
	def test(self, n_test_episodes=5, render=False, save_dataset=False, save_results=None, random_dynamics=False, seed=None):
		r, final_r = 0, 0
		infos = []
		distances = []
		cumulative_rewards = []
		if random_dynamics:
			config_true = self.env.get_attr("config")[0]
			config_true.update({'test': True})
			self.env.set_attr("config", config_true, 0)
		else:
			config_true = self.env.__getattr__("config")
			config_true.update({'test': True})
			self.env.__setattr__("config", config_true)
		if save_dataset:
				save_path_dataset = str(pathlib.Path(__file__).parent.absolute())+"/../Dataset"
				env_name = self.env.__getattr__("config")["scene"]
				os.makedirs(save_path_dataset, exist_ok=True)
				save_path_dataset_env = save_path_dataset+"/"+env_name
				os.makedirs(save_path_dataset_env, exist_ok=True)
		dataset = {
				'observations': [],
				'actions': [],
				'next_observations': [],
				'terminals': []
			}
		if save_results is not None:
			f = open(save_results+"results.txt", "w")
		for t in range(n_test_episodes):
			print("Start >> ", "- Test", t)
			if save_results is not None:
				f.write("Start >>  Test "+str(t)+'\n')
			
			obs = self.model.get_env().reset()
			if render:
				self.model.get_env().render()
			rewards = []
			times = []
			done = False
			id = 0
			dataset['observations'].append(obs[0])

			while not done:
				start_time = time.time()
				action, _states = self.predict(obs, deterministic = True)
				dataset['actions'].append(action[0])
				obs, reward, done, info = self.model.get_env().step(action)
				dataset['next_observations'].append(obs[0])
				dataset['terminals'].append(done[0])
				if not done[0]:
					dataset['observations'].append(obs[0])
				print("Test", t, "- Step ", id ,"- Took action: ", action[0], "- Got reward: ", reward[0], "- Info: ", info)
				if save_results is not None:
					f.write("Test "+str(t)+"- Step "+str(id)+"- Took action: "+str(action[0])+"- Got reward: "+str(reward[0])+"- Info: "+str(info)+'\n')
				if render:
					self.model.get_env().render()
				rewards.append(reward)
				id+=1
				step_time = time.time() - start_time
				# print("--- %s seconds ---" % (step_time))
				times.append(step_time)
			print("Done >> Test", t, "- Reward = ", rewards, "- Sum reward: ", sum(rewards), "- Final episode info: ", info)
			print("--- Tot per episode: %s seconds ---" % (sum(times)))
			print("--- Avg per steps: %s seconds ---" % (sum(times)/len(times)))
			if save_results is not None:
					f.write("Done >> Test "+str(t)+"- Reward = "+str(rewards)+"- Sum reward: "+str(sum(rewards))+"- Final episode info: "+str(info)+'\n')
			r+= sum(rewards)
			cumulative_rewards.append(sum(rewards)[0])
			final_r+= reward
			infos.append(info[0])
			if 'distance' in info[0]:
				distances.append(info[0]['distance'])

		if save_dataset:
			for k in dataset.keys():
				dataset[k] = np.array(dataset[k])
			np.save(save_path_dataset_env+"/"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+"_"+str(n_test_episodes)+"episodes.npy", dataset)

		mean_infos = {}
		std_infos = {}
		for k in infos[0].keys():
			if k not in ['episode', 'terminal_observation']:
				infos_k = [d[k] for d in infos]
				mean = sum(infos_k) / len(infos)
				std = np.std(infos_k)
				mean_infos[k] = mean
				std_infos[k] = std

		print("[INFO]  >> Mean reward: ", r/n_test_episodes, " - Mean final reward: ", final_r/n_test_episodes)
		print("[INFO]  >> Avg final infos: ", mean_infos, " - Std final infos: ", std_infos)
		print("[INFO]  >> Avg cumulative_reward: ", sum(cumulative_rewards)/n_test_episodes, " - Std cumulative_reward: ", np.std(cumulative_rewards))

		if save_results is not None:
			f.write("[INFO]  >> Avg final infos: "+str(mean_infos)+" - Std final infos: "+str(std_infos))
			f.write("\n[INFO]  >> Avg cumulative_reward: "+str(sum(cumulative_rewards)/n_test_episodes)+" - Std cumulative_reward: "+str(np.std(cumulative_rewards))+'\n')
			np.save(save_results+str(seed)+"_cum_reward_per_episode.npy", cumulative_rewards)
			if 'distance' in info[0]:
				np.save(save_results+str(seed)+"_final_dist_per_episode.npy", distances)
	def predict(self, state, deterministic=False):
		return self.model.predict(state, deterministic=deterministic)

	def state_dict(self):
		return self.model.policy.state_dict()

	def save_state_dict(self, pathname):
		torch.save(self.state_dict(), pathname)

	def load_state_dict(self, path_or_state_dict):
		if type(path_or_state_dict) is str:
			self.model.policy.load_state_dict(torch.load(path_or_state_dict, map_location=torch.device(self.device)), strict=True)
		else:
			self.model.policy.load_state_dict(path_or_state_dict, strict=True)

	def save_full_state(self, pathname):
		self.model.save(pathname)

	def load_full_state(self, pathname):
		raise ValueError('Use the constructor with load_from_pathname parameter')
		pass