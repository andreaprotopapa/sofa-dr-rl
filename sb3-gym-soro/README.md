# SB3 interface for gym environments
For more info refer to the official SB3 documentation at https://stable-baselines3.readthedocs.io/

# Notes
- total_timesteps parameter in sb3 is GLOBAL, i.e. it includes all state-transitions across all (potentially parallel) environments. E.g. -t 100 --now 4, means step 4 parallel environmets together 25 times.
- save_freq in sb3 is NOT global instead, and refers to the number of `env.step()` calls. Therefore, for --now parallel environments, a single env.step() call actually corresponds to --now state transitions. If you want to save_freq your environment every X global state-transitions, just transform save_freq to `max(save_freq // --now, 1)`
- on_policy algorithms in sb3 actually proceed with [batch-updates](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html) (intuitively), so:
	- num_timesteps is global current number of state transitions across any environment (potentially parallel)
	- n_steps (default 2048 in PPO) is the number of env.step() calls before a batch update on the policy network is performed
	- Therefore, a full epoch update is done every 2048\*num_envs num_timesteps, proceding by mini-batches of batch size samples.  For this reason, 2048\*num_envs is the minimum number of timesteps the environment will perform, even if total_timesteps is < 2048\*num_envs.
	```
	while t < max_t
		collect(2048*num_envs)
		do N_epochs on buffer (with mini-batches)
	```
	- The total number of state transitions during training will always be a multiple of "2048\*num_envs".
- With args.resume you can resume a run if your process has been crashed, starting loggin from the last step and loading the last checkpoint
    - For more info refer to https://docs.wandb.ai/guides/track/advanced/resuming
	- The argument `timesteps` is referred to the remaining number of timesteps to do