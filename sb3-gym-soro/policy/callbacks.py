from stable_baselines3.common.callbacks import BaseCallback
import pdb
import wandb
import os
import numpy as np

class WandbRecorderCallback(BaseCallback):
    """
    A custom callback that allows to print stuff on wandb after every evaluation

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_freq=None, wandb_loss_suffix="", verbose=0):
        super(WandbRecorderCallback, self).__init__(verbose)

        self.child_eval_freq = eval_freq
        self.n_eval_calls = 0
        self.wandb_loss_suffix = wandb_loss_suffix


    def _on_step(self) -> bool:
        """
        This method is called as a child callback of the `EventCallback`),
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.

        Print stuff on wandb
        """
        self.n_eval_calls += 1
        last_mean_reward = self.parent.last_mean_reward
        # current_timestep = self.parent.n_calls
        # current_timestep = self.n_eval_calls*self.child_eval_freq
        current_timestep = self.num_timesteps # this number is multiplied by the number of parallel envs
        wandb.log({"train_mean_reward"+self.wandb_loss_suffix: last_mean_reward, "timestep": current_timestep}, commit=False)
        for info, value in self.parent.locals.get('infos')[0].items():
            wandb.log({info: value}, commit=False)
        wandb.log(data={}, commit=True)
        return True

class CheckpointLastCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()`` overwriting the previous one if exists.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save a checkpoint every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Prefix to the saved model
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        keep_prev: bool = False,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.keep_prev = keep_prev

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if not self.keep_prev:
                for f in os.listdir(self.save_path): # Remove previous ckpts
                    os.remove(os.path.join(self.save_path, f))
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")
            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


"""
    Template for custom callback
"""
# class CustomCallback(BaseCallback):
#     """
#     A custom callback that derives from ``BaseCallback``.

#     :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
#     """
#     def __init__(self, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         # Those variables will be accessible in the callback
#         # (they are defined in the base class)
#         # The RL model
#         # self.model = None  # type: BaseAlgorithm
#         # An alias for self.model.get_env(), the environment used for training
#         # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
#         # Number of time the callback was called
#         # self.n_calls = 0  # type: int
#         # self.num_timesteps = 0  # type: int
#         # local and global variables
#         # self.locals = None  # type: Dict[str, Any]
#         # self.globals = None  # type: Dict[str, Any]
#         # The logger object, used to report things in the terminal
#         # self.logger = None  # stable_baselines3.common.logger
#         # # Sometimes, for event callback, it is useful
#         # # to have access to the parent object
#         # self.parent = None  # type: Optional[BaseCallback]

#     def _on_training_start(self) -> None:
#         """
#         This method is called before the first rollout starts.
#         """
#         pass

#     def _on_rollout_start(self) -> None:
#         """
#         A rollout is the collection of environment interaction
#         using the current policy.
#         This event is triggered before collecting new samples.

#         This method is not actually called after every single .reset()
#         """

#         # env = self.model.get_env()

#         # pdb.set_trace()
#         # env.step_counter()
#         # self.training_env.env_method('step_counter')

#         pass

#     def _on_step(self, *args, **kwargs) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.

#         ---> For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.

#         :return: (bool) If the callback returns False, training is aborted early.
#         """



#         print('Last mean reward:', self.parent.last_mean_reward)
#         last_mean_reward = self.parent.last_mean_reward

#         return True

#     def _on_rollout_end(self) -> None:
#         """
#         This event is triggered before updating the policy.
#         """
#         pass

#     def _on_training_end(self) -> None:
#         """
#         This event is triggered before exiting the `learn()` method.
#         """
#         pass