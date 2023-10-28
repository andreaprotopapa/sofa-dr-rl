# -*- coding: utf-8 -*-
"""AbstractEnv to make the link between Gym and Sofa.
Developed as an extension of the original work https://github.com/SofaDefrost/SofaGym,
by adding Domain Randomization techinques.
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"

import gym
from gym.utils import seeding
from gym import spaces

import numpy as np
import copy
import os
import sys
from scipy.stats import truncnorm
import pdb

#sys.path.append('/home/aprotopapa/code/MySofa/sofa_rl/stlib3')
import splib
from splib import animation

from sofagym.env.common.viewer import Viewer
from sofagym.env.common.rpc_server import start_server, add_new_step, get_result, clean_registry, close_scene

class AbstractEnv(gym.Env):
    """Use Sofa scene with a Gym interface.

    Methods:
    -------
        __init__: classical __init__ method.
        initialization: Initialization of all arguments.
        seed: Initialization of the seed.
        step: Realise a step in the environment.
        async_step: Realise a step without blocking queue.
        reset: Reset the environment and useful arguments.
        render: Use viewer to see the environment.
        _automatic_rendering: Automatically render the intermediate frames
            while an action is still ongoing.
        close: Terminate the simulation.
        configure: Add element in the configuration.
        clean: clean the registery.
        _formataction.. : transforme the type of action to use server.

    Arguments:
    ---------
        config: Dictionary.
            Contains the configuration of the environment.
            Minimum:
                - scene : the name of the simulation.
                    Note: define the name of the toolbox <scene>Toolbox and the
                    scene <scene>Scene in the directory ../<scene>.
                - deterministic: whether or not the environment is deterministic.
                - source,target: definition of the Sofa camera point of view.
                - goalList : list of the goals to reach (position or index).
                - start_node: the start node (position or index).
                - scale_factor: int that define the number of step in simulation.
                - timer_limit: int that define the maximum number of steps.
                - timeout: int that define the timeout for the server/client requests.
                - display_size: tuple of int that define the size of the Viewer
                    window.
                - save_path: path to save the image of the simulation.
                - render: wheter or not the viewer displays images.
                    0: no rendering.
                    1: render after simulation.
                    2: render all steps.
                    Warning: we can't change this value after initialization.
                - save_data: wheter or not the data are saved.
                - save_image: wheter or not the images are saved.
                - planning: if realise planning or not.
                - discrete: if the environment is discrete or not.
                - timer_limit: the limit of the time.
                - seed : the seed.
                - start_from_history: list of actions that have to be carried
                    out before starting the training.
                - python_version: the version of python.
                - time_before_start: initialize the simulation with time_before_start steps.
        observation_space: spaces.Box
            Define the size of the environment.
        past_actions: list of int.
            Keeps track of past actions. Allows you to retrieve past
            configurations of the environment.
        goalList: list
            List of possible objectives to be achieved.
        goal: list
            Current objective.
        num_envs: int
            The number of environment.
        np_random:  np.random.RandomState()
             Exposes a number of methods for generating random numbers
        viewer: <class viewer>
            Allows to manage the visual feedback of the simulation.
        automatic_rendering_callback:
            Callback function used in _automatic_rendering.
        timer:
            Number of steps already completed.
        deterministic:
            Whether the environment is deterministic or not.
        timeout:
            Number of times the queue is blocking. Allows to avoid blocking
            situations.

    Notes:
    -----
        It is necessary to define the specificity of the environment in a
        subclass.

    Usage:
    -----
        Use the reset method before launch the environment.


    """
    def __init__(self, config=None):
        """
        Classic initialization of a class in python.

        Parameters:
        ----------
        config: Dictionary or None, default = None
            Customisable configuration element.

        Returns:
        ---------
            None.

        """

        # Define a DEFAULT_CONFIG in sub-class.
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)
        self.goal_idx = 0
        self.initialization()

    def initialization(self):
        """Initialization of all parameters.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.
        """

        self.goalList = None
        self.goal = None
        
        self.past_actions = []


        self.num_envs = 40

        self.np_random = None

        self.seed(self.config['seed'])

        self.viewer = None
        self.automatic_rendering_callback = None

        self.timer = 0
        self.timeout = self.config["timeout"]

        # Start the server which distributes the calculations to its clients
        start_server(self.config)

        if 'save_data' in self.config and self.config['save_data']:
            save_path_results = self.config['save_path']+"/data"
            os.makedirs(save_path_results, exist_ok=True)
        else:
            save_path_results = None

        if 'save_image' in self.config and self.config['save_image']:
            save_path_image = self.config['save_path']+"/img"
            os.makedirs(save_path_image, exist_ok=True)
        else:
            save_path_image = None

        if 'save_video' in self.config and self.config['save_video']:
            save_path_video = self.config['save_path']+"/video"
            os.makedirs(save_path_video, exist_ok=True)
        else:
            save_path_video = None

        self.configure({"save_path_image": save_path_image, "save_path_results": save_path_results, "save_path_video": save_path_video})

    # methods to override:
    # ----------------------------
    def get_search_bounds(self, index):
        """Get search space for current randomized parameter at index <i>
        """
        raise NotImplementedError

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for current randomized parameter at index <i>
        """
        raise NotImplementedError
    
    def get_task_upper_bound(self, index):
        """Returns lowest feasible value for current randomized parameter at index <i>
        """
        raise NotImplementedError

    def get_task(self):
        """Get current dynamics parameters"""
        raise NotImplementedError

    def set_task(self, *task):
        """Set dynamics parameters to <task>"""
        raise NotImplementedError

    # -----------------------------

    # DR methods ------------------
    def set_random_task(self):
        self.set_task(*self.sample_task())

    def set_dr_training(self, flag):
        """
            If True, new dynamics parameters
            are sampled and set during .reset()
        """
        self.dr_training = flag

    def get_dr_training(self):
        return self.dr_training

    def set_endless(self, flag):
        """
            If True, episodes are
            never reset (done always False)
        """
        self.endless = flag

    def get_endless(self):
        return self.endless

    def get_reward_threshold(self):
        return self.reward_threshold

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def set_dr_distribution(self, dr_type, distr):
        if dr_type == 'uniform':
            self.set_udr_distribution(distr)
        elif dr_type == 'truncnorm':
            self.set_truncnorm_distribution(distr)
        elif dr_type == 'gaussian':
            self.set_gaussian_distribution(distr)
        elif dr_type == 'fullgaussian':
            self.set_fullgaussian_distribution(distr['mean'], distr['cov'])
        else:
            raise Exception('Unknown dr_type:'+str(dr_type))

    def get_dr_distribution(self):
        if self.sampling == 'uniform':
            return self.min_task, self.max_task
        elif self.sampling == 'truncnorm':
            return self.mean_task, self.stdev_task
        elif self.sampling == 'gaussian':
            raise ValueError('Not implemented')
        else:
            return None

    def set_udr_distribution(self, bounds):
        self.sampling = 'uniform'
        for i in range(len(bounds)//2):
            self.min_task[i] = bounds[i*2]
            self.max_task[i] = bounds[i*2 + 1]
        return

    def set_truncnorm_distribution(self, bounds):
        self.sampling = 'truncnorm'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def set_gaussian_distribution(self, bounds):
        self.sampling = 'gaussian'
        for i in range(len(bounds)//2):
            self.mean_task[i] = bounds[i*2]
            self.stdev_task[i] = bounds[i*2 + 1]
        return

    def set_fullgaussian_distribution(self, mean, cov):
        self.sampling = 'fullgaussian'
        self.mean_task[:] = mean
        self.cov_task = np.copy(cov)
        return

    def set_task_search_bounds(self):
        """Sets the task search bounds based on how they are specified in get_search_bounds"""
        dim_task = len(self.get_task())
        for i in range(dim_task):
            b = self.get_search_bounds(i)
            self.min_task[i], self.max_task[i] = b[0], b[1]

    def get_task_search_bounds(self):
        dim_task = len(self.get_task())
        min_task = np.empty(dim_task)
        max_task = np.empty(dim_task)
        for i in range(dim_task):
            b = self.get_search_bounds(i)
            min_task[i], max_task[i] = b[0], b[1]
        return min_task, max_task

    def sample_task(self):
        if self.sampling == 'uniform':
            return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

        elif self.sampling == 'truncnorm':
            a,b = -2, 2
            sample = []

            for i, (mean, std) in enumerate(zip(self.mean_task, self.stdev_task)):
                obs = truncnorm.rvs(a, b, loc=mean, scale=std)
                
                lower_bound = self.get_task_lower_bound(i)
                upper_bound = self.get_task_upper_bound(i)

                attempts = 0
                while obs < lower_bound or obs > upper_bound:
                    obs = truncnorm.rvs(a, b, loc=mean, scale=std)

                    attempts += 1
                    if attempts > 2:
                        obs = (lower_bound+upper_bound)/2

                sample.append( obs )

            return np.array(sample)

        elif self.sampling == 'gaussian':
            sample = []

            for mean, std in zip(self.mean_task, self.stdev_task):

                # Assuming all parameters > 0.001
                
                obs = np.random.randn()*std + mean
                lower_bound = self.get_task_lower_bound(i)
                upper_bound = self.get_task_upper_bound(i)
                attempts = 0
                while obs < lower_bound:
                    obs = np.random.randn()*std + mean

                    attempts += 1
                    if attempts > 2:
                        raise Exception('Not all samples were above > lower_bound after 2 attempts')
                attempts = 0
                while obs > upper_bound:
                    obs = np.random.randn()*std + mean

                    attempts += 1
                    if attempts > 2:
                        raise Exception('Not all samples were above < upper_bound after 2 attempts')

                sample.append( obs )

            return np.array(sample)

        elif self.sampling == 'fullgaussian':
            # Assumes that mean_task and cov_task are work in a normalized space [0, 4]
            sample = np.random.multivariate_normal(self.mean_task, self.cov_task)
            sample = np.clip(sample, 0, 4)

            sample = self.denormalize_parameters(sample)
            return sample

        else:
            raise ValueError('sampling value of random env needs to be set before using sample_task() or set_random_task(). Set it by uploading a DR distr from file.')

        return

    def denormalize_parameters(self, parameters):
        """Denormalize parameters back to their original space
        
            Parameters are assumed to be normalized in
            a space of [0, 4]
        """
        assert parameters.shape[0] == self.task_dim

        min_task, max_task = self.get_task_search_bounds()
        parameter_bounds = np.empty((self.task_dim, 2), float)
        parameter_bounds[:,0] = min_task
        parameter_bounds[:,1] = max_task

        orig_parameters = (parameters * (parameter_bounds[:,1]-parameter_bounds[:,0]))/4 + parameter_bounds[:,0]

        return np.array(orig_parameters)

    # def load_dr_distribution_from_file(self, filename):
    #     dr_type = None
    #     bounds = None

    #     with open(filename, 'r', encoding='utf-8') as file:
    #         reader = csv.reader(file, delimiter=',')
    #         dr_type = str(next(reader)[0])
    #         bounds = []

    #         second_row = next(reader)
    #         for col in second_row:
    #             bounds.append(float(col))

    #     if dr_type is None or bounds is None:
    #         raise Exception('Unable to read file:'+str(filename))

    #     if len(bounds) != self.task_dim*2:
    #         raise Exception('The file did not contain the right number of column values')

    #     if dr_type == 'uniform':
    #         self.set_udr_distribution(bounds)
    #     elif dr_type == 'truncnorm':
    #         self.set_truncnorm_distribution(bounds)
    #     elif dr_type == 'gaussian':
    #         self.set_gaussian_distribution(bounds)
    #     else:
    #         raise Exception('Filename is wrongly formatted: '+str(filename))

    #     return
    # -----------------------------

    def seed(self, seed=None):
        """
        Computes the random generators of the environment.

        Parameters:
        ----------
        seed: int, 1D array or None, default = None
            seed for the RandomState.

        Returns:
        ---------
            [seed]

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _formataction(self, action):
        """Change the type of action to be in [list, float, int].

        Parameters:
        ----------
            action:
                The action with no control on the type.

        Returns:
        -------
            action: in [list, float, int]
                The action with  control on the type.
        """
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, np.int64):
            action = int(action)
        elif isinstance(action, np.float64):
            action = float(action)
        elif isinstance(action, tuple):
             action = self._formatactionTuple(action)
        elif isinstance(action, dict):
            action = self._formatactionDict(action)
        return action

    def _formatactionTuple(self, action):
        """Change the type of tuple action to be in [list, float, int].

        Parameters:
        ----------
            action:
                The action with no control on the type.

        Returns:
        -------
            action:
                The action with  control on the type.
        """
        return self._formataction(action[0]), self._formataction(action[1])

    def _formatactionDict(self, action):
        """Change the type of tuple action to be in [list, float, int].

        Parameters:
        ----------
            action:
                The action with no control on the type.

        Returns:
        -------
            action:
                The action with  control on the type.
        """
        for key in action.keys():
            action[key] = self._formataction(action[key])

        return action

    def clean(self):
        """Function to clean the registery .

        Close clients who are processing unused sequences of actions (for
        planning)

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.
        """

        clean_registry(self.past_actions)

    def step(self, action):
        """Executes one action in the environment.

        Apply action and execute scale_factor simulation steps of 0.01 s.

        Parameters:
        ----------
            action: int
                Action applied in the environment.

        Returns:
        -------
            obs:
                The new state of the agent.
            reward:
                The reward obtain after applying the action in the current state.
            done:
                Whether the goal is reached or not.
            {}: additional information (not used here)
        """

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action = self._formataction(action)

        # Pass the actions to the server to launch the simulation.
        result_id = add_new_step(self.past_actions, action)
        self.past_actions.append(action)

        # Request results from the server.
        # print("[INFO]   >>> Result id:", result_id)
        results = get_result(result_id, timeout=self.timeout)
        obs = np.array(results["observation"])  # to work with baseline
        reward = results["reward"]
        done = results["done"]
        info = results["info"]

        # Avoid long explorations by using a timer.
        self.timer += 1
        if self.timer >= self.config["timer_limit"]:
            # reward = -150
            done = True

        if self.config["planning"]:
            self.clean()

        if self.endless:
            done = False

        return obs, reward, done, info

    def async_step(self, action):
        """Executes one action in the environment.

        Apply action and execute scale_factor simulation steps of 0.01 s.
        Like step but useful if you want to parallelise (blocking "get").
        Otherwise use step.

        Parameters:
        ----------
            action: int
                Action applied in the environment.

        Returns:
        -------
            LateResult:
                Class which allows to store the id of the client who performs
                the calculation and to return later the usual information
                (observation, reward, done) thanks to a get method.

        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        result_id = add_new_step(self.past_actions, action)
        self.past_actions.append(action)

        class LateResult:
            def __init__(self, result_id):
                self.result_id = result_id

            def get(self, timeout=None):
                results = get_result(self.result_id, timeout=timeout)
                obs = results["observation"]
                reward = results["reward"]
                done = results["done"]
                return obs, reward, done, {}

        return LateResult(copy.copy(result_id))

    def reset(self):
        """Reset simulation.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.

        """
        self.close()
        self.initialization()
        splib.animation.animate.manager = None
        if not self.goalList:
            self.goalList = self.config["goalList"]

        if isinstance(self.goalList, list):
            # Set a new random goal from the list
            id_goal = self.np_random.choice(range(len(self.goalList)))
            self.config.update({'goal_node': id_goal})
            self.goal = self.goalList[id_goal]
        elif self.goalList is None: # random goal position
            if self.config["test"]:
                self.goal = self.config["goalList_test"][self.goal_idx//2]
                self.goal_idx += 1
            else:
                self.goalList = spaces.Box(np.array(self.config["goal_low"]), np.array(self.config["goal_high"]), dtype=np.float32)
                self.goal = self.goalList.sample().tolist()
                self.goalList = None

        self.timer = 0
        self.past_actions = []

        return

    def render(self, mode='rgb_array', createVideo=None):
        """See the current state of the environment.

        Get the OpenGL Context to render an image (snapshot) of the simulation
        state.

        Parameters:
        ----------
            mode: string, default = 'rgb_array'
                Type of representation.
            createVideo: string, default = None
                Title name of the video

        Returns:
        -------
            None.
        """
        if self.config['render'] != 0:
            # Define the viewer at the first run of render.
            if not self.viewer:
                display_size = self.config["display_size"]  # Sim display
                if 'zFar' in self.config:
                    zFar = self.config['zFar']
                else:
                    zFar = 0
                self.viewer = Viewer(self, display_size, zFar=zFar, save_path=self.config["save_path_image"], create_video=createVideo)

            # Use the viewer to display the environment.
            self.viewer.render()
        else:
            print(">> No rendering")

    def _automatic_rendering(self):
        """Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps
        corresponding to agent decision-making.
        If a callback has been set, use it to perform the rendering. This is
        useful for the environment wrappers such as video-recording monitor that
        need to access these intermediate renderings.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.

        """
        if self.viewer is not None:
            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render()

    def close(self):
        """Terminate simulation.

        Close the viewer and the scene.

        Parametres:
        ----------
            None.

        Returns:
        -------
            None.
        """
        if self.viewer is not None:
            self.viewer.close()

        close_scene()
        print("All clients are closed. Bye Bye.")

    def configure(self, config):
        """Update the configuration.

        Parameters:
        ----------
            config: Dictionary.
                Elements to be added in the configuration.

        Returns:
        -------
            None.

        """
        self.config.update(config)
