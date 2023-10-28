# -*- coding: utf-8 -*-
"""Specific environment for the multigait, tasked with walking forward.
Developed as an extension of the original work https://github.com/SofaDefrost/SofaGym,
by adding Domain Randomization techinques.
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"

from sofagym.env.common.AbstractEnv import AbstractEnv
from sofagym.env.common.rpc_server import start_scene
from gym.envs.registration import register

from gym import spaces
import os
import numpy as np

import json

class MultiGaitRobotEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "MultiGaitRobot",
                      "deterministic": True,
                      "goalList": [[200, 0, 0]], #[250, 0, 0], [-100, 0, 0]
                      "source": [150.0, -300, 150],
                      "target": [150, 0, 0],
                      "start_node": None,
                      "scale_factor": 60,
                      "timer_limit": 6,
                      "timeout": 200,
                      "dt": 0.05,
                      "display_size": (1400, 500),
                      "render": 2,
                      "save_data": False,
                      "save_video": True,
                      "save_image": False,
                      "save_path": path + "/Results" + "/MultiGaitRobot",
                      "planning": True,
                      "discrete": True,
                      "seed": None,
                      "start_from_history": None,
                      "python_version": "python3.8"
                      }

    def __init__(self, config=None):
        if config is None:
            config_path = os.path.dirname(os.path.abspath(__file__)) + "/MultiGaitRobot/MultiGaitRobot_random_config.json"
            with open(config_path) as config_random:
                config = json.load(config_random)

        self.task_dim = len(config["dynamic_params"])


        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dynamics_indexes = {}
        for i in range(self.task_dim):
            self.dynamics_indexes[i] = config["dynamic_params"][i]

        super().__init__(config)

        if self.config['discrete']:
            # discrete
            nb_actions = 6
            self.action_space = spaces.Discrete(nb_actions)
            self.nb_actions = str(nb_actions)
        else:
            # Continuous
            nb_actions = -1
            low_coordinates = np.array([-1]*3)
            high_coordinates = np.array([1]*3)
            self.action_space = spaces.Box(low_coordinates, high_coordinates,
                                           dtype='float32')
        self.nb_actions = str(nb_actions)

        # dim_state = 32
        dim_state = 8
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

        self.sampling = None
        self.dr_training = False
        self.preferred_lr = None
        self.reward_threshold = None
        self.endless = False

# DR methods ------------------

    def get_search_bounds(self, index):
        """Get search bounds for a specific parameter optimized.
        """
        search_bounds = {}
        for i in range(self.task_dim):
            search_bounds[i] = (self.config[self.dynamics_indexes[i]+"_min_search"], self.config[self.dynamics_indexes[i]+"_max_search"])
        return search_bounds[index]
    
    def get_search_bounds_all(self):
        """Get search bounds for all the parameters optimized.
        """
        min_search = []
        max_search = []
        for i in range(self.task_dim):
            min_search.append(self.config[self.dynamics_indexes[i]+"_min_search"])
            max_search.append(self.config[self.dynamics_indexes[i]+"_max_search"])
        return min_search, max_search

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics
        Used for resampling unfeasible values during domain randomization
        """

        lowest_value = {}
        for i in range(self.task_dim):
            lowest_value[i] = self.config[self.dynamics_indexes[i]+"_lowest"]
        return lowest_value[index]

    def get_task_upper_bound(self, index):
        """Returns highest feasible value for each dynamics
        Used for resampling unfeasible values during domain randomization
        """

        highest_value = {}
        for i in range(self.task_dim):
            highest_value[i] = self.config[self.dynamics_indexes[i]+"_highest"]
        return highest_value[index]

    def get_task(self):
        dynamic_params_values = np.array( self.config["dynamic_params_values"])
        return dynamic_params_values

    def set_task(self, *task):
        self.config["dynamic_params_values"] = task

# ------------------

    def step(self, action):
        return super().step(action)

    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()

        self.config.update({'goalPos': self.goal})
        # obs = super().reset()
        # return np.array(obs)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        obs = start_scene(self.config, self.nb_actions)

        return np.array(obs['observation'])

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
        if createVideo is None and self.config["save_video"]:
            createVideo = self.config['save_path_video']+"/"
        # Use the viewer to display the environment.
        super().render(mode, createVideo)

    def get_available_actions(self):
        """Gives the actions available in the environment.

        Parameters:
        ----------
            None.

        Returns:
        -------
            list of the action available in the environment.
        """
        return list(range(int(self.nb_actions)))


register(
    id='multigaitrobot-v0',
    entry_point='sofagym.env:MultiGaitRobotEnv',
)
