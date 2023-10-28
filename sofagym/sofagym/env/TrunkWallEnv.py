# -*- coding: utf-8 -*-
"""Specific environment for the trunk, aiming to lift a flat object in the
presence of a nearby wall . 
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
import pdb

import json


class TrunkWallEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "TrunkWall",
                      "deterministic": True,

                      "source": [0, 100, 500], #[450, 0, 80],
                      "target": [0, 0, 0], #[0, 0, 80],
                      #"goalList": [[40, -60, 200], [-10, 20, 80]],
                      "goalList": [[0.0, 50.0, 200.0]],
                      #"goalList": None, 
                      "goal_low": [-50., -50], # for random goal pos: just two dimensions, the y-axis is fixed
                      "goal_high": [50., 50.], # for random goal pos: just two dimensions, the y-axis is fixed

                      "floorHeight": -100.0,
                      "cube_initial_translation": [0.0, -60.0, 100.0],
                      "cube_initial_rotation": [0.0, 0.0, 0.0],  # for random rotation: apply rotation on y-axis for the cube_initial_rotation
                      "cube_scale": 5,
                      "start_node": None,
                      "scale_factor": 5,
                      "timer_limit": 100,
                      "dt": 0.01,
                      "timeout": 50,
                      "display_size": (1400, 500),
                      "render": 2,
                      "save_data": False,
                      "save_video": True,
                      "save_image": False,
                      "save_path": path + "/Results" + "/TrunkWall",
                      "planning": False,
                      "discrete": True,
                      "seed": None,
                      "start_from_history": None,
                      "python_version": "python3.8"
                      }

    def __init__(self, config_ext_dict = None):
        config_path = os.path.dirname(os.path.abspath(__file__)) + "/TrunkWall/TrunkWall_random_config.json"
        with open(config_path) as config_random:
            config = json.load(config_random)
        if config_ext_dict is not None:
            config.update(config_ext_dict)

        self.task_dim = len(config["dynamic_params"])


        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dynamics_indexes = {}
        for i in range(self.task_dim):
            self.dynamics_indexes[i] = config["dynamic_params"][i]

        super().__init__(config)
        nb_actions = 16
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = 73
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

        print(f"list: {self.goalList}")
        if len(self.goal) == 2: # spaces.Box case
            self.goal = [self.goal[0], self.config["floorHeight"], self.goal[1]]
        self.config.update({'goalPos': self.goal})
        print(f"goal: {self.goal}")

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        obs = start_scene(self.config, self.nb_actions)

        return obs['observation']

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
    id='trunkwall-v0',
    entry_point='sofagym.env:TrunkWallEnv',
)
