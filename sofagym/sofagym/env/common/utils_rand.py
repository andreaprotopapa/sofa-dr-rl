"""utils for Domain Randomization techinques
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"

import numpy as np
import random
import pdb

def set_initial_state_distr(config):
    param_set = []
    for param_name in config["initial_state_distr"]:
        init = np.array(config[param_name+"_init"])

        if config[param_name+"_distr"] == "gaussian":
            init = np.array(config[param_name+"_init"])
            noise = np.random.normal(config[param_name+"_gaussMean"], config[param_name+"_gaussStd"], init.shape)
            for i in config[param_name+"_applyOn"]: # we modify only the defined coordinates
                init[i] += noise[i]
        elif config[param_name+"_distr"] == "uniform":
            for i in config[param_name+"_applyOn"]: # we modify only the defined coordinates
                init[i] += round(random.uniform(config[param_name+"_unifLow"][i],config[param_name+"_unifHigh"][i]), 2)

        for i, x in enumerate(init):
            if x < config[param_name+"_min_search"][i]:
                init[i] = config[param_name+"_min_search"][i]
            if x > config[param_name+"_max_search"][i]:
                init[i] = config[param_name+"_max_search"][i]

        if init.shape[0] > 1:
            param = list(init)
        else:
            param = init[0]

        print(f"Actual {param_name}: {param}")
        param_set.append(param)

    return param_set

def set_dynamic_params(config):
    param_set = config["dynamic_params_values"]
    for i, param_name in enumerate(config["dynamic_params"]):
        print(f"Actual {param_name}: {param_set[i]}")
    return param_set