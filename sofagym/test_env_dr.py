# -*- coding: utf-8 -*-
"""Test the ...Env.

Usage:
-----
    python3.7 test_env_dr.py
Developed as an extension of the original work https://github.com/SofaDefrost/SofaGym,
by adding Domain Randomization techinques.
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"

import sys
import os
import time
import gym
import argparse
import pdb

from sofagym import *

RANDOM = False

import psutil
pid = os.getpid()
py = psutil.Process(pid)

import random as rd

sys.path.insert(0, os.getcwd()+"/..")

__import__('sofagym')

# env_dict = {
#     0: 'multigaitrobot-v0',
#     1: 'gripper-v0',
#     2: 'trunk-v0',
#     3: 'trunkcup-v0',
#     4: 'diamondrobot-v0',
#     5: 'maze-v0',
#     6: 'simple_maze-v0',
#     7: 'concentrictuberobot-v0',
#     8: 'trunkcube-v0',
#     9: 'multigaitrobotnotred-v0',
#     10: 'trunkwall-v0'
# }

env_dict = {
    0: 'multigaitrobot-v0',
    1: 'trunk-v0',
    2: 'trunkcube-v0'
}

start_multi_discrete =  [2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,]*4

parser = argparse.ArgumentParser()
parser.add_argument("-ne", "--num_env", help = "Number of the env",
                    type=int, required = True)
args = parser.parse_args()

env_name = env_dict[args.num_env]
print("Start env ", env_name)

env = gym.make(env_name)
min_space, max_space = env.get_search_bounds_all()
print("min search space: ", min_space)
print("max search space: ", max_space)
pdb.set_trace()
env.configure({"render":2})
env.configure({"dt":0.01})

# Testing on 8: 'trunkcube-v0'
# "dynamic_params":        ["cubeMass", "frictionCoeff", "trunkMass", "trunkPoissonRatio", "trunkYoungModulus"]
# "dynamic_params_values": [ 0.05,       0.3,             0.42,        0.45,                4500]
env.set_dr_distribution(dr_type='uniform', distr=[0.01, 0.1,   # Randomize cubeMass uniformly
                                                  0.1, 0.5,    # Randomize frictionCoeff uniformly
                                                  0.1, 0.5,    # Randomize trunkMass uniformly
                                                  0.4, 0.5,    # Randomize trunkPoissonRatio uniformly
                                                  2000, 7000]) # Randomize trunkYoungModulus uniformly
print("get_dr_distribution: ", env.get_dr_distribution())
env.set_dr_training(True)

print("Start ...")
num_tests = 3
for i in range(num_tests):

    print("Actual task: ", env.get_task())
    env.set_task(0.06, 0.9, 0.43, 0.46, 4500)
    print("Actual task: ", env.get_task())
    pdb.set_trace()
    env.reset()
    print("Actual task: ", env.get_task())
    pdb.set_trace()
    print("get_search_bounds for 0: ", env.get_search_bounds(0))
    print("get_task_lower_bound for 0: ", env.get_task_lower_bound(0))
    print("get_task_upper_bound for 0: ", env.get_task_upper_bound(0))
    pdb.set_trace()
    env.render()



    print("\n--------------------------------")
    print("TEST - ", i)
    print("--------------------------------\n")
    idx = 0
    tot_reward = 0
    tot_rtf = 0
    done = False
    while not done and idx < 50:
        idx += 1

        # ACTION TYPE
        multigaitrobot =  start_multi_discrete[idx-1]
        # gripper = rd.randint(0,3) #gripper = rd.randint(0,7)
        trunk = rd.randint(0,15)
        # trunkcup = rd.randint(0,15)
        # diamondrobot = rd.randint(0,7)
        # maze = rd.randint(0,6)
        # simple_maze = rd.randint(0,3)
        # concentrictuberobot = rd.randint(0,11)
        trunkcube = rd.randint(0,15)
        # multigaitrobotnotred =  start_multi_discrete[idx-1]

        # action_type = [multigaitrobot, gripper, trunk, trunkcup, diamondrobot, maze, simple_maze, concentrictuberobot, trunkcube, multigaitrobotnotred]
        action_type = [multigaitrobot, trunk, trunkcube]

        #action = strat_multi[idx-1] # it gives a continous action
        #action = 0 # constant action
        #action = rd.randint(0,15) # discrete action
        action = action_type[args.num_env]
        start_time = time.time()
        state, reward, done, _ = env.step(action)
        step_time = time.time()-start_time
        print("[INFO]   >>> Time:", step_time)
        rtf = env.config["dt"]*env.config["scale_factor"]/step_time
        print("[INFO]   >>> RTF:", rtf)
        tot_reward+= reward
        tot_rtf+= rtf
        env.render()

        print("Step ", idx, " action : ", action, " reward : ", reward, " done:", done)

    print("[INFO]   >>> TOTAL REWARD IS:", tot_reward)
    print("[INFO]   >>> FINAL REWARD IS:", reward)
    print("[INFO]   >>> MEAN RTF IS:", tot_rtf/idx)
    memoryUse = py.memory_info()[0]/2.**30
    print("[INFO]   >>> Memory usage:", memoryUse)
    print("[INFO]   >>> Object size:", sys.getsizeof(env))

# ... policy = ppo.train(source_env)
env.set_dr_training(False)
env.reset()


print(">> TOTAL REWARD IS:", tot_reward)
env.close()
print("... End.")
