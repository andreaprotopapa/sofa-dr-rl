# -*- coding: utf-8 -*-
"""Test the ...Env.

Usage:
-----
    python3.7 test_env.py
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

parser = argparse.ArgumentParser()
parser.add_argument("-ne", "--num_env", help = "Number of the env",
                    type=int, required = True)
args = parser.parse_args()

env_name = env_dict[args.num_env]
print("Start env ", env_name)

env = gym.make(env_name)
env.configure({"render":2})
env.configure({"dt":0.01})
env.reset()

env.render()
done = False

strat_multi = [[-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
                            [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
                            [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
                            [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
                            [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0,-1.0, -1.0]]*100
# strat_multi = [[1.0, 1.0, 1.0, 1, 1]]*100


# strat_multi =  [[-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
#             [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
#             [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
#             [-1.0, 1, -1.0, -1.0, 1], [-1.0, -1, 1.0, 1.0, -1],
#             [1.0, 1, -1.0, -1.0, 1], [-1.0, -1.0, 1.0, -1.0, 1],
#             [1.0, -1.0, 1.0, -1.0, -1.0], [-1.0, -1, 1.0, 1.0, -1],
#             [1.0, 1.0, -1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 1.0, -1],
#             [1.0, -1, 1.0, 1.0, -1]]

start_multi_discrete =  [2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,]*4

strat_jimmy_1 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-0.6, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.6, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0]]+[[0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0]]*100

strat_jimmy_0 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [-0.8, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]] + [[-0.8, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]]*100


print("Start ...")
num_episodes = 3
for i in range(num_episodes):
    print("\n--------------------------------")
    print("EPISODE - ", i)
    print("--------------------------------\n")
    idx = 0
    tot_reward = 0
    tot_rtf = 0
    done = False
    while not done and idx < 100:
        idx += 1

        # ACTION TYPE
        multigaitrobot =  start_multi_discrete[idx-1]#[rd.uniform(-1, 1) for i in range(5)] # OR strat_multi[idx-1] OR start_multi_discrete
        # gripper = rd.randint(0,3) #gripper = rd.randint(0,7)
        trunk = rd.randint(0,15)
        # trunkcup = rd.randint(0,15)
        # diamondrobot = rd.randint(0,7)
        # maze = rd.randint(0,6)
        # simple_maze = rd.randint(0,3)
        # concentrictuberobot = rd.randint(0,11)
        trunkcube = rd.randint(0,15)
        # multigaitrobotnotred =  start_multi_discrete[idx-1]
        trunkwall = rd.randint(0,15)

        # action_type = [multigaitrobot, gripper, trunk, trunkcup, diamondrobot, maze, simple_maze, concentrictuberobot, trunkcube, multigaitrobotnotred, trunkwall]
        action_type = [multigaitrobot, trunk, trunkcube, trunkwall]

        #action = strat_multi[idx-1] # it gives a continous action
        #action = 0 # constant action
        #action = rd.randint(0,15) # discrete action
        action = action_type[args.num_env]
        start_time = time.time()
        state, reward, done, info = env.step(action)
        step_time = time.time()-start_time
        print("[INFO]   >>> Time:", step_time)
        rtf = env.config["dt"]*env.config["scale_factor"]/step_time
        print("[INFO]   >>> RTF:", rtf)
        tot_reward+= reward
        tot_rtf+= rtf
        env.render()

        print("Step ", idx, " action : ", action, " reward : ", reward, " done:", done, "- info: ", info)

    print("[INFO]   >>> TOTAL REWARD IS:", tot_reward)
    print("[INFO]   >>> FINAL REWARD IS:", reward)
    print("[INFO]   >>> MEAN RTF IS:", tot_rtf/idx)
    memoryUse = py.memory_info()[0]/2.**30
    print("[INFO]   >>> Memory usage:", memoryUse)
    print("[INFO]   >>> Object size:", sys.getsizeof(env))

    env.reset()


print(">> TOTAL REWARD IS:", tot_reward)
env.close()
print("... End.")
