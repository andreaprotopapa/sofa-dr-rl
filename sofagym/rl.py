# -*- coding: utf-8 -*-
"""Test the MultiGaitRobotEnv by learning a policy to move in the x direction.

Usage:
-----
    python3.7 rl_multigait.py
"""

__authors__ = "emenager, pschegg"
__contact__ = "etienne.menager@ens-rennes.fr, pierre.schegg@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020,Inria"
__date__ = "Nov 10 2020"


from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
# from AVEC.stable_baselines import PPO2
# from AVEC.stable_baselines.sac import SAC as SAC_AVEC

import gym

import sys
import os
import json
import pathlib
import numpy as np
import torch
import random
import argparse
import time
import pdb

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


# Adapted from rl-agents
def load_environment(id, rank, seed = 0):
    def _init():
        __import__('sofagym')
        env = gym.make(id)
        env.seed(seed + rank)
        env.reset()
        return env

    return _init



def test(env, model, epoch, n_test=1, render = False):
    if render:
        env.config.update({"render":2})
    r, final_r = 0, 0
    for t in range(n_test):
        print("Start >> Epoch", epoch, "- Test", t)
        obs = env.reset()
        if render:
            env.render()
        rewards = []
        done = False
        id = 0
        while not done:
            #print("a")
            action, _states = model.predict(obs, deterministic = False)
            #print("b")
            obs, reward, done, info = env.step(action)
            #print("c")
            if render:
                print("Test", t, "- Epoch ", id ,"- Took action: ", action, "- Got reward: ", reward)
                env.render()
                #print("d")
            rewards.append(reward)
            id+=1
        print("Done >> Test", t, "- Reward = ", rewards, "- Sum reward:", sum(rewards))
        r+= sum(rewards)
        final_r+= reward
    print("[INFO]  >> Mean reward: ", r/n_test, " - Mean final reward:", final_r/n_test)
    return r/n_test, final_r/n_test

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    return d

class Env:
    def __init__(self, id, name, timer_limit, continues, n_epochs, 
                       gamma, learning_rate, value_coeff, batch_size, size_layer):
        self.id = id
        self.name = name
        self.timer_limit = timer_limit
        self.continues = continues
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.value_coeff = value_coeff
        self.batch_size = batch_size
        self.size_layer = size_layer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--num_env", help = "Number of the env",
                        type=int, required = True)
    parser.add_argument("-na", "--num_algo", help = "Number of the algorithm",
                        type=int, required = True)
    parser.add_argument("-nc", "--num_cpu", help = "Number of cpu",
                        type=int)
    parser.add_argument("-s", "--seed", help = "The seed",
                        type=int, required = True)
    parser.add_argument("-train", "--training", help = "Training mode",
                        default=False, action="store_true")
    parser.add_argument("-test", "--testing", help = "Testing mode",
                        default=False, action="store_true")
    parser.add_argument("-restart", "--restart", help = "Restart training from id",
                        type=int, default=0)
    args = parser.parse_args()

    if args.training == args.testing: # both True or both False
        print("[ERROR] >> Pass only one between -train or -test")
        exit(0)

    env_dict = {
        0: Env(0, 'cartstemcontact-v0', timer_limit=30, continues=True, n_epochs=600, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=200, size_layer=[512, 512, 512]),
        1: Env(1, 'cartstem-v0', timer_limit=80, continues=False, n_epochs=200, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=256, size_layer=[512, 512, 512]),
        2: Env(2, 'stempendulum-v0', timer_limit=50, continues=True, n_epochs=10001, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=64, size_layer=[512, 512, 512]),
        3: Env(3, 'catchtheobject-v0', timer_limit=30, continues=True, n_epochs=200, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=256, size_layer=[512, 512, 512]),
        4: Env(4, 'multigaitrobot-v0', timer_limit=18, continues=False, n_epochs=600, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=144, size_layer=[512, 512, 512]),
        5: Env(5, 'trunk-v0', timer_limit=18, continues=False, n_epochs=600, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=144, size_layer=[512, 512, 512]),
        6: Env(6, 'gripper-v0', timer_limit=18, continues=False, n_epochs=600, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=144, size_layer=[512, 512, 512]),
        7: Env(7, 'trunkcube-v0', timer_limit=18, continues=False, n_epochs=600, gamma=0.99, learning_rate=1e-4,
                    value_coeff=0, batch_size=144, size_layer=[512, 512, 512]),
    }

    gamma = env_dict[args.num_env].gamma
    learning_rate = env_dict[args.num_env].learning_rate
    value_coeff = env_dict[args.num_env].value_coeff
    batch_size = env_dict[args.num_env].batch_size
    size_layer = env_dict[args.num_env].size_layer

    id = env_dict[args.num_env].name
    timer_limit = env_dict[args.num_env].timer_limit
    cont = env_dict[args.num_env].continues

    n_epochs = env_dict[args.num_env].n_epochs

    if args.num_algo == 0 and cont:
        env = load_environment(id, rank = 0, seed = args.seed*10)()
        test_env = env

        algo = 'SAC'
        policy_kwargs = dict(net_arch=dict(pi=size_layer, qf=size_layer))
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=gamma, learning_rate=learning_rate, batch_size = batch_size, ent_coef='auto', learning_starts=500)
    elif args.num_algo == 1:

        if args.num_cpu is None:
            print("[WARNING] >> Default number of cpu: 4.")
            n_cpu = 4
        else:
            n_cpu = args.num_cpu

        env = SubprocVecEnv([load_environment(id, i, seed = args.seed) for i in range(n_cpu)])
        test_env = load_environment(id, 0, seed = args.seed*10)()
        algo = 'PPO'

        policy_kwargs = dict(net_arch=[dict(pi=size_layer, vf=size_layer)])
        model = PPO("MlpPolicy", env, n_steps=timer_limit*20, batch_size=batch_size, gamma=gamma, 
                    policy_kwargs=policy_kwargs, verbose = 1, learning_rate=learning_rate, device='cpu')
    elif args.num_algo == 2:
        env = load_environment(id, rank = 0, seed = args.seed*10)()
        test_env = env

        algo = 'PPO_AVEC'
        policy_kwargs = dict(net_arch=[dict(pi=size_layer, vf=size_layer)])
        #model = PPO2('MlpPolicy', env, avec_coef=1., vf_coef=value_coeff, n_steps=timer_limit*20, nminibatches = 40, gamma=gamma, policy_kwargs=policy_kwargs, verbose = 1, learning_rate=learning_rate)
    elif args.num_algo == 3 and cont:
        env = load_environment(id, rank = 0, seed = args.seed*10)()
        test_env = env

        algo = 'SAC_AVEC'
        layers = size_layer
        #model = SAC_AVEC('CustomSACPolicy', env, avec_coef=1., value_coef=value_coeff, policy_kwargs={"layers":layers}, verbose=1, gamma=gamma, learning_rate=learning_rate, batch_size = batch_size, ent_coef='auto', learning_starts=500)
    else:
        if not cont and args.num_algo in [0, 3]:
            print("[ERROR] >> SAC is used with continue action space.")
        else:
            print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
        exit(1)

    name = algo + "_" + id + "_" + str(args.seed*10)
    os.makedirs("./Results_benchmark/" + name, exist_ok = True)


    seed = args.seed*10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    #env.seed(seed)
    env.action_space.np_random.seed(seed)

    rewards, final_rewards, steps = [], [], []
    best = -100000
    last_el = 0

    idx = 0
    print("\n-------------------------------")
    print(">>>    Start")
    print("-------------------------------\n")
    start_time = time.time()

    if args.restart != 0:
        idx = args.restart
        print(">>>    Restart training from n°", idx+1)
        del model
        save_path = "./Results_benchmark/" +  name + "/latest"

        if args.num_algo == 0 and cont:
            model = SAC.load(save_path, env=env)
        elif args.num_algo == 1:
            model = PPO.load(save_path, env=env)
        #elif args.num_algo == 2:
            #model = PPO2.load(save_path, env=env)
        #elif args.num_algo == 3 and cont:
            #model = SAC_AVEC.load(save_path, env=env)
        else:
            if not cont and args.num_algo in [0, 3]:
                print("[ERROR] >> SAC is used with continue action space.")
            else:
                print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
            exit(1)
        
        with open("./Results_benchmark/" +  name + "/rewards_"+id+".txt", 'r') as fp:
            rewards, steps = json.load(fp)
        with open("./Results_benchmark/" +  name + "/final_rewards_"+id+".txt", 'r') as fp:
            final_rewards, steps = json.load(fp)
        best = max(rewards)
        print("Restored values:")
        print("\tSteps: ", steps)
        print("\tRewards: ", rewards)
        print("\tFinal rewards: ", final_rewards)
        print("\tBest reward: ", best)

    if args.training:
        while idx < n_epochs:

           try:    
               print("\n-------------------------------")
               print(">>>    Start training n°", idx+1)
               print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
               print("[INFO]  >>    scene: ", id)
               print("[INFO]  >>    algo: ", algo)
               print("[INFO]  >>    seed: ", seed)
               print("-------------------------------\n")

               model.learn(total_timesteps=timer_limit*20, log_interval=20)
               model.save("./Results_benchmark/" + name + "/latest")

               print("\n-------------------------------")
               print(">>>    Start test n°", idx+1)
               print("[INFO]  >>    scene: ", id)
               print("[INFO]  >>    algo: ", algo)
               print("[INFO]  >>    seed: ", seed)
               print("-------------------------------\n")

               r, final_r = test(test_env, model, idx, n_test=5)
               final_rewards.append(final_r)
               rewards.append(r)
               steps.append(timer_limit*20*(idx+1))


               with open("./Results_benchmark/" +  name + "/rewards_"+id+".txt", 'w') as fp:
                   json.dump([rewards, steps], fp)
               with open("./Results_benchmark/" +  name + "/final_rewards_"+id+".txt", 'w') as fp:
                   json.dump([final_rewards, steps], fp)

               if r >= best:
                   print(">>>    Save training n°", idx+1)
                   model.save("./Results_benchmark/" +  name + "/best")

               idx+=1
           except:
                print("[ERROR]  >> The simulation failed. Restart from previous id.")

        model.save("./Results_benchmark/" +  name + "/latest")

        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))

    if args.testing:
        print(">>>    Start testing")
        del model
        save_path = "./Results_benchmark/" +  name + "/best"

        if args.num_algo == 0 and cont:
            model = SAC.load(save_path)
        elif args.num_algo == 1:
            model = PPO.load(save_path)
        #elif args.num_algo == 2:
            #model = PPO2.load(save_path)
        #elif args.num_algo == 3 and cont:
            #model = SAC_AVEC.load(save_path)
        else:
            if not cont and args.num_algo in [0, 3]:
                print("[ERROR] >> SAC is used with continue action space.")
            else:
                print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
            exit(1)

        r, final_r = test(test_env, model, -1, n_test=5, render = True)
        print("[INFO]  >>    Best reward : ", r, " - Final reward:", final_r)
