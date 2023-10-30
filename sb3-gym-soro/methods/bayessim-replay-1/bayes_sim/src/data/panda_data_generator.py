from src.sim.pendulum import *
from sklearn import preprocessing
from tqdm import tqdm
from stable_baselines.ppo2 import PPO2
from copy import deepcopy
import pickle
import time
import os
import dropo
import panda_gym
import glob

class PandaDataGenerator():
    def __init__(self, demo_file="", episodes_per_params=1, seed=1995,
                 steps_per_episode=200, sufficient_stats="Cross-Correlation",
                 load_from_file=False, assets_path=".", filename="",
                 env_name="PandaPushFixedStart-PosCtrl-v0", max_data=100,
                 noise_var=0., dyn_type="mfcom"):

        self.env_name = env_name
        self.env = gym.make(self.env_name)
        obs_max = np.array([np.inf]*27)
        obs_min = -obs_max
        self.env.observation_space = gym.spaces.Box(obs_min, obs_max)
        self.env.set_random_dynamics_type(dyn_type)
        self.env.set_dropo_mode()
        self.env.set_task_search_bounds()
        bounds = np.zeros(self.env.min_task.shape[0]*2)
        bounds[::2] = self.env.min_task
        bounds[1::2] = self.env.max_task
        print(bounds)
        self.env.set_udr_distribution(bounds)
        self.seed = seed
        self._raw_mjstate = deepcopy(self.env.get_sim_state())  # Save fresh full mjstate
        self.max_t = max_data

        self.actions = None

        self.expert_observations = np.load(glob.glob(os.path.join(demo_file, '*observations.npy'))[0])
        self.expert_actions = np.load(glob.glob(os.path.join(demo_file, '*actions.npy'))[0])
        self.expert_terminals = np.load(glob.glob(os.path.join(demo_file, '*terminals.npy'))[0])

        print("Loaded obs shape:", self.expert_observations.shape)

        try:
            self.expert_next_observations = np.load(glob.glob(os.path.join(demo_file, '*nextobservations.npy'))[0])
        except IndexError:
            self.expert_next_observations = self.expert_observations[1:]
            self.expert_observations = self.expert_observations[:-1]
            self.expert_actions = self.expert_actions[:-1]
            self.expert_terminals = self.expert_terminals[:-1]
            print("No separate next_obs file found, shifting observations")

        # Add noise to next observations
        self.expert_next_observations += np.random.randn(*self.expert_next_observations.shape) * np.sqrt(noise_var)

        print(f"data laoded {len(self.expert_actions)}")

        self.cached_data = None
        self.params_scaler = None

        param_inds = sorted(self.env.dynamics_indexes.keys())
        self.params = [self.env.dynamics_indexes[i] for i in param_inds]

        self.steps_per_episode = steps_per_episode
        self.sufficient_stats = sufficient_stats
        self.assets_path = assets_path
        self.load_from_file = load_from_file
        self.data_file = os.path.join(assets_path+filename)

        prior_low = self.env.min_task
        prior_high = self.env.max_task

    def reseed(self, *args):
        return

    @property
    def param_dim(self):
        dim = self.env.min_task.shape[0]
        return dim

    @property
    def feature_dim(self):
        sdim = self.env.observation_space.shape[0]
        adim = self.env.action_space.shape[0]

        feature_dim = sdim * adim + 2*sdim

        return feature_dim

    def save(self, data, file):
        """Saves data to a file."""
        f = open(file, 'wb')
        pickle.dump(data, f)
        f.close()

    def load(self, file):
        """Loads data from file."""
        f = open(file, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def get_real_data(self):
        ep_data = {"observation": self.expert_next_observations[:self.max_t],
                         "action": self.expert_actions[:self.max_t]}

        if self.sufficient_stats == "Cross-Correlation":
            ep_data = self.calculate_cross_correlation(ep_data)

        if self.sufficient_stats == "State-Action":
            ep_data = np.concatenate((ep_data['observation'], ep_data['action']), axis=1)

        return ep_data

    def scale_params(self, params):
        params_scaler = preprocessing.MinMaxScaler()
        params = params_scaler.fit_transform(np.array(params))
        cur_params = []
        # for body in params:
        #     for par in params[body]:
        #         cur_params.append(params[body][par])
        return params, params_scaler

    def calculate_cross_correlation(self, episode):
        n_steps = len(episode['action'])

        cur_state = episode['observation']
        #next_state = obs['observation'][idx][1:]
        cur_action = episode['action']
        sdim = cur_state.shape[1]
        adim = cur_action.shape[1]
        #state_difference = np.array(list(next_state - cur_state))
        state_difference = np.array(cur_state)
        actions = np.array(cur_action)
        sample = np.zeros((sdim, adim))
        for i in range(sdim):
            for j in range(adim):
                sample[i, j] = np.dot(state_difference[:, i], actions[:, j]) / (n_steps-1)
                # Add mean of absolut states changes and std to the summary statistics

        sample = sample.reshape(-1)
        sample = np.append(sample, np.mean(state_difference, axis=0))
        sample = np.append(sample, np.std(state_difference.astype(np.float64), axis=0))

        stats = np.array(sample)

        return stats

    def rollout(self):
        t = 0
        total_reward = 0
        cur_params = self.env.get_task()
        ep_tuple = ({"pendulum":
                        {f"mass{i}": cur_params[i] for i in range(len(cur_params))}},
                        {"observation": [], "action": []})

        self.env.seed(1995)

        state = self.env.reset()
        set_state = True

        for s, a, ns, d in zip(self.expert_observations,\
                self.expert_actions, self.expert_next_observations,\
                self.expert_terminals):
            t += 1

            # If new episode, reset and set initial state
            if set_state:
                self.env.reset()
                self.env.set_sim_state(self.env.get_full_mjstate(s, self._raw_mjstate))

            # Execute t-th action
            next_state, reward, _, _ = self.env.step(a)

            # Record the state
            ep_tuple[1]['action'].append(a)
            ep_tuple[1]['observation'].append(next_state.ravel())

            total_reward += reward

            # Should we reset?
            set_state = d

            if d: break

            if t > self.max_t:
                break

        action = np.array(ep_tuple[1]['action'])
        observation = np.array(ep_tuple[1]['observation'])
        ep_tuple[1]['action'] = action
        ep_tuple[1]['observation'] = observation

        return ep_tuple

    def gen_single(self, param):
        self.env.set_task(*param)

        all_data = {"data": None, "params": None}
        ep_tuple = self.rollout()

        if self.sufficient_stats == "Cross-Correlation":
            all_data["data"] = self.calculate_cross_correlation(ep_tuple[1])

        if self.sufficient_stats == "State-Action":
            all_data["data"] = np.concatenate((ep_tuple[1]['observation'], ep_tuple[1]['action']), axis=1)

        all_data["params"] = ep_tuple[0]

        return all_data

    def gen(self, n_samples, save=False):

        if self.load_from_file:
            print("Loading simulation data from disk: {}".format(self.data_file))
            all_params, all_data = self.load(self.data_file)

            if n_samples is not None and n_samples <= len(all_params):
                indexes = np.random.randint(0, len(all_params),n_samples)
            elif n_samples is not None and n_samples >= len(all_params):
                indexes = list(range(len(all_params)))

            return all_params[indexes], all_data[indexes]

        if self.cached_data is not None and n_samples == len(self.cached_data["params"]):
            print('Data has already been generated, loading from memory...')
            return self.cached_data["params"], self.cached_data["data"]
        else:
            self.cached_data = {"data": None, "params": None}

        all_params = []
        all_data = []
        print("\nDrawing Parameters and Running simulation...")
        for ep in tqdm(range(n_samples)):

            # if ep > 0 and ep % self.episodes_per_params == 0:
            #     mass = self.m_prior()
            #     length = self.l_prior()
            #     self.env.set_dynamics(mass=mass, length=length)
            params = self.env.sample_task()
            cur_params = np.copy(params)

            all_data.append(self.gen_single(cur_params)["data"])
            all_params.append(cur_params)

        all_params = np.array(all_params)
        all_data = np.array(all_data)

        self.cached_data["data"] = all_data
        self.cached_data["params"] = all_params

        if save:
            filename = os.path.join(self.assets_path, "pendulum_"+str(n_samples)+"_samples_"
                                    +self.sufficient_stats.lower()+"_"+time.strftime("%Y%m%d-%H%M%S")+".pkl")
            self.save((all_params, all_data), filename)

        return all_params, all_data

    def run_forward_model(self, true_obs, ntest=10):
        true_obs = np.ravel(true_obs)
        dt = []

        for _ in range(ntest):
            dt.append(self.gen_single(true_obs)["data"])

        # data = np.mean(data, axis=1)
        # params = np.mean(params, axis=1)

        return true_obs, np.mean(dt, axis=0)


if __name__ == "__main__":
    g = PendulumDataGenerator()
    dt = g.gen(10000, save=True)

    g_2 = PendulumDataGenerator(sufficient_stats="State-Action")
    dt_2 = g_2.gen(10000, save=True)

