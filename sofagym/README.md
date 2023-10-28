# Use the pre-built docker
If you want you can pull [this docker](https://hub.docker.com/repository/docker/andreaprotopapa/sofa) already set with all the dependecies and installations. The structure would be the following:
```
MySofa/
├── ext_plugin_repo/
├── sb3-gym-soro/
├── sofa/
└── sofagym/
```
Where `sb3-gym-soro` is [a git repo](https://github.com/andreaprotopapa/sb3-gym-soro) containing the code used for training and testing through Stable Baselines3.

After the pulling of the docker image, you can:
- create a new container: 
	```
	docker run -it andreaprotopapa/sofa bash
	```
- restart an existing container: 
	```
	docker start -ai <container_name>
	```
Inside the container, this is an example of usage to start a training.
```
sudo apt install nano
sudo apt-get install git-all
. activate sofa_source
conda deactivate
conda activate sofa_source
export PYTHONPATH=/home/aprotopapa/code/MySofa/sofa/build/lib/python3/site-packages
export SOFA_ROOT=/home/aprotopapa/code/MySofa/sofa/build
wandb login
...
cd sb3-gym-soro
python train.py --env trunk-v0 --algo ppo --now 7 --seed 1 -t 1000000 --eval_freq 10000
```
Note:
- To test if your enviroment is correctly set, you can start SOFA in no-GUI mode:
```
./sofa/build/bin/runSofa -g batch ./ext_plugin_repo/SoftRobots/docs/sofapython3/tutorials/CableGripper/details/step6.py
```
- It is raccomended to do your wandb login before starting the training (see the code above)
- It is raccomanded to do your github login and pull from the `sofagym` and `sb3-gym-soro` directories before starting the training
- If the internet connection doesn't seem to start, you probably have to set some network permission configuration in your docker. This is an example working for `nike` host in VANDAL:
	```
	# For internet connection
	sudo su
	echo 'Acquire::http::Proxy "http://proxy.polito.it:8080/";' > /etc/apt/apt.conf.d/proxy.conf
	echo 'Acquire::https::Proxy "http://proxy.polito.it:8080/";' >> /etc/apt/apt.conf.d/proxy.conf
	```
# Installation - Create repo from scratch
## Requirements
- [Python 3.8](https://www.python.org/downloads/release/python-3810/) +
- Tested on:
	- Ubuntu 20.04 with Python 3.8.10
	- gcc-9, g++-9
	- Sofa v22.06
## Setup 
To set up clean repositories, we recommend to arrange the project directories as follows:
```
MySofa/
├── ext_plugin_repo/
│   ├── BeamAdapter/
│   ├── Cosserat/
│   ├── ModelOrderReduction/
│   ├── SoftRobots/
│   ├── SPLIB/
│   ├── STLIB/
│   ├── CMakeLists.txt
├── sofa/
│   ├── build/
│   ├── src/
└── sofagym/
```
- build and compile sofa in `sofa`: https://www.sofa-framework.org/community/doc/getting-started/build/linux/ (read the NOTE before compiling)
- clone these plugin repos in `ext_plugin_repo` following these [instructions](https://www.sofa-framework.org/community/doc/plugins/build-a-plugin-from-sources/) (read the NOTE for more information)
    - [BeamAdapter](https://github.com/SofaDefrost/BeamAdapter)
    - [Cosserat](https://github.com/SofaDefrost/plugin.Cosserat)
    - [ModelOrderReduction](https://github.com/SofaDefrost/ModelOrderReduction)
    - [SoftRobots](https://github.com/SofaDefrost/SoftRobots)
    - [SPLIB](https://github.com/SofaDefrost/SPLIB)
    - [STLIB](https://github.com/SofaDefrost/STLIB)
- clone this repository in `sofagym`
	- `pip install -r required_python_libs.txt`	 
	- `pip install -e .`	

After all the installations, it is necessary to:
-  ~~change `sofa/build/lib/python3/site-packages/splib3` content with `sofagym/stdlib3/splib` content~~ (Need to be tested)

**NOTE**:
1. If you are using a virtual enviroment (e.g., a conda enviroment set in `envs/sofa`) with a specific python version installed, you must change some values before you configure `cmake-gui`:
	- PYTHON_INCLUDE_DIRS: ...envs/sofa/include/python3.8/
	- PYTHON_LIBRARIES: ...envs/sofa/lib/libpython3.8.so
	- PYTHON_VERSION: 3.8.10
	- SP3_PYTHON_PACKAGES_DIRECTORY: ...envs/sofa/lib/python3.8/site-packages/
	- SP3_PYTHON_PACKAGES_LINK_DIRECTORY: ...envs/sofa/lib/python3.8/site-packages/
	- PYBIND:
		- 	```
			conda activate sofa
			pip install pybind11
	  		```
		- pybind11_DIR: ...envs/sofa/lib/python3.8/site-packages/pybind11/share/cmake/pybind11
2. For each plugin, you need to add it to Sofa through the GUI (launching `./sofa/build/bin/runSofa`) in `Edit > Plugin Manager > Add > sofa/build/lib/<plugin>.so`. If you don't find it (e.g., ModelOrderReduction), go to `./sofa/build/external_directories/ext_plugin_repo/<plugin>/lib` and copy each one of the following files in this order in the common `lib` folder:
 	- `<plugin>.so.<version>`
	- `<plugin>.so`
3. To make SofaGym able to run Sofa, you need to set an enviromental variable:
	- `export SOFA_ROOT= <path>/<to>/<sofa>/<build>`
		- example: `export SOFA_ROOT=/home/aprotopapa/code/MySofa/sofa/build`
	- `export PYTHONPATH=<path>/<to>/<python3>/<site-packages>`
		- example: `export PYTHONPATH=/home/aprotopapa/code/MySofa/sofa/build/lib/python3/site-packages`
 
## Testing
- For testing Sofa enviroment with SoftRobots plugin:
	```
	./sofa/build/bin/runSofa ./ext_plugin_repo/SoftRobots/docs/sofapython3/tutorials/CableGripper/details/step6.py
	```
- For testing SofaGym enviroment:
	- 	```
		python ./sofagym/test_env.py
		```
	- 	```
		python ./sofagym/rl.py -ne 5 -na 1 -nc 12 -s 10
		```
- If you want to try an enviroment scene on Sofa software (that allows a better and interactive visualization), e.g. `TrunkScene.py`, you need to:
	- add the file to the PYTHONPATH variable
		- e.g., `export PYTHONPATH=$PYTHONPATH:/home/andrea/MySofa/SofaGym/sofagym/env/Trunk/TrubkScene.py`
	- edit `sofa/build/lib/python3/site-packages/splib/obkectmodel/__init__.py`, adding this block at line n.50:
		```
		verbose = True
				if o.__module__ in sys.modules.keys(): #if it exists the module in the dict of modules
					path = os.path.abspath(sys.modules[o.__module__].__file__)
					print(f"{o.__module__} in sys.modules")
					if verbose:
						print(path)
						print(str(os.path.dirname(path)))
				else:
					print(f"{o.__module__} not in sys.modules")
					if verbose:
						print(f"sys.path: {sys.path}")
					if any(o.__module__ in s for s in sys.path): #if the module as name is contained in sys.path array
						path = [s for s in sys.path if o.__module__ in s][0] #take the first occurence between all the times the module is contained as name in sys.path array
						if verbose:
							print(path)
							print(str(os.path.dirname(path)))
					else:
						print(f"ERROR: {o.__module__} not in sys.path")
				
		```
		*NOTE: Probably it would be better to leave the modified SPLIB folder available instead of doing this, so you just need to change the original SPLIB folder with the new one.*
	- launch Sofa
		- e.g., `./sofa/build/bin/runSofa ./sofagym/sofagym/env/Trunk/TrunkScene.py`

# SofaGym - original from https://github.com/SofaDefrost/SofaGym

Software toolkit to easily create an OpenAI Gym environment out of any SOFA scene.
The toolkit provides an API based on the standard OpenAI Gym API, allowing to train classical Reinforcement Learning algorithms. The toolkit also comprises example scenes based on the SoftRobots plugin for SOFA to illustrate how to include SOFA simulations and train learning algorithms on them.

## Usage

The Gym framework allows to interact with an environment using well-known keywords:
- *step(a)*: allows to perform a simulation step when the agent performs the action *a*. Given the current state of the system *obs_t* and the action *a*, the environment then changes to a new state *obs_{t+1}* and the agent receives the reward *rew*. If the goal is reached, the *done* flag changes to *True*.
- *reset*: resets the environment.
- *render*: gives a visual representation of *obs_t*.

The use of this interface allows intuitive interaction with any environment, and this is what SofaGym allows when the environment is a Sofa scene. For more information on Gym, check the official documentation page [here](https://gym.openai.com/docs/).

Example of use

```python
import gym
import sofagym.env

env = gym.make('trunk-v0')
env.reset()

done = False
while not done:
    action = ... # Your agent code here
    state, reward, done, info = env.step(action)
    env.render()
    print("Step ", idx, " done : ",done,  " state : ", state, " reward : ", reward)

env.close()
```

The classic running of an episode is therefore:
- *env = gym.make(env_name)*: creation of the environment using the gym interface.
- *env.reset()*: Initialization of the environment at its starting point.
- *action = model(obs_t)* and *env.step(action)*: execution of actions in the environment.
- *env.close()*: end of simulation and destruction of the environment.



## Citing

If you use the project in your work, please consider citing it with:
```bibtex
@misc{SofaGym,
  authors = {Ménager, Etienne and Schegg, Pierre and Duriez, Christian and Marchal, Damien},
  title = {SofaGym: An OpenAI Gym API for SOFASimulations},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```
## The tools

### Server/worker architecture

The major difficulty encountered in this work is the fact that it is not possible to copy the *root* from a Sofa simulation. This implies that when two sequences of actions *A_1 = [a_1, ..., a_n, new_action_1]* and *A_2 = [a_1, ..., a_n, new_action_2]* have to be tried, it is necessary to start again from the beginning each time and simulate again *[a_1, ..., a_n]*. This leads to a huge loss of performance. To solve this problem a server/worker architecture is set up.

A server takes care of distributing the calculations between several clients. Each client *i* is associated with an action sequence *A_i = [a_{i1}, ...., a_{in}]*. Given an action sequence *A = [a_{1}, ...., a_{n}]* and a new action *a*, the server looks for the client with the action sequence *A_i*. This client forks and the child executes the new action *a*. The father and son are referenced to the server as two separate clients and the action sequence *[a_{1}, ...., a_{n}]* and *[a_{1}, ...., a_{n}, a]* can be accessed.

A cleaning system is used to close clients that are no longer used. This makes it possible to avoid having an exponential number of open clients.

When it is not necessary to have access to the different states of the environment, i.e. when the actions are used sequentially, only one client is open and performs the calculations sequentially.

### Vectorized environment


Simulation training can be time consuming. It is therefore necessary to be able to parallelise the calculations. Since the actions are chosen sequentially, it is not possible to parallelise the calculations for one environment. The result depends on the previous result. However, it is possible to parallelise on several environments, meaning to run several simulations in parallel. This is done with the baseline of OpenAI: SubprocVecEnv.


### Separation between visualisation and computations

SofaGym separates calculations and visualisation. In order to achieve this, two scenes must be created: a scene *A* with all visual elements and a scene *B* with calculation elements (solvers, ...). Scene *A* is used in a viewer and scene *B* in the clients. Once the calculations have been performed in scene *B*, the positions of the points are given to the viewer which updates scene *A*.

### Adding new environment


It is possible to define new environments using SofaGym. For this purpose different elements have to be created:
- *NameEnv*: inherits from *AbstractEnv*. It allows to give the specificity of the environment like the action domain (discrete or continuous) and the configuration elements.
- *NameScene*: allows to create the Sofa scene. It must have the classic createScene function and return a *root*. To improve performance it is possible to separate the visual and computational aspects of the scene using the *mode* parameter (*'visu'* or *'simu'*). It allows you to choose the elements in the viewer-related scene or in the client-related scene. We also integrate two Sofa.Core.Controller (rewardShaper and goalSetter) that allow to integrate goal and reward in the scene.
- *NameToolbox*: allows to customize the environment. It defines the functions to retrieve the reward and the state of the environment as well as the command to apply to the system (link between the Gym action and the Sofa command). Note that we can define the Sofa.Core.Controller here. 

These different elements make it possible to create and personalise the task to be performed. See examples of environments for implementation.

## The environments

### Gripper

The  Gripper  Environmentoffers  two  different  scenes.   In  both  scenes,  the objective is to grasp a cube and bring it to a certain height.  The closer the cube is to the target, the greater the reward.

The two scenes are distinguished by their action  space.   In  one  case  the  actions  are  discrete and correspond to a particular movement. We define a correspondence between a Gym action (int) and corresponding Sofa displacement and direction.

```python
env = gym.make("gripper-v0")
```

In the second case,  the actions are continuous  and  correspond  directly  to  a  movement  ofthe gripper’s fingers.  This difference is indicated when defining the environment

```python
env = gym.make("continuegripper-v0")
```

### Trunk

The Trunk environment offers two scenarios.  Both are based on the trunk robot.  The first is to bring the trunk’s tip to a certain position.

```python
env = gym.make("trunk-v0")
```

The second scenario is to manipulate a cup using the trunk to get the cup’s center of gravity in a predefined position.

```python
env = gym.make("trunkcup-v0")
```

The  Trunk  is  controlled  by  eight  cables  that can be contracted or extended by one unit.  There are therefore 16 possible actions. The action space presented here is discrete but could easily be ex-tended to become continuous.


### MultiGait Robot

The multigait Softrobot has  one  scene. The goal is to move the robot forward in the *x* direction with the highest speed.

```python
env = gym.make("multigaitrobot-v0")
```

### Maze

The Maze environment offers one scene  of a ball navigating in a maze. The maze is attached to the tripod robot and the ball is moved by gravity by modifying the maze’s orientation.

```python
env = gym.make("maze-v0")
```

The tripod is actuated by three servomotors. Similarly to the Trunk Environment, the Maze environment has a dicrete action space of 6 actions, moving  each  servomotor  by  one  increment,  and could easily be extended to be continuous.


## Results

In this section we demonstrate some use cases of the environments available in SofaGym, namely Reinforcement Learning, Imitation Learning, planning using Monte Carlo Tree Search and shape optimisation using Bayesian Optimization.

### Reinforcement Learning: Learning to grasp different objects with GripperEnv
### Imitation Learning: Learning to imitate an inverse controller with TrunkEnv
### Monte Carlo Tree Search: solving MazeEnv with planning


## Notes

1. At the moment the available action spaces are: continuous, discrete, tuple and dictionary.
