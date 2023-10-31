# SOFA-DR-RL: Training Reinforcement Learning Policies for Soft Robots with Domain Randomization in SOFA Framework

This repository contains the code for the [paper](https://arxiv.org/abs/2303.04136) "**Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**" (Gabriele Tiboni, Andrea Protopapa, Tatiana Tommasi, Giuseppe Averta - IROS2023), here presented as an easy-to-use extension for SofaGym and SOFA Framework.

[Preprint](https://arxiv.org/abs/2303.04136) / [Website](https://andreaprotopapa.github.io/dr-soro/) / [Video](https://andreaprotopapa.github.io/dr-soro/)

## Abstract
Soft robots are gaining popularity due to their safety and adaptability, and the SOFA Framework plays a crucial role in this field by enhancing soft robot modeling and simulation. However, modeling complexity, often approximated, challenges the efficacy of reinforcement learning (RL) in real-world scenarios due to a significant domain gap between simulations and physical platforms.

In this work, we leverage the [SOFA simulation platform](https://github.com/sofa-framework/sofa) to demonstrate how Domain Randomization (DR) enhances RL policies for soft robots. Our approach improves robustness against unknown dynamics parameters and drastically reduces training time by using simplified dynamic models. We introduce an algorithmic extension for offline adaptive domain randomization (RF-DROPO) to facilitate sim-to-real transfer of soft-robot policies. Our method accurately infers complex dynamics parameters and trains robust policies that transfer to the target domain, especially for contact-reach tasks like cube manipulation. 

All DR-compatible benchmark tasks and our method's implementation are accessible as a user-friendly extension of the [SofaGym](https://github.com/SofaDefrost/SofaGym) framework. This software toolkit includes essential elements for applying Domain Randomization to any SOFA scene within a [Gym](https://github.com/openai/gym) environment, using the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) library for Reinforcement Learning training, allowing for the creation of multiparametric SOFA scenes and training control policies capable of achieving Sim2Real transfer. Example scenes are provided to guide users in effectively incorporating SOFA simulations and training learning algorithms.

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/670be649-b3fa-4b34-b715-41d4ad8688b4 alt="Offline Adaptive DR paradigm for soft robots." width="700"/>
</p>

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Details](#details)
4. [Examples](#examples)
5. [Citing](#citing)

## Installation
### Requirements
- [Python 3.8](https://www.python.org/downloads/release/python-3810/) +
- Tested on:
	- Ubuntu 20.04 with Python 3.8.10
	- gcc-9, g++-9
	- SOFA v22.06
- For installing **SOFA v22.06**, you can choose between:
  - [SOFA v22.06 binaries installation](https://github.com/sofa-framework/sofa/releases/tag/v22.06.00) (faster option)
  - [Build and compile SOFA v22.06](https://www.sofa-framework.org/community/doc/getting-started/build/linux/)
- Mandatory plugins:
    * [SofaPython3](https://github.com/sofa-framework/SofaPython3)
    * [BeamAdapter](https://github.com/sofa-framework/BeamAdapter)
    * [STLIB](https://github.com/SofaDefrost/STLIB)
    * [SoftRobots](https://github.com/SofaDefrost/SoftRobots)
    * [ModelOrderReduction](https://github.com/SofaDefrost/ModelOrderReduction)
    * [Cosserat](https://github.com/SofaDefrost/plugin.Cosserat)
    * Note: [Plugins installation](https://www.sofa-framework.org/community/doc/plugins/build-a-plugin-from-sources/#in-tree-build) with a in-tree build is preferred.

### Install modules and requirements
Our toolkit currently works with `gym` v0.21.0 and `stable-baselines3` v1.6.2.

**Mandatory** - You need to install python packages and the `sofagym` module for using and testing our framework:
```
pip install setuptools==65.5.0 "wheel<0.40.0"
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r ./required_python_libs.txt
pip install -e ./sofagym
```
**Optional** - If you want to use a specific Domain Randomization algorithm different from Uniform Domain Randomization (UDR), you have to install it as follows:
- **RF-DROPO**
```
pip install -r ./sb3-gym-soro/methods/dropo-dev/required_python_libs.txt
pip install -e ./sb3-gym-soro/methods/dropo-dev
```
- **BayesSim**
```
pip install -r ./sb3-gym-soro/methods/bayessim-replay/delfi/required_python_libs.txt
pip install -e ./sb3-gym-soro/methods/bayessim-replay/delfi
pip install -r ./sb3-gym-soro/methods/bayessim-replay/required_python_libs.txt
pip install -e ./sb3-gym-soro/methods/bayessim-replay
```

## Quick Start
To make SofaGym able to run SOFA, you need to set some enviromental variables:
```python
export PYTHONPATH=<path>/<to>/<python3>/<site-packages>:$PYTHONPATH
export PYTHONPATH=<path>/<to>/<sofa-dr-rl>/sofagym/stlib3:$PYTHONPATH
export SOFA_ROOT=<path>/<to>/<sofa>/<build>
```
For example, if you have installed SOFA binaries, you should launch something similar to:
```python
export PYTHONPATH=~/SOFA/v22.06.00/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
export PYTHONPATH=~/code/sofa-dr-rl/sofagym/stlib3:$PYTHONPATH
export SOFA_ROOT=~/SOFA/v22.06.00
```
This software toolkit is organized in two main parts, described as follows:
- **sb3-gym-soro** contains all the code for RL training algorithms, with the use of Domain Randomization techinques. 
Work inside this directory for any experiment and test.
- **sofagym** contains the API provided by [SofaGym](https://github.com/SofaDefrost/SofaGym) for creating standard Gym enviroments for Soft Robots interfaced with the SOFA simulator. This toolkit has been extended for integrating Domain Randomization techinques.

Test this implementation on the *TrunkCube* Gym environment with `sb3-gym-soro/test.py`. This script shows the result of a trained policy using RF-DROPO as DR method for the *TrunkPush* task.
```
cd sb3-gym-soro
python test.py --test_env trunkcube-v0 --offline --test_render
```
See below for more examples on testing the toolkit, in the [Examples section](https://github.com/andreaprotopapa/sofa-dr-rl/tree/main#examples).

### Features
- Gym environments for Soft Robots with Domain Randomization support: *TrunkReach*, *TrunkPush*, *TrunkLift*, and *Multigait*
  - Unmodeled variant for each the *TrunkPush*
- DR parametric distributions: uniform, normal, truncnormal
- Automatic sampling of new dynamics when `env.reset()` is called
- DR inference methods: *RF-DROPO*, *BayesSim*
  
### Environments
| Gym name          | task                                                             | dim $\xi$ | $\xi$                                                                                                    | unmodeled variant |
|-------------------|------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------|-------------------|
|`trunk-v0`         | *TrunkReach*: reach a target goal position                       | 3         | Trunk Mass, Poisson Ratio, Young's Modulus                                                               | -                 |
|`trunkcube-v0`     | *TrunkPush*: push a cube to a target goal position               | 5         | Cube Mass, Friction Coefficient, Trunk Mass, Poisson Ratio, Young's Modulus                              | yes               |
|`trunkwall-v0`     | *TrunkLift*: lift a flat object in the presence of a nearby wall | 1         | Wall Position                                                                                            | -                 |
|`multigaitrobot-v0`| *Multigait*: walking forward with the highest speed              | 3         | Multigait mass, PDMS Poisson Ratio, PDMS Young's Modulus, EcoFlex Poisson Ratio, EcoFlex Young's Modulus | -                 |


where $\xi \in \mathbb{R}^{dim \ \xi}$ is the dynamics parameter vector. The unmodeled variants represent under-modeled parameterizations of the environments where dynamics parameters not included are misidentified by 20% (read more in Sec. V-B of our [work](https://arxiv.org/abs/2303.04136)).

### Getting Started
Inside  **sb3-gym-soro**, the workflow of each training pipeline follows this skeleton:
```
import gym
from sofagym import *

env = gym.make('trunkcube-v0')

env.set_dr_distribution(dr_type='truncnorm', distr=[0.06, 0.004, 0.30, 0.0015, 0.52, 0.003, 0.45, 0.0004, 5557.07, 2.44])  # Randomize dynamics parameters following a truncated normal distribution
env.set_dr_training(True)

# ... train a policy

env.set_dr_training(False)

# ... evaluate policy in non-randomized env
```

## Details
### 0. Randomized configuration of the Gym environment
Each Gym environment is defined inside `sofagym`, as an extension of pre-existing enviroments of the [SofaGym](https://github.com/SofaDefrost/SofaGym) API. To allow the use of Domain Randomization techinques, two main steps are required:

1. Augment the simulated environment (e.g., `TrunkEnv.py`) with the following methods to allow Domain Randomization and its optimization:
  - `env.set_task(*new_task) # Set new dynamics parameters`
  - `env.get_task() # Get current dynamics parameters`   
  - `env.get_search_bounds(i) # Get search bounds for a specific parameter optimized`
  - `env.get_search_bounds_all() # Get search bounds for all the parameters optimized`
  - `env.get_task_lower_bound(i) # Get lower bound for i-th dynamics parameter`
  - `env.get_task_upper_bound(i) # Get upper bound for i-th dynamics parameter`

2. Create a randomized configuration (e.g., `Trunk_random_config.json`), where all the details of each dynamics parameter are specified:
```python
{

...

"dynamic_params": ["trunkMass", "trunkPoissonRatio", "trunkYoungModulus"],
"dynamic_params_values": [0.42, 0.45, 4500],

"trunkMass_init": 0.42,
"trunkMass_min_search": 0.005,
"trunkMass_max_search": 1.0,
"trunkMass_lowest": 0.0001,
"trunkMass_highest": 10000,

...
}
```
  - For each dynamics parameter to be randomized, set:
    - the name (inside `dynamic_params`)
    - the target value (inside `dynamic_params_values`)
    - the initial value (`_init`)
    - the search bounds (`_min_search` and `_max_search`)
    - the physical bounds (`_lowest` and `_highest`)

### 1. Dataset collection and formatting
Prior to running the code for the *inference* phase, an offline dataset of trajectories from the target (real) environment needs to be collected. This dataset can be generated either by rolling out any previously trained policy, or by kinesthetic guidance of the robot.

The `dataset` object must be formatted as follows:
```
    n : int
          state space dimensionality
    a : int
          action space dimensionality
    t : int
          number of state transitions

    dataset : dict,
          object containing offline-collected trajectories

    dataset['observations'] : ndarray
          2D array (t, n) containing the current state information for each timestep

    dataset['next_observations'] : ndarray
          2D array (t, n) containing the next-state information for each timestep

    dataset['actions'] : ndarray
          2D array (t, a) containing the action commanded to the agent at the current timestep

    dataset['terminals'] : ndarray
          1D array (t,) of booleans indicating whether or not the current state transition is terminal (ends the episode)
```
### 2. Dynamics Parameters Inference
We offer two distinct methods for inferring the dynamics parameters:

1. **ResetFree-DROPO** (**RF-DROPO**): Our proprietary method, developed as an extension of [DROPO](https://github.com/gabrieletiboni/dropo). In this approach, we relax the original assumption of resetting the simulator to each visited real-world state. Instead, we consider that we only know the initial full configuration of the environment, and actions are replayed in an open-loop fashion, always starting from the initial state configuration. For further details, please refer to Sec. IV-A in our [paper](https://arxiv.org/abs/2303.04136).

2. **[BayesSim](https://github.com/rafaelpossas/bayes_sim/tree/master)**: This method represents the classical baseline in Domain Randomization, adapted here to the offline inference setting by replaying the original action sequence during data collection.

Both of these methods are accessible within the `sb3-gym-soro/methods` directory.
As the output, we generate a distribution of the dynamics parameters saved in an `.npy` file. You can refer to the `sb3-gym-soro/BestBounds` directory to access previous inference results that we have made available.

### 3. Policy Training
The primary objective of Domain Randomization is to randomly sample new dynamics parameters, denoted as $\xi$, from the distribution $p_\phi(\xi)$ at the beginning of each training episode. If an inference algorithm like *RF-DROPO* or *BayesSim* has been used, then $p_\phi(\xi)$ represents the output from the previous step.

Additionally, we have included another baseline method known as **Uniform Domain Randomization** (**UDR**). Unlike the aforementioned inference-based approaches, UDR does not require an inference step, as $p_\phi(\xi)$ is a uniform distribution that is statically fixed in the randomized configuration file of the environment.

Upon training the agent in the source environment for a specified number of `timesteps`, the optimal policy is obtained as output and is saved in `best_model.zip`.

### 4. Evaluation
To evaluate the effectiveness of various methods in a Sim-to-Real setting, it is common practice to start with a Sim-to-Sim scenario. This allows us to test the transferability of learned policies using simulation alone. To do this, we initially worked in a source environment where the dynamics parameters were unknown. Our aim was to estimate an optimal policy that would be suitable for the unknown target domain.
Subsequently, we can now evaluate the learned policy by applying it to a target simulated environment with the nominal target dynamics parameters that we attempted to infer during the inference phase.

## Examples
Notes:
- Each of the following examples should be executed within the training directory `sb3-gym-soro`. Therefore, please ensure that you change the current working directory to this location (i.e., `cd sb3-gym-soro`).
- Our toolkit is integrated with `wandb`. If you wish to use it, remember to log in beforehand and include the corresponding option in the command (i.e., `--wandb_mode online`).
- To parallelize the inference or policy training execution, use the dedicated `--now` parameter.
- Please note that both the *inference* phase and *policy training* are relatively time-consuming experiments required to reach convergence. If you are primarily interested in our results, you can quickly evaluate some pre-trained policies that we have made available in the `sb3-gym-soro/example-results` directory or following the commands reported in **Evaluation**.
  - During the evaluation of a learned policy, it is possible to visualize the execution of the task with the option `--test_render`.
- Additionally, the datasets and distributions of dynamics parameters that have already been inferred are provided in the `sb3-gym-soro/Dataset` and `sb3-gym-soro/BestBounds` directories, respectively.
### TrunkReach

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/47170f5d-9b51-48db-9f42-0e61ff083476 alt="trunkreach" width="400"/>
</p>

For this task, we offer various methods for training with Domain Randomization, including *RF-DROPO* (our method), *BayesSim*, and *UDR*. To keep it simple, we will provide example commands for *RF-DROPO* here. However, you can refer to the in-code documentation of each method if you wish to try them as well.
- **Inference**
  - Dataset is here collected by executing a set of 100 random actions before the inference phase.
  - ```
    python train_dropo.py --env trunk-v0 --test_env trunk-v0 --seed 0 --now 1 -n 1 --budget 5000 --data random --clipping 100 --inference_only --run_path ./runs/RFDROPO --wandb_mode disabled
    ```
- **Policy Training**
  - Inference bounds (i.e., the dynamics parameters distributions) have here already been determined in a previous inference step and are simply loaded.
  - ```
    python train_dropo.py --env trunk-v0 --test_env trunk-v0 --seed 0 --now 1 -t 2000000  --training_only --run_path ./runs/RFDROPO --bounds_path ./BestBounds/Trunk/RFDROPO/seed0_8CK3V_best_phi.npy --wandb_mode disabled
    ```
- **Evaluation** (suggested for an out-of-the-box testing)
  - A control policy has here already been trained in a previous policy training step and is simply loaded.
  - ```
    python test.py --test_env trunk-v0 --test_episodes 1 --seed 0 --offline --load_path ./example-results/trunk/RFDROPO/2023_02_28_20_31_32_trunk-v0_ppo_t2000000_seed2_login027851592_TM84F --test_render
    ```
### TrunkPush

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/87781dcb-ca14-487e-b276-f47795910501 alt="trunkpush" width="400"/>
</p>

For this task, we offer various methods for training with Domain Randomization, including *RF-DROPO* (our method), *BayesSim*, and *UDR*. To keep it simple, we will provide example commands for *RF-DROPO* here. However, you can refer to the in-code documentation of each method if you wish to try them as well.

It is also possible to train on an unmodeled setting, by using the option `--unmodeled`, which referers to the use of a different randomized configuration file (i.e., `TrunkCube_random_unmodeled_config.json`).

- **Inference**
  -  Dataset has here been pre-collected by a semi-converged policy and is simply loaded.
  - ```
    python train_dropo.py --env trunkcube-v0 --test_env trunkcube-v0  --seed 0 --now 1 -eps 1.0e-4 -n 1 --budget 5000 --data custom --data_path ./Dataset/TrunkCube/20230208-091408_1episodes.npy --inference_only --run_path ./runs/RFDROPO --wandb_mode disabled
    ```
- **Policy Training**
  - Inference bounds (i.e., the dynamics parameters distributions) have here already been determined in a previous inference step and are simply loaded.
  - ```
    python train_dropo.py --env trunkcube-v0 --test_env trunkcube-v0 --seed 0 --now 1 -t 2000000  --training_only --run_path ./runs/RFDROPO --bounds_path ./BestBounds/TrunkCube/RFDROPO/bounds_A1S0X.npy --wandb_mode disabled
    ```
- **Evaluation** (suggested for an out-of-the-box testing)
  - A control policy has here already been trained in a previous policy training step and is simply loaded.
  - ```
    python test.py --test_env trunkcube-v0 --test_episodes 1 --seed 0 --offline --load_path ./example-results/trunkcube/RFDROPO/2023_07_10_11_34_58_trunkcube-v0_ppo_t2000000_seed1_7901a3c94a22_G0QXG --test_render
    ```
### TrunkLift 

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/78eb23cf-9a8d-4d48-91d1-c818576f3748 alt="trunklift" width="400"/>
</p>

For this example, we did not perform the inference of dynamics parameter distributions. Our focus was on examining the impact of randomizing the wall position during training (as defined in the corresponding `TrunkWall_random_config.json`). Read more in Sec. V-D of our [work](https://arxiv.org/abs/2303.04136) for further details.

- **Policy Training - fixed DR**
- ```
    python train.py --env trunkwall-v0 --algo ppo --now 1 --seed 0 -t 2000000 --run_path ./runs/trunkwall --wandb_mode disabled
    ```
- **Evaluation** (suggested for an out-of-the-box testing)
  - A control policy has here already been trained in a previous policy training step and is simply loaded.
  - ```
    python test.py --test_env trunkwall-v0 --test_episodes 1 --seed 0 --offline --load_path ./example-results/trunkwall/2023_02_26_20_46_59_trunkwall-v0_ppo_t2000000_seed3_mn011935323_R922D --test_render
    ```
### Multigait 

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/0cf97be8-c4b3-4dd7-9d62-a9d897428499 alt="multi-red" width="400"/>
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/61cb8f3b-04a6-4d7c-bda3-29c070ff0711 alt="multi-compl" width="400"/>
</p>

For this example, we did not perform the inference of dynamics parameter distributions. Our focus was on examining the impact of randomization (as defined in the corresponding `MultiGaitRobot_random_config.json`) during training using a simplified model to then evaluate the performance on a more complex version of model.

We found that Domain Randomization is effective in enhancing robustness during training. This approach allows us to reduce the training time by utilizing simplified models for training while still achieving successful transfer of learned behavior to more accurate models during evaluation. Read more in Sec. V-C of our [work](https://arxiv.org/abs/2303.04136) for further details.
    
- **Policy Training - fixed DR**
  - ```
    python train_fixed_dr.py --env multigaitrobot-v0 --test_env multigaitrobot-v0 --eval_freq 12000 --seed 0 --now 1 -t 500000 --run_path ./runs/multigait --bounds_path ./BestBounds/MultiGait/gauss_bounds.npy --distribution_type truncnorm --wandb_mode disabled
    ```
- **Evaluation** (suggested for an out-of-the-box testing)
  - A control policy has here already been trained in a previous policy training step and is simply loaded.
  - It is possible to observe how the policy performs in both simplified and complex models by simply adjusting the value of the `reduced` attribute in the `MultiGaitRobot_random_config.json` file.
  - ```
    python test.py --test_env multigaitrobot-v0 --test_episodes 1 --seed 0 --offline --load_path ./example-results/multigait/2023_02_07_08_37_02_multigaitrobot-v0_ppo_t341000_seed1_hactarlogin358482_X54NP --test_render
    ```
## Troubleshooting
- If you are using a conda environment to run this tooolkit, you may fail in some errors with OpenGL libraries (e.g., `libGL error`). In this case you can try to install `conda install -c conda-forge libstdcxx-ng` or follow [this guide](https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris) for more troubleshooting.

## Citing
If you use this repository, please consider citing us:

```bibtex
@misc{tiboni2023dr_soro,
  doi = {10.48550/ARXIV.2303.04136},
  title = {Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots},
  author = {Tiboni, Gabriele and Protopapa, Andrea and Tommasi, Tatiana and Averta, Giuseppe},  
  keywords = {Robotics (cs.RO), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},  
  publisher = {arXiv},  
  year = {2023}
}
```
Also, consider to cite the original SofaGym work:
```bibtex
@article{schegg2022sofagym,
  title={SofaGym: An open platform for Reinforcement Learning based on Soft Robot simulations},
  author={Schegg, Pierre and M{\'e}nager, Etienne and Khairallah, Elie and Marchal, Damien and Dequidt, J{\'e}r{\'e}mie and Preux, Philippe and Duriez, Christian},
  journal={Soft Robotics},
  year={2022},
  publisher={Mary Ann Liebert, Inc., publishers 140 Huguenot Street, 3rd Floor New~…}
}
```
