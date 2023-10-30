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
...

## Installation
### Requirements
- [Python 3.8](https://www.python.org/downloads/release/python-3810/) +
- Tested on:
	- Ubuntu 20.04 with Python 3.8.10
	- gcc-9, g++-9
	- SOFA v22.06
- For installing **SOFA v22.06**, you can choose between:
  - [SOFA v22.06 binaries installation](https://github.com/sofa-framework/sofa/releases/tag/v22.06.00) (faster option)
  - [Build and compile SOFA v22.06](https://www.sofa-framework.org/community/doc/getting-started/build/linux/) with mandatory plugins:
    * [SofaPython3](https://github.com/sofa-framework/SofaPython3) (fetchable within sofa) 
    * [BeamAdapter](https://github.com/sofa-framework/BeamAdapter) (fetchable within sofa)
    * [SPLIB](https://github.com/SofaDefrost/SPLIB)
    * [STLIB](https://github.com/SofaDefrost/STLIB)
    * [SoftRobots](https://github.com/SofaDefrost/SoftRobots)
    * [ModelOrderReduction](https://github.com/SofaDefrost/ModelOrderReduction)
    * [Cosserat](https://github.com/SofaDefrost/plugin.Cosserat)
    * Note: [Plugins installation](https://www.sofa-framework.org/community/doc/plugins/build-a-plugin-from-sources/#in-tree-build) with a in-tree build is preferred.

### Install modules and requirements
Our toolkit currently works with `gym` v0.21.0 and `stable-baselines3` v1.6.2.

**Mandatory** - You need to install python packages and the `sofagym` module for using and testing our framework:
```
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
export SOFA_ROOT=<path>/<to>/<sofa>/<build>
export PYTHONPATH=<path>/<to>/<python3>/<site-packages>:$PYTHONPATH
```
For example, if you have installed SOFA binaries, you should launch something similar to:
```python
export SOFA_ROOT=/home/andreaprotopapa/SOFA/v22.06.00
export PYTHONPATH=/home/andreaprotopapa/SOFA/v22.06.00/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
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
  - Unmodeled variant for each the TrunkPush
- DR parametric distributions: uniform, normal, truncnormal
- Automatic sampling of new dynamics when `env.reset()` is called
  
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
### 2. Dynamics Parameters Inference
### 3. Policy Training
### 4. Evaluation

## Examples
### TrunkReach
- Inference 
- Policy Training
- Evaluation
### TrunkPush
- Inference 
- Policy Training
- Evaluation
### TrunkLift 
- Policy Training - fixed DR
- Evaluation
### Multigait 
- Policy Training - fixed DR
- Evaluation

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
