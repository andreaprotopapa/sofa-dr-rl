# SOFA-DR-RL: Training Reinforcement Learning Policies for Soft Robots with Domain Randomization in SOFA Framework

This repository contains the code for the [paper](https://arxiv.org/abs/2303.04136) "**Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**" (Gabriele Tiboni, Andrea Protopapa, Tatiana Tommasi, Giuseppe Averta - IROS2023), here presented as an easy-to-use extension for SofaGym and SOFA Framework.

[Preprint](https://arxiv.org/abs/2303.04136) / [Website](https://andreaprotopapa.github.io/dr-soro/) / [Video](https://andreaprotopapa.github.io/dr-soro/)

## Abstract
Soft robots are gaining popularity due to their safety and adaptability, and the SOFA Framework plays a crucial role in this field by enhancing soft robot modeling and simulation. However, modeling complexity, often approximated, challenges the efficacy of reinforcement learning (RL) in real-world scenarios due to a significant domain gap between simulations and physical platforms.

In this work, we leverage the [SOFA simulation platform](https://github.com/sofa-framework/sofa) to demonstrate how Domain Randomization (DR) enhances RL policies for soft robots. Our approach improves robustness against unknown dynamics parameters and drastically reduces training time by using simplified dynamic models. We introduce an algorithmic extension for offline adaptive domain randomization (RF-DROPO) to facilitate sim-to-real transfer of soft-robot policies. Our method accurately infers complex dynamics parameters and trains robust policies that transfer to the target domain, especially for contact-reach tasks like cube manipulation. 

All DR-compatible benchmark tasks and our method's implementation are accessible as a user-friendly extension of the [SofaGym](https://github.com/SofaDefrost/SofaGym) framework. This software toolkit includes essential elements for applying Domain Randomization to any SOFA scene within a [Gym](https://github.com/openai/gym) environment, using the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) library for Reinforcement Learning training, allowing for the creation of multiparametric Sofa scenes and training control policies capable of achieving Sim2Real transfer. Example scenes are provided to guide users in effectively incorporating SOFA simulations and training learning algorithms.

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
Mandatory:
- You need both modules for using and testing our framework
- **Install sofagym**
```
pip install -r sofagym/required_python_libs.txt
pip install -e ./sofagym
```
- **Install sb3-gym-soro**
```
pip install -r sb3-gym-soro/required_python_libs.txt
pip install -e ./sb3-gym-soro
```
Optionally:
- If you want to use a specific Domain Randomization algorithm different from Uniform Domain Randomization (UDR), you have to install it as follows:
- **RF-DROPO**
```
pip install -r sb3-gym-soro/methods/dropo-dev/required_python_libs.txt
pip install -e ./sb3-gym-soro/methods/dropo-dev/
```
- **BayesSim**
```
pip install -r sb3-gym-soro/methods/bayessim-replay/required_python_libs.txt
pip install -e ./sb3-gym-soro/methods/bayessim-replay/
```

## Quick Start

## Examples

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
