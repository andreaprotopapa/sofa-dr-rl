# Training Reinforcement Learning Policies for Soft Robots with Domain Randomization in SOFA Framework

This repository contains the code for the [paper](https://arxiv.org/abs/2303.04136) "**Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**" (Gabriele Tiboni, Andrea Protopapa, Tatiana Tommasi, Giuseppe Averta - IROS2023), here presented as an easy-to-use extension for SofaGym and SOFA Framework.

[Preprint](https://arxiv.org/abs/2303.04136) / [Website](https://andreaprotopapa.github.io/dr-soro/) / [Video](https://andreaprotopapa.github.io/dr-soro/)

## Abstract
Soft robots are gaining popularity due to their safety and
adaptability, and the SOFA Framework plays a crucial role in this field
by enhancing soft robot modeling and simulation. However, the complexity
of modeling, often approximated, challenges the efficacy of reinforcement
learning (RL) in real-world scenarios due to a significant domain
gap between simulated models and physical platforms. In this work, by
leveraging SOFA simulation platform, we demonstrate how Domain Randomization
(DR) can improve RL policies for soft robots with: i) robustness
w.r.t. unknown dynamics parameters; ii) reduced training time by
exploiting drastically simpler dynamic models for learning. Moreover, we
introduce an algorithmic extension for offline adaptive domain randomization
for sim-to-real transfer of soft-robot policies. Our method accurately
infers complex dynamics parameters and trains robust policies that
transfer to the target domain, especially for contact-reach tasks like cube
manipulation. All DR-compatible benchmark tasks and our method’s implementation
are available as an easy-to-use extension of [SofaGym framework](https://github.com/SofaDefrost/SofaGym).

<p align="center">
  <img src=https://github.com/andreaprotopapa/sofa-dr-rl/assets/44071949/670be649-b3fa-4b34-b715-41d4ad8688b4 alt="Offline Adaptive DR paradigm for soft robots." width="700"/>
</p>

## Table of Contents
...

## Installation
### Requirements
### Prerequisites
### Install modules

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
