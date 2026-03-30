# HA-MADDPG

**Hybrid-Action Multi-Agent Deep Reinforcement Learning for Multi-UAV Task Offloading, Trajectory Planning, and Resource Allocation in Air-Ground Integrated MEC**


## Overview

This repository provides the research implementation of a **hybrid-action multi-agent deep reinforcement learning framework** for **multi-UAV-assisted mobile edge computing (MEC)** systems. The proposed method is designed for dynamic **air-ground integrated** scenarios, where multiple unmanned aerial vehicles (UAVs) collaboratively serve mobile ground vehicles with time-varying task demands.

The core objective is to **minimize task accomplishment latency** through the joint optimization of:

- **UAV trajectory planning**
- **task scheduling / AGV selection**
- **collaborative multi-UAV service decisions**

To tackle the coexistence of continuous and discrete decision variables, the framework adopts a **hybrid-action MADDPG architecture**, where continuous actions are used for UAV motion control and discrete actions are modeled through a **Gumbel-Softmax-based differentiable action generation mechanism**.

This codebase is developed for our research on **multi-UAV collaborative edge computing**, with a focus on **task offloading**, **trajectory optimization**, and **delay-aware scheduling** in dynamic MEC environments.

---

## Highlights

- **Multi-UAV collaborative MEC environment** with dynamic AGV mobility
- **Hybrid-action MADDPG** for joint continuous-discrete decision making
- **Gumbel-Softmax reparameterization** for differentiable discrete action learning
- **Prioritized replay buffer** for more efficient experience sampling
- **Target actor-critic networks** for stable training
- **Curriculum-style training strategy** for improved convergence robustness
- **Delay-aware reward design** considering communication and computation effects

---

## Project Structure

```
.
├── env.py/                 # Environment simulation for MT-UAV MEC system
├── algorithm.py/           # Hybrid-action MADDPG algorithm implementation
├── train.py/               # Main training script
├── models/                 # Saved checkpoints during training
├── results/                # Reward curves and exported training data
├── requirements.txt        # Python Dependency list                
├── README.md               # Project documentation
```

## Environment Description

The simulation models a dynamic MEC system with the following components:

### System Scale

- **Service area:** 1500 m × 1500 m
- **Number of UAVs:** 4
- **Number of AGVs:** 10
- **Episode length:** 100 time slots
- **UAV altitude:** 60 m
- **UAV service range:** 300 m

### Mobility and Tasks

- AGVs move dynamically with random direction perturbation and boundary reflection
- Each AGV generates tasks at every time slot
- Small tasks can be computed locally
- Larger tasks are offloaded to the associated UAV when feasible

### Delay Model

The environment considers the following delay components:

- uplink transmission delay
- UAV computation delay
- downlink result transmission delay
- inter-UAV migration related delay when mobility causes service handover

---

## Core Algorithm

The proposed framework is based on **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** with **hybrid action modeling**.

Each UAV is treated as an independent agent that makes decisions according to its local observation.

### Action Space

Each UAV outputs:

- **Continuous actions:** UAV movement control in 2D space
- **Discrete actions:** AGV preference / selection scores for task service decisions

### Key Mechanisms

- **Actor network** outputs both continuous and discrete action branches
- **Gumbel-Softmax** is used during training to enable differentiable discrete policy learning
- **Centralized critic / decentralized actors** support cooperative multi-agent learning
- **Prioritized experience replay** improves sampling efficiency
- **Target networks** improve optimization stability

---

## Dependencies

Recommended environment:

- **Python 3.7 or 3.9**
- **PyTorch 1.12+**
- CPU execution is supported
- GPU is optional for faster training

### requirements.txt

```txt
numpy>=1.21,<2.0
scipy>=1.7
matplotlib>=3.5
torch>=1.12
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training

```bash
python train.py
```

---

## Training Outputs

During training, the code will automatically:

- create the `models/` directory for checkpoint saving
- create the `results/` directory for exported curves and arrays
- save intermediate model checkpoints every 100 episodes
- save reward data as:
  - `results/all_rewards.npy`
  - `results/smoothed_rewards.npy`
- generate the final convergence plot:
  - `results/reward.png`

---

## Example Workflow

1. Initialize the MEC environment with multiple UAVs and mobile AGVs
2. Collect observations for each UAV agent
3. Generate hybrid actions through actor networks
4. Execute actions in the environment
5. Compute rewards based on service quality and delay-aware objectives
6. Store experiences in the prioritized replay buffer
7. Update actor-critic networks iteratively
8. Save checkpoints and visualize training convergence

---

## Research Scope

This repository mainly focuses on the following research topics:

- multi-UAV collaborative mobile edge computing
- trajectory planning for aerial edge servers
- task offloading in mobile air-ground integrated systems
- hybrid-action multi-agent reinforcement learning
- delay-aware resource scheduling in dynamic MEC environments

---

## Citation

If you use this code in your research, please cite the corresponding paper or repository.


## Notes

- This repository is primarily intended for **academic research and experimental reproduction**.
- Before public release, you may further refine:
  - author information
  - paper title
  - repository link
  - license file
  - pretrained model release

---

## License

This project is released for **academic use only** unless otherwise specified.


---

## Acknowledgement

We acknowledge the open-source deep reinforcement learning community for providing useful references and implementation inspiration for multi-agent learning and continuous control.
