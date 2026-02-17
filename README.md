# How Exploration Breaks Cooperation in Shared-Policy Multi-Agent Reinforcement Learning

Official implementation of the paper:

> **How Exploration Breaks Cooperation in Shared-Policy Multi-Agent Reinforcement Learning**

This repository contains the PyTorch implementation of the shared-policy Deep Q-Network (DQN) model used to study exploration-induced cooperation collapse in multi-agent reinforcement learning under a spatial Prisoner's Dilemma setting.

---

## Overview

This project investigates how exploration intensity affects cooperation stability in shared-policy multi-agent reinforcement learning (MARL).

We implement:

- Shared-policy DQN
- Double DQN
- 5-step return
- Boltzmann (softmax) exploration with temperature annealing
- Target network updates
- Gradient clipping
- Evaluation phase without learning
- Periodic 2D grid (von Neumann neighborhood)

The environment consists of agents placed on a 2D periodic lattice interacting through the Prisoner's Dilemma game.

---

## Model Details

### Environment

- Grid topology (periodic boundary)
- 4-neighbor von Neumann interaction
- Prisoner's Dilemma payoff:
  - R = 1
  - P = 0
  - S = -Dr
  - T = 1 + Dr

### State Representation

Each agent observes a 5-dimensional binary state:

- 4 neighbor actions (previous step)
- 1 self action (previous step)

### Network Architecture

- 1 hidden layer
- Hidden size: 96
- Activation: ReLU
- Output: Q-values for 2 actions (C/D)

### Optimization

- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Loss: SmoothL1Loss (Huber)
- Gradient clipping: max_norm = 0.5
- Replay buffer: 90,000
- Batch size: 256
- n-step return: 5

### Target Network

- Hard update every 2000 steps

### Exploration

Boltzmann exploration:

\[
\pi(a|s) \propto \exp(Q(s,a)/\tau)
\]

- Initial τ = 1.0
- Final τ = 0.10
- Linear annealing during training phase
- Evaluation τ = 0.10

Exploration intensity metric:

\[
B = \text{mean}(\tau) \text{ over the first half of training}
\]

---

## Training Protocol

- Total steps: 100,000
- Training phase: 95,000 steps
- Evaluation phase: 5,000 steps (learning disabled)
- Cooperation rate measured during evaluation phase

---

## Reproducing Experiments

### Seed Configuration

Paper experiments use:

`seed_values = list(range(195, 225))` # 30 seeds


### Dr Sweep

Typical sweep range:

Dr ∈ [0.10, 0.40]


Modify `dr_values` in the script to reproduce full experiment curves.

### Running

```bash
pip install -r requirements.txt
```

The script will:

- Train models for each (Dr, seed)

- Output cooperation rates

- Save aggregated results

## Requirements

Create environment:
```bash
pip install -r requirements.txt
```
```txt
requirements.txt
torch>=2.0
numpy>=1.23
```

GPU is recommended but not required.

## Citation

If you use this code, please cite:

```bibtex
@article{weng2026exploration,
  title   = {How Exploration Breaks Cooperation in Shared-Policy Multi-Agent Reinforcement Learning},
  author  = {Yi-Ning Weng and Hsuan-Wei Lee},
  journal = {Under review},
  year    = {2026}
}
```


