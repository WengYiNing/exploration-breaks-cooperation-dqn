# Exploration-Induced Cooperation Collapse in Deep Reinforcement Learning Driven Spatial Dilemmas

This repository contains the source code used to generate the main-text figures for the paper:

**Exploration-Induced Cooperation Collapse in Deep Reinforcement Learning Driven Spatial Dilemmas**

## Overview

The code implements a shared-policy Deep Q-Network (DQN) framework for studying cooperation collapse in learning-driven spatial Prisoner's Dilemma environments.

The repository contains experimental and plotting scripts for the main-text figures:

- **Figure 1:** Cooperation landscape of the shared-policy DQN over exploration strength (B) and payoff harshness (D_r).
- **Figure 2:** Empirical collapse boundaries for shared-policy DQN and grouped DQN.
- **Figure 3:** State-augmentation experiments examining the role of temporal and exploration-related observability.
- **Figure 4:** Hidden-representation diagnostics based on learned DQN activations.
- **Figure 5:** Action-value diagnostics, including average Q-value level and Q-gap.
- **Figure 6:** Topology comparison across grid, modular 4-regular, random 4-regular, and rewired 4-regular networks.

The `experiments/` directory contains scripts for running the simulations, and the `plotting/` directory contains scripts for generating the corresponding figures from the simulation outputs.

## Requirements

The code was developed with Python 3 and uses the following main packages:

```txt
numpy
torch
matplotlib
scikit-learn
umap-learn
networkx
```

The required packages can be installed with:

```bash
pip install -r requirements.txt
```

## Plotting scripts and input files

The plotting scripts assume that the raw simulation outputs have been collected into figure-specific text files. These text files are generated from the corresponding experiment scripts by saving or redirecting the printed simulation outputs.

For example, a typical workflow is:

```bash
python experiments/fig1_experiment_shared_dqn.py > figure1.txt
python plotting/fig1_plot.py
```

Each plotting script expects its input file to follow the output format produced by the corresponding experiment script. The expected input filename is specified near the top of each plotting script.

Because the full parameter sweeps require substantial computation, the repository focuses on providing the experiment and plotting code needed to reproduce the main-text figures. Users may rerun the scripts with the same parameter settings and random seeds used in the paper, or modify the parameter lists in the experiment scripts for smaller test runs.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@article{weng2026exploration,
  title   = {Exploration-Induced Cooperation Collapse in Deep Reinforcement Learning Driven Spatial Dilemmas},
  author  = {Weng, Yi-Ning and Lee, Hsuan-Wei},
  journal = {TBD},
  year    = {2026}
}
```
