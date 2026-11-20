# Exploration-Induced Cooperation Collapse in Parameter-Shared Deep Q-Networks

This repository contains the experimental, analysis, and plotting code for the paper:

**Exploration-Induced Cooperation Collapse in Parameter-Shared Deep Q-Networks**  
Yi-Ning Weng and Hsuan-Wei Lee (2026)

## Overview

We investigate exploration-induced cooperation collapse in parameter-shared Deep Q-Networks (DQNs) using a dynamic spatial Prisoner's Dilemma. The experiments examine how exploration strength, parameter-sharing scale, replay-buffer capacity, observation information, learned value representations, and interaction topology affect collective cooperation. Comparisons with independent DQN and tabular Q-learning, together with exhaustive greedy-policy classification, help characterize the role of parameter sharing in the emergence and collapse of cooperation.

The repository provides experimental and plotting scripts for all nine main-text figures:

- **Figure 1:** Cooperation landscape of shared DQN across exploration strength (_B_) and payoff harshness (_Dr_).
- **Figure 2:** Empirical cooperation-collapse boundaries for shared and grouped DQN.
- **Figure 3:** Effects of parameter-sharing scale and replay-buffer capacity on cooperation.
- **Figure 4:** Independent-learner and tabular controls, including seed-paired comparisons among DQN architectures.
- **Figure 5:** State-augmentation experiments comparing baseline, exploration-temperature, annealing-progress, and combined representations.
- **Figure 6:** Hidden-representation analysis using UMAP and silhouette scores.
- **Figure 7:** Action-value diagnostics, including Q-mean and Q-gap.
- **Figure 8:** All-C policy fraction across trained checkpoints and cooperation under pure greedy evaluation.
- **Figure 9:** Cooperation across grid, modular 4-regular, random 4-regular, and rewired 4-regular interaction topologies.

## Requirements

The code uses Python 3 and the following packages: NumPy, PyTorch, Matplotlib, pandas, SciPy, scikit-learn, and umap-learn.

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

SciPy is installed as a dependency of scikit-learn and is used directly in the statistical analysis scripts. A CUDA-capable GPU can be used where supported by the experimental scripts, but CPU execution is also supported. Full experimental sweeps are computationally intensive.

## Experiments

The `experiments/` directory contains the simulation and analysis scripts, and `plotting/` contains the corresponding figure-generation scripts. Run commands from the repository root. Simulation parameters and random seeds are specified within each experimental script.

Unless otherwise specified, experiments use a 30 × 30 periodic grid (900 agents), with 95,000 training steps followed by 5,000 evaluation steps with learning disabled. The standard evaluation temperature is 0.1; Figure 8 additionally examines pure greedy evaluation. Figure 4 uses a 10 × 10 grid, and Figure 5 uses a 50 × 50 grid.

### Figures 1–3: Shared and grouped DQN

Figure 1 uses `fig1_experiment -  shared DQN.py` to generate the cooperation landscape. Figure 2 uses `fig2_experiment - grouped DQN.py` and the corresponding shared-DQN results from Figure 1.

Figure 3 compares the following configurations at Dr = 0.25:

| Configuration | Networks | Agents per network | Replay buffer per network |
| ------------- | -------: | -----------------: | ------------------------: |
| Shared DQN    |        1 |                900 |                    90,000 |
| Grouped DQN   |       10 |                 90 |                     9,000 |
| Grouped DQN   |      100 |                  9 |                     9,000 |
| Grouped DQN   |      100 |                  9 |                       900 |

The shared-DQN configuration can reuse Figure 1 results for the corresponding payoff and exploration settings. For the grouped configurations, run `fig3_experiment - grouped DQN.py` separately with `NUM_GROUPS` and `REPLAY_SIZE` set to the values in the table. Figure 3 uses seeds 195–209 for shared DQN and 195–199 for grouped DQN.

### Figure 4: Independent-learner and tabular controls

Figure 4 compares four learning configurations on a 10 × 10 periodic grid at Dr = 0.25, using seeds 195–209 and six exploration settings.

| Configuration      | Networks or Q-tables | Agents per learner | Replay buffer per network |
| ------------------ | -------------------: | -----------------: | ------------------------: |
| Shared DQN         |                    1 |                100 |                    90,000 |
| Grouped DQN        |                   10 |                 10 |                     9,000 |
| Independent DQN    |                  100 |                  1 |                       900 |
| Tabular Q-learning |                  100 |                  1 |            Not applicable |

The three DQN configurations have the same aggregate replay capacity of 90,000 transitions. The tabular baseline uses independent, zero-initialized Q-tables and online one-step Q-learning without experience replay or a target network.

- **Shared DQN:** Reuse the Figure 1 experiment with `SIZE = 10`, `Dr = 0.25`, and the six initial temperatures listed below. Retain the replay capacity of 90,000 and seeds 195–209.
- **Grouped DQN:** Reuse the Figure 3 experiment with `SIZE = 10`, `NUM_GROUPS = 10`, `REPLAY_SIZE = 9000`, `Dr = 0.25`, and change the seed range to `range(195, 210)`.
- **Independent DQN:** Run `fig4_experiment - independent DQN.py`.
- **Tabular Q-learning:** Run `fig4_experiment - tabular Q-learning.py`.

For the reduced-scale shared-DQN run, set `dr_values = [0.25]` and `tau_init_values = [0.127, 0.260, 0.393, 0.527, 0.660, 0.793]` in the sweep. Save the four experiment logs separately using the filenames listed below. The plotting script computes cooperation means and seed-paired mean differences with 95% confidence intervals based on Student's t distribution.

### Figure 5: State augmentation

Run `fig5_experiment - state size augmentation.py` separately for each of the four `STATE_MODE` settings:

- `baseline`: 5-dimensional state.
- `tau`: 6-dimensional state with normalized exploration temperature.
- `progress`: 6-dimensional state with annealing progress.
- `both`: 7-dimensional state with both additional features.

All four configurations use a 50 × 50 grid and a replay capacity of 250,000 transitions. Save the printed results separately for each state mode.

### Figure 6: Hidden representations

Run `fig6_experiment - checkpoints and states.py` to generate trained shared-DQN checkpoints and sampled evaluation states in `checkpoints_shared/` and `states_95k_100k/`, respectively. Run `plotting/fig6_plot.py` for UMAP visualizations and clustering-based silhouette scores.

The script `fig8_experiment - policy classification.py` also uses these checkpoints and sampled states to perform exhaustive 32-state analysis, including action-label silhouette scores and linear-probe ROC-AUC where defined.

### Figure 7: Action-value diagnostics

Run `fig7_experiment - Qmean.py` and `fig7_experiment - Qgap.py` to generate evaluation-stage Q-mean and Q-gap statistics. Both scripts save individual scalar result files to `results/`. Copy the `valuelevel_eval_*.txt` files into `Figure 5 Q mean/` and the `final_qgap_*.txt` files into `Figure 5 Q gap/`, as expected by the current plotting script. These directory names are retained for compatibility with the existing code.

### Figure 8: Greedy policy classification and evaluation

Figure 8 compares the fraction of checkpoints that select cooperation in all 32 possible binary states with mean cooperation under pure greedy evaluation. The analysis uses 90 single-hidden-layer shared-DQN checkpoints (six exploration settings × 15 seeds).

First generate the Figure 6 checkpoints and sampled states if they are not already available. Then run:

```bash
mkdir -p figure8
python "experiments/fig8_experiment - policy classification.py"
python "experiments/fig8_experiment - greedy evaluation.py" > figure8/greedy_DQN.txt
python "experiments/fig8_experiment - policy validation.py"
python plotting/fig8_plot.py
```

The classification script evaluates all 32 states and writes `figure8/greedy_32state_checkpoints.csv`. The greedy-evaluation script trains shared DQNs with the specified exploration schedules and evaluates them using greedy action selection without further learning. The validation script matches results by exploration strength and seed, checks policy classifications against greedy cooperation outcomes, and generates `figure8/greedy_policy_validation_summary.csv`. The plotting script reads this summary and computes pointwise Wilson 95% confidence intervals for the All-C fraction; the mean greedy-cooperation curve is shown without a confidence band.

### Figure 9: Interaction topology

Run the following scripts separately and save their printed results to different files:

- `fig9_experiment - modular 4 regular.py`
- `fig9_experiment - random 4 regular.py`
- `fig9_experiment - rewired 4 regular.py`

The grid configuration can reuse the 30 × 30 shared-DQN results from Figure 1 at Dr = 0.25 and the corresponding six exploration settings. The topology experiment scripts use the same time-series output naming convention as the grid experiment; preserve or move the files in `results/` between runs if individual cooperation trajectories are needed, to avoid overwriting them. The plotting script uses the separately saved experiment logs.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@unpublished{weng2026exploration,
  title  = {Exploration-Induced Cooperation Collapse in Parameter-Shared Deep Q-Networks},
  author = {Weng, Yi-Ning and Lee, Hsuan-Wei},
  year   = {2026}
}
```

## License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for details.
