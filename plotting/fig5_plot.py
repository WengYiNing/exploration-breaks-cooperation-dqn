import os
from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

QMEAN_ROOT = Path("Figure 5 Q mean")
QGAP_ROOT  = Path("Figure 5 Q gap")

pattern_B = re.compile(r"SEED(\d+).*_B_([\d\.]+)\.txt")

qmean_by_B = defaultdict(list)

for file in QMEAN_ROOT.iterdir():
    if not file.is_file() or file.suffix != ".txt":
        continue

    m = pattern_B.search(file.name)
    if not m:
        continue

    B = float(m.group(2))

    try:
        with open(file, "r") as f:
            val = float(f.read().strip())
        qmean_by_B[B].append(val)
    except Exception as e:
       print(f"Warning: skipped {file}: {e}")

sorted_B_qmean = sorted(qmean_by_B.keys())
mean_qmean = [np.mean(qmean_by_B[B]) for B in sorted_B_qmean]
std_qmean  = [np.std(qmean_by_B[B], ddof=1) for B in sorted_B_qmean]
ci95_qmean = [1.96 * np.std(qmean_by_B[B], ddof=1) / np.sqrt(len(qmean_by_B[B])) for B in sorted_B_qmean]

qgap_by_B = defaultdict(list)

for file in QGAP_ROOT.iterdir():
    if not file.is_file() or file.suffix != ".txt":
        continue

    m = pattern_B.search(file.name)
    if not m:
        continue

    B = float(m.group(2))

    try:
        with open(file, "r") as f:
            val = float(f.read().strip())
        qgap_by_B[B].append(val)
    except Exception as e:
        print(f"Warning: skipped {file}: {e}")

sorted_B_qgap = sorted(qgap_by_B.keys())
mean_qgap = [np.mean(qgap_by_B[B]) for B in sorted_B_qgap]
std_qgap  = [np.std(qgap_by_B[B], ddof=1) for B in sorted_B_qgap]
ci95_qgap = [1.96 * np.std(qgap_by_B[B], ddof=1) / np.sqrt(len(qgap_by_B[B])) for B in sorted_B_qgap]

plt.figure(figsize=(12, 5))

ax1 = plt.subplot(1, 2, 1)
ax1.plot(
    sorted_B_qmean, mean_qmean,
    marker="o", linewidth=5, color="tab:blue", markersize=11
)
ax1.fill_between(
    sorted_B_qmean,
    np.array(mean_qmean) - np.array(ci95_qmean),
    np.array(mean_qmean) + np.array(ci95_qmean),
    color="tab:blue",
    alpha=0.08
)
ax1.set_ylim(0, 55)
ax1.set_xlabel("B", fontsize=24)
ax1.set_ylabel("Average Q-mean", fontsize=24, labelpad=10)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(
    sorted_B_qgap, mean_qgap,
    marker="o", linewidth=5, color="tab:orange", markersize=11
)
ax2.fill_between(
    sorted_B_qgap,
    np.array(mean_qgap) - np.array(ci95_qgap),
    np.array(mean_qgap) + np.array(ci95_qgap),
    color="tab:orange",
    alpha=0.08
)
ax2.set_ylim(0, 75)
ax2.set_xlabel("B", fontsize=24)
ax2.set_ylabel("Average Q-gap", fontsize=24, labelpad=10)

ax1.tick_params(axis='both', labelsize=24)  
ax2.tick_params(axis='both', labelsize=24)   

ax1.text(
    0.02, 1.1, "(a)",
    transform=ax1.transAxes,
    fontsize=24,
    va="top",
    ha="left"
)

ax2.text(
    0.02, 1.1, "(b)",
    transform=ax2.transAxes,
    fontsize=24,
    va="top",
    ha="left"
)

plt.tight_layout()
plt.savefig("figure5_q_statistics.png", dpi=300, bbox_inches="tight")
plt.savefig("figure5_q_statistics.pdf", bbox_inches="tight")
plt.show()
