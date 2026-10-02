import os
from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

QMEAN_ROOT = Path("Figure 5 Q mean")
QGAP_ROOT  = Path("Figure 5 Q gap")


pattern_B = re.compile(
    r".*_SIZE(?P<size>\d+)"
    r"_SEED(?P<seed>\d+)"
    r"_Dr(?P<dr>[\d.]+)"
    r"_tauinit_[\d.]+"
    r"_taufinal_(?P<tau_final>[\d.]+)"
    r"_anneal_(?P<anneal>\d+)"
    r"_B_(?P<b>[\d.]+)\.txt"
)

def load_statistics(root, label):
    by_B = defaultdict(dict)
    used_data = set()

    for file in root.iterdir():
        if not file.is_file() or file.suffix != ".txt":
            continue

        m = pattern_B.fullmatch(file.name)
        if not m:
            continue

        size = int(m.group("size"))
        seed = int(m.group("seed"))
        dr = float(m.group("dr"))
        b = float(m.group("b"))
        tau_final = float(m.group("tau_final"))
        anneal = int(m.group("anneal"))

        if (size != 30 or abs(dr - 0.25) > 1e-8
                or abs(tau_final - 0.1) > 1e-8
                or anneal != 95000):
            continue

        try:
            val = float(file.read_text().strip())
        except Exception as e:
            print(f"Warning: skipped {file}: {e}")
            continue

        if seed in by_B[b]:
            if not np.isclose(by_B[b][seed], val, rtol=0, atol=1e-6):
                raise ValueError(
                    f"Conflicting results: B={b}, seed={seed}"
                )
            continue

        by_B[b][seed] = val
        used_data.add((seed, dr, b))

    if not used_data:
        raise ValueError(f"No matching data found in {root}")

    print(f"\n========== {label} Data Check ==========")
    print(f"Total unique seeds: {len({s for s, d, b in used_data})}")
    print(f"Total Dr values: {len({d for s, d, b in used_data})}")
    print(f"Total B values: {len({b for s, d, b in used_data})}")

    for b in sorted(by_B):
        print(f"B={b:.6f}: {len(by_B[b])} seeds")

    sorted_B = sorted(by_B)
    means = [np.mean(list(by_B[b].values())) for b in sorted_B]
    stds = [
        np.std(list(by_B[b].values()), ddof=1)
        for b in sorted_B
    ]
    ci95 = [
        1.96 * std / np.sqrt(len(by_B[b]))
        for b, std in zip(sorted_B, stds)
    ]

    return sorted_B, means, stds, ci95


sorted_B_qmean, mean_qmean, std_qmean, ci95_qmean = (
    load_statistics(QMEAN_ROOT, "Q-mean")
)

sorted_B_qgap, mean_qgap, std_qgap, ci95_qgap = (
    load_statistics(QGAP_ROOT, "Q-gap")
)


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
