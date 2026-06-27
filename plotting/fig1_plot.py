from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

TXT_PATH = Path("figure1.txt")

LINE_RE = re.compile(
    r"Dr:\s*([0-9.]+).*?"
    r"SEED:\s*(\d+).*?"
    r"B:\s*([0-9.]+).*?"
    r"result:\s*([0-9.]+)",
    re.IGNORECASE
)

rows = []
with TXT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        m = LINE_RE.search(line)
        if not m:
            continue
        dr   = float(m.group(1))
        seed = int(m.group(2))
        b    = float(m.group(3))
        res  = float(m.group(4))
        rows.append((b, dr, seed, res))

by_seed = defaultdict(list)
for b, dr, seed, res in rows:
    by_seed[(b, dr, seed)].append(res)

seed_avg = {k: np.mean(v) for k, v in by_seed.items()}

cell_vals = defaultdict(list)
for (b, dr, seed), val in seed_avg.items():
    cell_vals[(b, dr)].append(val)
cell_mean = {k: np.mean(v) for k, v in cell_vals.items()}

AUTO_RANGE = True
B_MIN, B_MAX = 0.10, 0.65
EPS = 1e-9

all_Bs  = sorted({b for (b, _) in cell_mean.keys()})
all_Drs = sorted({dr for (_, dr) in cell_mean.keys()})

if AUTO_RANGE:
    b_in_range  = all_Bs
    all_dr      = all_Drs
else:
    b_in_range = sorted([b for b in all_Bs if (B_MIN - EPS) <= b <= (B_MAX + EPS)])
    all_dr = all_Drs

M = np.full((len(b_in_range), len(all_dr)), np.nan)
for i, b in enumerate(b_in_range):
    for j, dr in enumerate(all_dr):
        M[i, j] = cell_mean.get((b, dr), np.nan)

cmap = plt.cm.viridis.copy()
cmap.set_bad(color='white')  
M_masked = np.ma.masked_invalid(M)

fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(
    M_masked,
    origin='lower',
    interpolation='none',
    aspect=1.6,
    vmin=0.0, vmax=1.0,
    cmap=cmap
)

ax.set_xlabel("Dr", fontsize=24, labelpad=12)
ax.set_ylabel("B", fontsize=24, labelpad=12)

visible_drs = [d for d in all_dr if abs((d*100) % 5) < 1e-9] 
ax.set_xticks([all_dr.index(d) for d in visible_drs])
ax.set_xticklabels([f"{d:.2f}" for d in visible_drs], ha='right')
ax.set_yticks(range(len(b_in_range)))
ax.set_yticklabels([f"{b:.2f}" for b in b_in_range])
ax.tick_params(axis='x', labelsize=24, pad=8)
ax.tick_params(axis='y', labelsize=24, pad=4)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, shrink=0.55)
cbar.set_label("Cooperation level", fontsize=24, labelpad=12)
cbar.ax.tick_params(labelsize=24)

start_x, start_y = 4, 1.5
full_end_x = len(all_dr) - 0.55
full_end_y = len(b_in_range) - 0.55
half_end_x = start_x + 0.6 * (full_end_x - start_x)
half_end_y = start_y + 0.6 * (full_end_y - start_y)

ax.annotate(
    "", xy=(half_end_x, half_end_y), xytext=(start_x, start_y),
    arrowprops=dict(
        arrowstyle="->",
        color="white",
        lw=6.0,
        alpha=0.95,
        mutation_scale=40
    )
)

ax.text(
    0.5, 0.85,   
    "Increasing  B and Dr drive cooperation collapse",
    fontsize=24,
    color="white",
    ha="center",
    va="center",
    transform=ax.transAxes, 
)


plt.tight_layout()
fig.savefig("figure1_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig("figure1_heatmap.pdf", bbox_inches="tight")
plt.show()
