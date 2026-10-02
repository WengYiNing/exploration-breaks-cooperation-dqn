import os
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

states_dir = "states_95k_100k"
ckpt_dir   = "checkpoints_shared"


class Net(nn.Module):
    def __init__(self, state_dim=5, hidden_dim=96, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward_hidden(self, x):
        h = self.relu(self.fc1(x))
        return h

    def forward(self, x):
        h = self.forward_hidden(x)
        q = self.fc2(h)
        return q


pattern = re.compile(
    r"state_Size(?P<size>\d+)_SEED(?P<seed>\d+)_Dr(?P<dr>[\d\.]+)_tauinit_(?P<tauinit>[\d\.]+)_B_(?P<B>[\d\.]+)\.npy"
)

state_files = sorted(os.listdir(states_dir))

all_hidden = []
all_B = []
all_labels = []
all_Q = []

for fname in state_files:
    m = pattern.match(fname)
    if m is None:
        continue

    seed    = int(m.group("seed"))
    dr_str  = m.group("dr")
    tau_str = m.group("tauinit")
    B_str   = m.group("B")
    Bv      = float(B_str)

    if abs(float(dr_str) - 0.25) > 1e-8:
        continue

    S = np.load(os.path.join(states_dir, fname)).astype(np.float32)
    S_t = torch.tensor(S, dtype=torch.float32)

    ckpt_name = (
        f"sharedDQN_SIZE30_SEED{seed}_Dr{dr_str}_tauinit_{tau_str}"
        f"_taufinal_0.1_anneal_95000_B_{B_str}.pt"
    )
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = Net(state_dim=S.shape[1], hidden_dim=96, action_dim=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        H = model.forward_hidden(S_t).numpy()
        Q = model(S_t).numpy()

    actions = np.argmax(Q, axis=1)

    all_hidden.append(H)
    all_Q.append(Q)
    all_B.extend([Bv] * len(H))
    all_labels.extend(actions.tolist())


all_hidden = np.vstack(all_hidden)
all_Q = np.vstack(all_Q)
all_B = np.array(all_B)
all_labels = np.array(all_labels)

mean = all_hidden.mean(axis=0)
std = all_hidden.std(axis=0) + 1e-6
Z_in = (all_hidden - mean) / std

unique_B_all = sorted({float(b) for b in all_B})
sil_by_B = {}

for Bv in unique_B_all:
    idx = np.abs(all_B - Bv) < 1e-8
    X_B = Z_in[idx]

    if X_B.shape[0] < 3:
        sil = float("nan")
    else:
        km = KMeans(n_clusters=2, random_state=0, n_init=10)
        clusters = km.fit_predict(X_B)

        if len(np.unique(clusters)) < 2:
            sil = float("nan")
        else:
            sil = float(silhouette_score(X_B, clusters))

    sil_by_B[Bv] = sil

N = Z_in.shape[0]
max_points = 3000

np.random.seed(0)
if N > max_points:
    sub_idx = np.random.choice(N, max_points, replace=False)
else:
    sub_idx = np.arange(N)

Z_in_sub   = Z_in[sub_idx]
B_sub      = all_B[sub_idx]
labels_sub = all_labels[sub_idx]

umap = UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    n_components=2,
    random_state=0,
)

Z = umap.fit_transform(Z_in_sub)

unique_B = sorted(set(B_sub))

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 8),
    sharex="col",
    sharey="row"
)

axes = axes.flatten()

x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
x_max = x_max + 10

y_min, y_max = Z[:, 1].min(), Z[:, 1].max()
y_min = y_min - 20

pad_x = 0.05 * (x_max - x_min)
pad_y = 0.05 * (y_max - y_min)

x_min, x_max = x_min - pad_x, x_max + pad_x
y_min, y_max = y_min - pad_y, y_max + pad_y

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for i, (Bv, plabel) in enumerate(zip(unique_B, panel_labels)):
    ax = axes[i]

    idx_B = B_sub == Bv
    idx_C = idx_B & (labels_sub == 0)
    idx_D = idx_B & (labels_sub == 1)

    if i == 0:
        ax.scatter(
            Z[idx_C, 0],
            Z[idx_C, 1],
            s=125,
            alpha=0.6,
            label="Cooperate"
        )
        ax.scatter(
            Z[idx_D, 0],
            Z[idx_D, 1],
            s=125,
            alpha=0.6,
            label="Defect"
        )
    else:
        ax.scatter(
            Z[idx_C, 0],
            Z[idx_C, 1],
            s=125,
            alpha=0.6
        )
        ax.scatter(
            Z[idx_D, 0],
            Z[idx_D, 1],
            s=125,
            alpha=0.6
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title(
        plabel,
        loc="left",
        x=0.0,
        y=1.02,
        fontsize=28
    )

    sil = sil_by_B[float(Bv)]
    sil_label = f"B = {Bv:.3f}\nsilhouette = {sil:.3f}"

    ax.text(
        0.96,
        0.04,
        sil_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=22,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.75,
            edgecolor="none"
        )
    )


for ax in axes[:3]:
    ax.tick_params(labelbottom=False)

for i, ax in enumerate(axes):
    if i % 3 != 0:
        ax.tick_params(labelleft=False)

for ax in axes:
    ax.tick_params(axis="both", labelsize=28)


handles, labels_ = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels_,
    loc="upper right",
    bbox_to_anchor=(1.12, 0.26),
    fontsize=28
)

plt.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
plt.savefig("figure4_hidden_umap.png", dpi=300, bbox_inches="tight")
plt.show()