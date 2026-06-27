import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


paths = [
    ("Rewired 4-regular", Path("rewired 4-regular.txt")),
    ("Grid", Path("grid.txt")),
    ("Random 4-regular", Path("random 4-regular.txt")),
    ("Modular 4-regular", Path("modular 4-regular.txt")),
]


def parse_file(path: Path, label: str) -> pd.DataFrame:
    pat = re.compile(
        r"SEED:\s*(\d+).*?B:\s*([0-9]*\.?[0-9]+).*?result:\s*([0-9]*\.?[0-9]+)"
    )
    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                seed = int(m.group(1))
                B = float(m.group(2))
                coop = float(m.group(3))
                rows.append((label, seed, B, coop))

    if not rows:
        raise ValueError(f"No matched rows found in file: {path}")

    return pd.DataFrame(rows, columns=["topology", "seed", "B", "coop"])


dfs = []
for label, path in paths:
    dfs.append(parse_file(path, label))

df = pd.concat(dfs, ignore_index=True)

Bs_by_topology = {
    topology: set(df[df["topology"] == topology]["B"].unique())
    for topology in df["topology"].unique()
}

common_B = sorted(set.intersection(*Bs_by_topology.values()))

df_common = df[df["B"].isin(common_B)].copy()

agg = (
    df_common.groupby(["topology", "B"])["coop"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
)

agg["ci95"] = 1.96 * agg["std"] / (agg["n"] ** 0.5)

plt.figure(figsize=(14, 7))

for topology in agg["topology"].unique():
    sub = agg[agg["topology"] == topology].sort_values("B")
    line, = plt.plot(
        sub["B"],
        sub["mean"],
        marker="o",
        label=topology,
        linewidth=8,
        markersize=16,
    )

    plt.fill_between(
        sub["B"],
        sub["mean"] - sub["ci95"],
        sub["mean"] + sub["ci95"],
        alpha=0.05,
        color=line.get_color(),
    )

plt.xlabel("B", fontsize=32)
plt.ylabel("Cooperation Level", fontsize=32)

plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(fontsize=24, bbox_to_anchor=(1.00, -0.04), loc="lower left")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure6_topology_comparison.png", dpi=300)
plt.savefig(OUTPUT_DIR / "figure6_topology_comparison.pdf")
plt.show()