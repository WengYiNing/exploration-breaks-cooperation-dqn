import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("figure3")

INPUT_FILES = {
    "τ + Anneal (7-dim state)": BASE_DIR / "state size=7, tau+anneal.txt",
    "τ (6-dim state)": BASE_DIR / "state size=6, tau.txt",
    "Anneal (6-dim state)": BASE_DIR / "state size=6, anneal.txt",
    "Baseline": BASE_DIR / "baseline.txt"
}

marker_map = {
    "Baseline": "o",
    "τ (6-dim state)": "D",
    "Anneal (6-dim state)": "^",
    "τ + Anneal (7-dim state)": "s",
}

pattern = re.compile(
    r"Dr:\s*([0-9.]+).*?"
    r"SIZE:\s*(\d+).*?"
    r"SEED:\s*(\d+).*?"
    r"B:\s*([0-9.]+).*?"
    r"result:\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE
)

fig, ax = plt.subplots(
    figsize=(14, 7),
    dpi=150
)

for label, file_path in INPUT_FILES.items():
    records = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)

            if m:
                dr = float(m.group(1))
                size = int(m.group(2))
                seed = int(m.group(3))
                b = round(float(m.group(4)), 2)
                result = float(m.group(5))

                if abs(dr - 0.25) > 1e-6 or size != 50:
                    continue

                records.append({
                    "B": b,
                    "Dr": dr,
                    "SEED": seed,
                    "result": result
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"No matching data found: {file_path}")

    df = (
        df.groupby(
            ["B", "Dr", "SEED"],
            as_index=False
        )["result"].mean()
    )

    print(f"\n========== {label} Data Check ==========")
    print(f"Total unique seeds: {df['SEED'].nunique()}")
    print(f"Total Dr values: {df['Dr'].nunique()}")
    print(f"Total B values: {df['B'].nunique()}")

    print("\nSeed counts by B:")
    print(
        df.groupby("B")["SEED"]
        .nunique()
        .to_string()
    )

    agg = (
        df.groupby(
            "B",
            as_index=False
        )
        .agg(
            coop_mean=("result", "mean"),
            coop_std=("result", "std"),
            n=("result", "count")
        )
        .sort_values("B")
    )

    agg["coop_ci95"] = (
        1.96
        * agg["coop_std"]
        / (agg["n"] ** 0.5)
    )

    lower = (
        agg["coop_mean"]
        - agg["coop_ci95"]
    ).clip(lower=0.0)

    upper = (
        agg["coop_mean"]
        + agg["coop_ci95"]
    ).clip(upper=1.0)

    line, = ax.plot(
        agg["B"],
        agg["coop_mean"],
        marker=marker_map[label],
        linewidth=7,
        markersize=16,
        label=label,
        zorder=3,
    )

    ax.fill_between(
        agg["B"],
        lower,
        upper,
        alpha=0.05,
        color=line.get_color(),
        zorder=1
    )

    print(f"\n[{label}] n per B:")
    print(
        agg[
            ["B", "n"]
        ].to_string(index=False)
    )

    print(
        f"[{label}] "
        f"avg n across B = {agg['n'].mean():.2f}, "
        f"min n = {agg['n'].min()}, "
        f"max n = {agg['n'].max()}"
    )

ax.set_xlabel(
    "B",
    fontsize=26,
    labelpad=14,
)

ax.set_ylabel(
    "Cooperation Level",
    fontsize=26,
    labelpad=18,
)

ax.tick_params(
    axis="both",
    labelsize=26
)

ax.legend(
    fontsize=22,
    loc="upper left",
    bbox_to_anchor=(1.02, 0.32),
    borderaxespad=0.0
)

plt.tight_layout()

plt.savefig(
    "figure4_state_augmentation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "figure4_state_augmentation.pdf",
    bbox_inches="tight"
)

plt.show()