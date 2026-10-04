# -*- coding: utf-8 -*-

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("figure3")

paths = [
    ("100 Groups, Buffer = 900", DATA_DIR / "100groups_buffer900.txt"),
    ("100 Groups, Buffer = 9000", DATA_DIR / "100groups_buffer9000.txt"),
    ("10 Groups, Buffer = 9000", DATA_DIR / "10groups_buffer9000.txt"),
    ("Shared DQN, Buffer = 90000", DATA_DIR / "shared_buffer90000.txt"),
]

TARGET_DR = 0.25

GROUPED_SEEDS = set(range(195, 200))  # 5 seeds
SHARED_SEEDS = set(range(195, 210))   # 15 seeds

SHARED_LABEL = "Shared DQN, Buffer = 90000"

TARGET_B = [
    0.120250,
    0.219999,
    0.319748,
    0.420248,
    0.519997,
    0.619746,
]

def parse_file(path: Path, label: str) -> pd.DataFrame:
    pattern = re.compile(
        r"Dr:\s*([0-9]*\.?[0-9]+).*?"
        r"SEED:\s*(\d+).*?"
        r"B:\s*([0-9]*\.?[0-9]+).*?"
        r"result:\s*([0-9]*\.?[0-9]+)"
    )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            match = pattern.search(line)

            if match:

                Dr = float(match.group(1))
                seed = int(match.group(2))
                B = float(match.group(3))
                coop = float(match.group(4))

                if label != SHARED_LABEL:
                    expected_groups, expected_group_size, expected_buffer = {
                        "10 Groups, Buffer = 9000": (10, 90, 9000),
                        "100 Groups, Buffer = 9000": (100, 9, 9000),
                        "100 Groups, Buffer = 900": (100, 9, 900),
                    }[label]

                    expected_fields = {
                        "groups": expected_groups,
                        "group_size": expected_group_size,
                        "replay_size": expected_buffer,
                    }

                    matches_config = True

                    for field, expected in expected_fields.items():
                        field_match = re.search(rf"\b{field}:\s*(\d+)", line)

                        if field_match and int(field_match.group(1)) != expected:
                            matches_config = False
                            break

                    if not matches_config:
                        continue

                rows.append(
                    (
                        label,
                        Dr,
                        seed,
                        B,
                        coop,
                    )
                )

    if not rows:
        raise ValueError(
            f"No matched rows found in file: {path}"
        )

    return pd.DataFrame(
        rows,
        columns=[
            "configuration",
            "Dr",
            "seed",
            "B",
            "coop",
        ],
    )

dfs = []

for label, path in paths:

    df_temp = parse_file(
        path,
        label
    )

    print(
        f"{label}: "
        f"{len(df_temp)} total rows loaded"
    )

    dfs.append(
        df_temp
    )


df = pd.concat(
    dfs,
    ignore_index=True,
)
df = df[
    df["Dr"].round(6)
    == round(TARGET_DR, 6)
].copy()

df = df[
    (
        (df["configuration"] == SHARED_LABEL)
        & df["seed"].isin(SHARED_SEEDS)
    )
    |
    (
        (df["configuration"] != SHARED_LABEL)
        & df["seed"].isin(GROUPED_SEEDS)
    )
].copy()

target_B_rounded = {
    round(B, 6)
    for B in TARGET_B
}

df = df[
    df["B"].round(6).isin(
        target_B_rounded
    )
].copy()

print(
    "\n========================================"
)
print(
    "Available B values after filtering"
)
print(
    "========================================"
)

for configuration in df[
    "configuration"
].unique():

    sub = df[
        df["configuration"]
        == configuration
    ]

    print(
        f"\n{configuration}"
    )

    print(
        sorted(
            sub["B"].unique()
        )
    )

expected_configurations = {label for label, _ in paths}
actual_configurations = set(df["configuration"].unique())

if actual_configurations != expected_configurations:
    raise ValueError(
        f"Missing configurations: "
        f"{expected_configurations - actual_configurations}"
    )

Bs_by_configuration = {
    configuration: set(
        df[
            df["configuration"]
            == configuration
        ]["B"].round(6).unique()
    )
    for configuration
    in df["configuration"].unique()
}


common_B = sorted(
    set.intersection(
        *Bs_by_configuration.values()
    )
)


print(
    "\n========================================"
)
print(
    "Common B values"
)
print(
    "========================================"
)

print(
    common_B
)


df_common = df[
    df["B"].round(6).isin(
        common_B
    )
].copy()

df_common = (
    df_common.groupby(
        ["configuration", "Dr", "B", "seed"],
        as_index=False
    )["coop"].mean()
)

print("\n========== Figure 3 Data Check ==========")

for configuration, sub in df_common.groupby("configuration"):
    print(f"\n{configuration}")
    print(f"Total unique seeds: {sub['seed'].nunique()}")
    print(f"Total Dr values: {sub['Dr'].nunique()}")
    print(f"Total B values: {sub['B'].nunique()}")

seed_check = (
    df_common
    .groupby(
        [
            "configuration",
            "B",
        ]
    )["seed"]
    .nunique()
    .reset_index(
        name="n_seeds"
    )
)


print(
    "\n========================================"
)
print(
    "Seed count for every point"
)
print(
    "========================================"
)

print(
    seed_check.to_string(
        index=False
    )
)

agg = (
    df_common
    .groupby(
        [
            "configuration",
            "B",
        ]
    )["coop"]
    .agg(
        mean="mean",
        std="std",
        n="count",
    )
    .reset_index()
)

agg["ci95"] = (
    1.96
    * agg["std"]
    / (agg["n"] ** 0.5)
)

print(
    "\n========================================"
)
print(
    "Aggregated results"
)
print(
    "========================================"
)

print(
    agg.to_string(
        index=False
    )
)

plot_order = [
    "Shared DQN, Buffer = 90000",
    "10 Groups, Buffer = 9000",
    "100 Groups, Buffer = 9000",
    "100 Groups, Buffer = 900",
]

grouped_plot_order = [
    "10 Groups, Buffer = 9000",
    "100 Groups, Buffer = 9000",
    "100 Groups, Buffer = 900",
]

marker_map = {
    "Shared DQN, Buffer = 90000": "o",
    "10 Groups, Buffer = 9000": "D",
    "100 Groups, Buffer = 9000": "^",
    "100 Groups, Buffer = 900": "s",
}

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 7),
)

ax_full = axes[0]
ax_grouped = axes[1]

for configuration in plot_order:

    sub = (
        agg[
            agg["configuration"]
            == configuration
        ]
        .sort_values("B")
    )

    if sub.empty:
        print(
            f"Warning: no data found for "
            f"{configuration}"
        )
        continue

    line, = ax_full.plot(
        sub["B"],
        sub["mean"],
        marker=marker_map[configuration],
        label=configuration,
        linewidth=8,
        markersize=15,
    )

    ax_full.fill_between(
        sub["B"],
        sub["mean"] - sub["ci95"],
        sub["mean"] + sub["ci95"],
        alpha=0.08,
        color=line.get_color(),
    )

for configuration in grouped_plot_order:

    sub = (
        agg[
            agg["configuration"]
            == configuration
        ]
        .sort_values("B")
    )

    if sub.empty:
        print(
            f"Warning: no data found for "
            f"{configuration}"
        )
        continue

    line, = ax_grouped.plot(
        sub["B"],
        sub["mean"],
        marker=marker_map[configuration],
        label=configuration,
        linewidth=8,
        markersize=15,
    )

    ax_grouped.fill_between(
        sub["B"],
        sub["mean"] - sub["ci95"],
        sub["mean"] + sub["ci95"],
        alpha=0.08,
        color=line.get_color(),
    )

ax_full.set_xlabel(
    "B",
    fontsize=28,
    labelpad=14,
)

ax_full.set_ylabel(
    "Cooperation Level",
    fontsize=28,
    labelpad=12,
)

ax_full.tick_params(
    axis="both",
    labelsize=26,
)

ax_full.set_ylim(
    0,
    0.9,
)

ax_full.text(
    0.02,
    1.07,
    "(a)",
    transform=ax_full.transAxes,
    fontsize=26,
    va="top",
    ha="left",
)

ax_grouped.set_xlabel(
    "B",
    fontsize=28,
    labelpad=14,
)

ax_grouped.set_ylabel(
    "Cooperation Level",
    fontsize=28,
    labelpad=12,
)

ax_grouped.tick_params(
    axis="both",
    labelsize=26,
)

ax_grouped.set_ylim(
    0,
    0.4,
)

ax_grouped.text(
    0.02,
    1.07,
    "(b)",
    transform=ax_grouped.transAxes,
    fontsize=26,
    va="top",
    ha="left",
)

ax_full.legend(
    fontsize=18,
    loc="upper right",
)

ax_grouped.legend(
    fontsize=18,
    loc="upper right",
)

plt.tight_layout()

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "figure3_group_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "figure3_group_comparison.pdf",
    bbox_inches="tight"
)

plt.show()