"""Plot Figure 8: All-C policy fraction and greedy evaluation mean.

Run the policy validation script first to generate the summary CSV.
Run this script from the repository root.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("figure8")
SUMMARY_CSV = DATA_DIR / "greedy_policy_validation_summary.csv"
OUTPUT_DIR = Path("figures")
EXPECTED_SEEDS = set(range(195, 210))
EXPECTED_B = np.array([0.120250, 0.219999, 0.319748, 0.420248, 0.519997, 0.619746])
FIGSIZE = (6.4, 4.2)
LINE_COLOR = "#1f77b4"
OUTPUT_STEM = "Figure8_AllC_fraction_Wilson95"


def wilson_interval(k, n, z=1.959963984540054):
    """Two-sided 95% Wilson score confidence interval for binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("Invalid k/n for Wilson interval")
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return np.clip(center - half_width, 0, 1), np.clip(center + half_width, 0, 1)


def plot_all_c_fraction(counts, output_dir):
    """Plot All-C checkpoint fraction with pointwise Wilson 95% intervals."""
    n = counts[['All-C', 'All-D', 'State-dependent']].sum(axis=1).to_numpy(dtype=int)
    k = counts['All-C'].to_numpy(dtype=int)
    if not np.all(n == len(EXPECTED_SEEDS)):
        raise AssertionError(f"Expected 15 seeds per B, got {n.tolist()}")
    b_values = counts.index.to_numpy(dtype=float)
    fraction = k / n
    lower, upper = wilson_interval(k, n)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 15,
        'axes.labelsize': 17,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(b_values, lower, upper, color=LINE_COLOR, alpha=0.16, linewidth=0)
    ax.plot(b_values, fraction, '-o', color=LINE_COLOR, linewidth=5,
            markersize=6, markerfacecolor=LINE_COLOR, label='All-C fraction', zorder=3)
    # Greedy evaluation mean
    ax.plot(
        b_values,
        counts['Greedy_mean'].to_numpy(dtype=float),
        '--s',
        color='#ff7f0e',
        linewidth=5,
        markersize=6,
        markerfacecolor='none',
        markeredgewidth=1.6,
        label='Greedy mean',
        zorder=4
    )
    ax.set_xlabel('B')
    ax.set_ylabel('Fraction of All-C policies')
    ax.set_xlim(b_values.min() - 0.025, b_values.max() + 0.025)
    ax.set_ylim(0, 1)
    ax.set_xticks(b_values)
    ax.set_xticklabels([f'{b:.2f}' for b in b_values])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right', fontsize=15)
    fig.tight_layout()
    for ext, dpi in [('png', 600), ('pdf', None)]:
        file_path = output_dir / f'{OUTPUT_STEM}.{ext}'
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        print(f'Saved: {file_path}')
    plt.show()
    plt.close(fig)


def main():
    counts = pd.read_csv(SUMMARY_CSV, index_col="B").sort_index()
    required = ["All-C", "All-D", "State-dependent", "Greedy_mean"]
    missing = set(required) - set(counts.columns)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    b_values = counts.index.to_numpy(dtype=float)
    if len(b_values) != len(EXPECTED_B) or not np.allclose(
        b_values, EXPECTED_B, atol=1e-6, rtol=0
    ):
        raise ValueError("Unexpected B values in summary CSV.")
    values = counts[required].to_numpy(dtype=float)
    k = values[:, :3]
    if not np.isfinite(values).all() or np.any(k < 0) or np.any(k != np.floor(k)):
        raise ValueError("Invalid policy counts or greedy means.")
    if np.any((values[:, 3] < 0) | (values[:, 3] > 1)):
        raise ValueError("Greedy means must be between 0 and 1.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_all_c_fraction(counts, OUTPUT_DIR)


if __name__ == "__main__":
    main()
