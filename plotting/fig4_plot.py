"""Generate Figure 4: cooperation levels and paired differences on a 10 x 10 lattice.

Run from the repository root. Input files contain printed simulation summaries.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# =========================
# 1. File paths
# =========================
DATA_DIR = Path("figure4")
OUTPUT_DIR = Path("figures")

GROUPED_PATH = DATA_DIR / "grouped_DQN.txt"
INDEPENDENT_PATH = DATA_DIR / "independent_DQN.txt"
QLEARNING_PATH = DATA_DIR / "tabular_Qlearning.txt"
SHARED_PATH = DATA_DIR / "shared_DQN.txt"

EXPECTED_SEEDS = list(range(195, 210))


# =========================
# 2. Parsing
# =========================
def parse_result_file(path, model_name):
    rows = []
    pattern = re.compile(
        r'Dr:\s*([-\d.]+),.*?SEED:\s*(\d+),.*?B:(\d+\.\d+),\s*result:\s*([-\d.]+)'
    )

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            match = pattern.search(line)
            if match:
                dr = float(match.group(1))
                seed = int(match.group(2))
                B = float(match.group(3))
                result = float(match.group(4))

                # Reject mixed experimental configurations before deduplication.
                size_match = re.search(r"\bSIZE:\s*(\d+)", line, re.IGNORECASE)
                if not np.isclose(dr, 0.25, rtol=0, atol=1e-9):
                    raise ValueError(f"{path}:{line_num}: expected Dr=0.25, got {dr}")
                if size_match is None or int(size_match.group(1)) not in (10, 100):
                    raise ValueError(f"{path}:{line_num}: expected a 10 x 10 lattice")

                rows.append({
                    'model': model_name,
                    'Dr': dr,
                    'seed': seed,
                    'B': B,
                    'result': result,
                    'line_num': line_num,
                    'raw_line': line
                })
            else:
                print(f'[Warning] Unparsed line in {model_name} at line {line_num}:')
                print(line)

    if not rows:
        raise ValueError(f"No valid simulation records found in {path}")
    df = pd.DataFrame(rows)
    return df


# =========================
# 3. Seed check + duplicate removal
# =========================
def check_and_clean(df, model_name):
    print('=' * 80)
    print(f'[Checking] {model_name}')
    print(f'Raw rows: {len(df)}')

    # Seed range check
    unique_seeds = sorted(df['seed'].unique().tolist())
    print(f'Unique seeds found: {unique_seeds}')

    missing_seeds = sorted(set(EXPECTED_SEEDS) - set(unique_seeds))
    extra_seeds = sorted(set(unique_seeds) - set(EXPECTED_SEEDS))

    if missing_seeds:
        print(f'[Warning] Missing seeds: {missing_seeds}')
    else:
        print('All expected seeds 195-209 are present.')

    if extra_seeds:
        print(f'[Warning] Unexpected extra seeds: {extra_seeds}')
    else:
        print('No unexpected seeds.')

    # Duplicate check: duplicated by (seed, B)
    dup_mask = df.duplicated(subset=['seed', 'B'], keep=False)
    dup_df = df[dup_mask].sort_values(['seed', 'B', 'line_num'])

    if len(dup_df) > 0:
        print(f'[Warning] Found duplicated (seed, B) rows: {len(dup_df)}')
        print(dup_df[['seed', 'B', 'result', 'line_num']].to_string(index=False))
    else:
        print('No duplicated (seed, B) rows.')

    # Remove duplicates: keep first occurrence
    df_clean = (
        df.sort_values('line_num')
          .drop_duplicates(subset=['seed', 'B'], keep='first')
          .copy()
    )

    print(f'Rows after duplicate removal: {len(df_clean)}')

    # Final completeness check
    B_values = sorted(df_clean['B'].unique().tolist())
    expected_total = len(EXPECTED_SEEDS) * len(B_values)

    if len(df_clean) != expected_total:
        print(f'[Warning] Expected {expected_total} rows after cleaning, got {len(df_clean)}')
        missing_pairs = []
        for seed in EXPECTED_SEEDS:
            for b in B_values:
                mask = (df_clean['seed'] == seed) & (df_clean['B'] == b)
                if not mask.any():
                    missing_pairs.append((seed, b))
        if missing_pairs:
            print('[Warning] Missing (seed, B) pairs:')
            for item in missing_pairs:
                print(item)
    else:
        print(f'Complete grid confirmed: {len(EXPECTED_SEEDS)} seeds × {len(B_values)} B values = {expected_total}')

    print()
    return df_clean


# =========================
# 4. Summary statistics
# =========================
def mean_ci_by_B(df):
    summary_rows = []

    for B, group in df.groupby('B'):
        values = group['result'].values
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        ci = t.ppf(0.975, df=n - 1) * sem if n > 1 else 0.0

        summary_rows.append({
            'B': B,
            'mean': mean,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci,
            'n': n
        })

    summary = pd.DataFrame(summary_rows).sort_values('B').reset_index(drop=True)
    return summary


# =========================
# 5. Paired difference
# =========================
def paired_difference(df_a, df_b, label):
    merged = pd.merge(
        df_a[['seed', 'B', 'result']],
        df_b[['seed', 'B', 'result']],
        on=['seed', 'B'],
        suffixes=('_a', '_b')
    )

    merged['diff'] = merged['result_a'] - merged['result_b']

    rows = []
    for B, group in merged.groupby('B'):
        values = group['diff'].values
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        ci = t.ppf(0.975, df=n - 1) * sem if n > 1 else 0.0

        rows.append({
            'comparison': label,
            'B': B,
            'mean_diff': mean,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci,
            'n': n
        })

    result = pd.DataFrame(rows).sort_values('B').reset_index(drop=True)
    return result, merged


def main():
    # =========================
    # 6. Load and clean data
    # =========================
    grouped_raw = parse_result_file(GROUPED_PATH, 'Grouped DQN')
    independent_raw = parse_result_file(INDEPENDENT_PATH, 'Independent DQN')
    qlearning_raw = parse_result_file(QLEARNING_PATH, 'Q-learning')
    shared_raw = parse_result_file(SHARED_PATH, 'Shared DQN')

    grouped = check_and_clean(grouped_raw, 'Grouped DQN')
    independent = check_and_clean(independent_raw, 'Independent DQN')
    qlearning = check_and_clean(qlearning_raw, 'Q-learning')
    shared = check_and_clean(shared_raw, 'Shared DQN')


    # =========================
    # 7. Summaries
    # =========================
    grouped_summary = mean_ci_by_B(grouped)
    independent_summary = mean_ci_by_B(independent)
    qlearning_summary = mean_ci_by_B(qlearning)
    shared_summary = mean_ci_by_B(shared)

    diff_grouped_shared, merged_grouped_shared = paired_difference(
        grouped, shared, 'Grouped DQN - Shared DQN'
    )

    diff_grouped_independent, merged_grouped_independent = paired_difference(
        grouped, independent, 'Grouped DQN - Independent DQN'
    )

    print('=' * 80)
    print('[Paired difference] Grouped DQN - Shared DQN')
    print(diff_grouped_shared.to_string(index=False))
    print()

    print('=' * 80)
    print('[Paired difference] Grouped DQN - Independent DQN')
    print(diff_grouped_independent.to_string(index=False))
    print()


    # =========================
    # 8. Plot
    # =========================
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 15

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # -------- Left panel: four models --------
    left_configs = [
        ('Shared DQN', shared_summary, 'tab:blue', 'o'),
        ('Grouped DQN (10 groups)', grouped_summary, 'tab:orange', 'D'),
        ('Independent DQN', independent_summary, 'tab:green', '^'),
        ('Q-learning', qlearning_summary, 'tab:red', 's'),
    ]

    for label, df_plot, color, marker in left_configs:
        x = df_plot['B'].values
        y = df_plot['mean'].values
        lower = df_plot['ci_lower'].values
        upper = df_plot['ci_upper'].values

        ax1.plot(
            x, y,
            color=color,
            marker=marker,
            linewidth=3,
            markersize=8,
            label=label
        )
        ax1.fill_between(
            x, lower, upper,
            color=color,
            alpha=0.12
        )

    ax1.set_xlabel('B')
    ax1.set_ylabel('Cooperation Level')
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(shared_summary['B'].values)
    ax1.set_xticklabels([f"{b:.2f}" for b in shared_summary['B'].values])
    ax1.text(0.02, 1.02, '(A)', transform=ax1.transAxes, fontsize=18)
    ax1.legend(
        loc='upper right',
        bbox_to_anchor=(1.0, 1.52),
        frameon=True
    )

    # -------- Right panel: paired differences --------
    right_configs = [
        ('Grouped DQN - Shared DQN', diff_grouped_shared, 'tab:purple', 'o'),
        ('Grouped DQN - Independent DQN', diff_grouped_independent, 'tab:brown', 'D'),
    ]

    for label, df_plot, color, marker in right_configs:
        x = df_plot['B'].values
        y = df_plot['mean_diff'].values
        lower = df_plot['ci_lower'].values
        upper = df_plot['ci_upper'].values

        ax2.plot(
            x, y,
            color=color,
            marker=marker,
            linewidth=3,
            markersize=8,
            label=label
        )
        ax2.fill_between(
            x, lower, upper,
            color=color,
            alpha=0.12
        )

    ax2.axhline(0, color='gray', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('B')
    ax2.set_ylabel('Paired Mean Difference')
    ax2.set_xticks(shared_summary['B'].values)
    ax2.set_xticklabels([f"{b:.2f}" for b in shared_summary['B'].values])
    ax2.text(0.02, 1.02, '(B)', transform=ax2.transAxes, fontsize=18)
    ax2.legend(
        loc='upper right',
        bbox_to_anchor=(1.0, 1.30),
        frameon=True
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "figure4_independent_learner_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure4_independent_learner_comparison.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
