"""Validate 32-state greedy policy labels against greedy evaluation outcomes for Figure 8.

Run the 32-state checkpoint classification first and save its results to
figure8/greedy_32state_checkpoints.csv. Save the greedy evaluation log to
figure8/greedy_DQN.txt. Run this script from the repository root.

The exported columns B, seed, n_C_all, n_D_all, n_near_ties,
min_abs_Q_gap, mean_abs_Q_gap must be included in that cell's df.
No retraining or additional representation analysis is required.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# Input files, relative to the repository root.
DATA_DIR = Path('figure8')
POLICY_CSV = DATA_DIR / 'greedy_32state_checkpoints.csv'
GREEDY_LOG = DATA_DIR / 'greedy_DQN.txt'

EXPECTED_SEEDS = set(range(195, 210))
EXPECTED_B = [0.120250, 0.219999, 0.319748, 0.420248, 0.519997, 0.619746]
B_TOL = 0.001
RESULT_TOL = 1e-6
TIE_TOL = 1e-8


def b_index(b):
    differences = np.abs(np.asarray(EXPECTED_B) - float(b))
    i = int(np.argmin(differences))
    if differences[i] > B_TOL:
        raise ValueError(f'Unexpected B={b}; expected approximately one of {EXPECTED_B}')
    return i


def load_policy_csv(path):
    df = pd.read_csv(path)
    required = {'B', 'seed', 'n_C_all', 'n_D_all', 'n_near_ties',
                'min_abs_Q_gap', 'mean_abs_Q_gap'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: missing columns {sorted(required - set(df.columns))}')
    df = df.copy()
    df['B_idx'] = df['B'].map(b_index)
    df['seed'] = df['seed'].astype(int)
    df['n_C_all'] = df['n_C_all'].astype(int)
    df['n_D_all'] = df['n_D_all'].astype(int)
    for col in ['n_near_ties', 'min_abs_Q_gap', 'mean_abs_Q_gap']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df[['n_near_ties', 'min_abs_Q_gap', 'mean_abs_Q_gap']].to_numpy(dtype=float)).all():
        raise ValueError('Near-ties diagnostics contain missing or non-finite values.')
    if not ((df['n_near_ties'] >= 0) & (df['n_near_ties'] <= 32) &
            (df['n_near_ties'] == np.floor(df['n_near_ties']))).all():
        raise ValueError('Invalid n_near_ties values.')
    df['n_near_ties'] = df['n_near_ties'].astype(int)
    if not ((df['min_abs_Q_gap'] >= 0) &
            (df['mean_abs_Q_gap'] >= df['min_abs_Q_gap'])).all():
        raise ValueError('Invalid Q-gap diagnostics.')
    if not ((df.n_C_all + df.n_D_all) == 32).all():
        raise ValueError('At least one checkpoint does not have exactly 32 greedy labels.')
    df['policy_type'] = np.select(
        [df.n_C_all.eq(32), df.n_D_all.eq(32)],
        ['All-C', 'All-D'], default='State-dependent'
    )
    return df[['B_idx', 'seed', 'n_C_all', 'n_D_all', 'policy_type',
               'n_near_ties', 'min_abs_Q_gap', 'mean_abs_Q_gap']]


def load_greedy_log(path):
    pattern = re.compile(
        r'Dr:\s*(?P<dr>[0-9.]+).*?SIZE:\s*(?P<size>\d+)'
        r'.*?SEED:\s*(?P<seed>\d+).*?B:\s*(?P<B>[0-9.]+)'
        r'.*?result:\s*(?P<result>[0-9.]+)', re.I
    )
    rows = []
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        m = pattern.search(line)
        if not m:
            raise ValueError(f'Unrecognized greedy log line {lineno}: {line[:150]}')
        x = m.groupdict()
        if int(x['size']) != 900 or not np.isclose(float(x['dr']), 0.25):
            raise ValueError(f'Wrong simulation spec at line {lineno}: {line}')
        rows.append({
            'B_idx': b_index(x['B']), 'seed': int(x['seed']),
            'greedy_cooperation': float(x['result'])
        })
    return pd.DataFrame(rows)


def validate_grid(df, name):
    expected_keys = {(b, seed) for b in range(len(EXPECTED_B)) for seed in EXPECTED_SEEDS}
    if df.duplicated(['B_idx', 'seed']).any():
        examples = df.loc[df.duplicated(['B_idx', 'seed'], keep=False), ['B_idx', 'seed']]
        raise ValueError(f'{name}: duplicate keys:\n{examples.to_string(index=False)}')
    found = set(map(tuple, df[['B_idx', 'seed']].to_numpy()))
    missing, extra = expected_keys - found, found - expected_keys
    if missing or extra:
        raise ValueError(f'{name}: missing={sorted(missing)}, unexpected={sorted(extra)}')
    print(f'{name}: PASS — 90 unique (B, seed) combinations')


def main():
    policy = load_policy_csv(POLICY_CSV)
    greedy = load_greedy_log(GREEDY_LOG)
    validate_grid(policy, '32-state policy classifications')
    validate_grid(greedy, 'Greedy evaluation')

    merged = policy.merge(greedy, on=['B_idx', 'seed'], how='outer', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Incomplete policy/greedy merge')
    merged = merged.drop(columns='_merge')
    merged['B'] = merged.B_idx.map(lambda i: EXPECTED_B[i])

    is_c = merged.policy_type.eq('All-C')
    is_d = merged.policy_type.eq('All-D')
    is_mixed = merged.policy_type.eq('State-dependent')

    merged['status'] = np.where(
        (is_c & np.isclose(merged.greedy_cooperation, 1, atol=RESULT_TOL)) |
        (is_d & np.isclose(merged.greedy_cooperation, 0, atol=RESULT_TOL)),
        'PASS', np.where(is_mixed, 'MIXED_CHECK', 'MISMATCH')
    )

    merged = merged[['B', 'seed', 'policy_type', 'n_C_all', 'n_D_all',
                     'n_near_ties', 'min_abs_Q_gap', 'mean_abs_Q_gap',
                     'greedy_cooperation', 'status']].sort_values(['B', 'seed'])
    output_dir = POLICY_CSV.parent
    merged.to_csv(output_dir / 'greedy_policy_validation_90.csv', index=False)

    counts = merged.pivot_table(index='B', columns='policy_type', values='seed', aggfunc='count', fill_value=0)
    counts = counts.reindex(columns=['All-C', 'All-D', 'State-dependent'], fill_value=0)
    means = merged.groupby('B')['greedy_cooperation'].mean()
    counts['All-C_fraction'] = counts['All-C'] / 15
    counts['Greedy_mean'] = means
    counts['Near_tie_checkpoints'] = merged.groupby('B')['n_near_ties'].apply(lambda s: int((s > 0).sum()))
    counts['Near_tie_states'] = merged.groupby('B')['n_near_ties'].sum().astype(int)
    counts['Min_abs_Q_gap'] = merged.groupby('B')['min_abs_Q_gap'].min()
    counts['Mean_abs_Q_gap'] = merged.groupby('B')['mean_abs_Q_gap'].mean()
    counts.to_csv(output_dir / 'greedy_policy_validation_summary.csv')

    print('\n=== Summary (per B) ===')
    print(counts.to_string(float_format=lambda x: f'{x:.6f}'))
    print('\n=== Near-ties validation (32 states per checkpoint) ===')
    print(f'Threshold: |Q(C)-Q(D)| <= {TIE_TOL:g}')
    print(f'Checkpoints with near-ties: {(merged.n_near_ties > 0).sum()} / {len(merged)}')
    print(f'Total near-tie states: {merged.n_near_ties.sum()} / {32 * len(merged)}')
    print(f'Global minimum absolute Q-gap: {merged.min_abs_Q_gap.min():.12g}')
    if (merged.n_near_ties > 0).any():
        print(merged.loc[merged.n_near_ties > 0,
                         ['B', 'seed', 'policy_type', 'n_near_ties', 'min_abs_Q_gap']].to_string(index=False))
        print('Near-ties validation: FLAGGED — inspect checkpoints above')
    else:
        print('Near-ties validation: PASS (zero near-ties)')
    print('\n=== Validation status ===')
    print(merged.status.value_counts().to_string())
    print('\n=== State-dependent checkpoints (require individual interpretation) ===')
    print(merged.loc[is_mixed, ['B', 'seed', 'n_C_all', 'n_D_all', 'greedy_cooperation']].to_string(index=False))
    mismatches = merged.loc[merged.status.eq('MISMATCH')]
    if not mismatches.empty:
        print('\nMISMATCHES:\n', mismatches.to_string(index=False))
        raise AssertionError(f'{len(mismatches)} constant-policy / greedy-evaluation mismatches')

    expected_counts = [(10, 3, 2), (9, 6, 0), (8, 7, 0),
                       (6, 9, 0), (4, 11, 0), (2, 13, 0)]
    actual_counts = list(counts[['All-C', 'All-D', 'State-dependent']].itertuples(index=False, name=None))
    if actual_counts != expected_counts:
        print('\nWARNING: Classification counts differ from the previously reported 32-state table.')
        print('Actual:', actual_counts)
    else:
        print('\nPreviously reported 32-state counts: PASS')
    print(f'\nSaved: {output_dir / "greedy_policy_validation_90.csv"}')
    print(f'Saved: {output_dir / "greedy_policy_validation_summary.csv"}')
    print('Note: matching (B, seed) and outcomes does not by itself prove identical checkpoint weight files.')


if __name__ == '__main__':
    main()
