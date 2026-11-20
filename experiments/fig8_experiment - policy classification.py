"""Analyze all 32 binary states for each trained shared-DQN checkpoint.

Run from the repository root after generating checkpoints and sampled states
with the Figure 6 experiment. The exported CSV is used by Figure 8 policy
validation. Action-label silhouette and linear-probe AUC are also reported.
"""

from pathlib import Path
import torch.nn as nn
import re
import itertools
import numpy as np
import pandas as pd
import torch

from scipy.stats import t
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


# ============================================================
# Figures 6 and 8: Action Discriminability and Policy Classification
# All 32 possible states, per checkpoint
# ============================================================

STATES_DIR = Path("states_95k_100k")
CHECKPOINT_DIR = Path("checkpoints_shared")
OUTPUT_DIR = Path("figure8")
TIE_TOL = 1e-8


class Net(nn.Module):
    """Single-hidden-layer DQN matching the checkpoint architecture."""

    def __init__(self, state_dim=5, hidden_dim=96, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward_hidden(self, x):
        return self.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.forward_hidden(x))


EXPECTED_SEEDS = list(range(195, 210))
EXPECTED_B_COUNT = 6

CV_FOLDS = 5
CV_SEED = 0

state_pattern = re.compile(
    r"state_Size(?P<size>\d+)_SEED(?P<seed>\d+)"
    r"_Dr(?P<dr>[\d.]+)_tauinit_(?P<tauinit>[\d.]+)"
    r"_B_(?P<B>[\d.]+)\.npy"
)

ALL_STATES = np.array(
    list(itertools.product([0, 1], repeat=5)),
    dtype=np.float32
)

ALL_STATES_T = torch.from_numpy(ALL_STATES)


def mean_ci95(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    mean = float(np.mean(values))

    if n == 1:
        return mean, np.nan, np.nan, np.nan, 1

    sd = float(np.std(values, ddof=1))
    sem = sd / np.sqrt(n)

    margin = float(
        t.ppf(0.975, df=n - 1) * sem
    )

    return (
        mean,
        sd,
        mean - margin,
        mean + margin,
        n
    )


def main():
    records = []


    # ============================================================
    # 1. Load states and checkpoints
    # ============================================================

    for state_path in sorted(STATES_DIR.iterdir()):

        if not state_path.is_file():
            continue
        fname = state_path.name

        m = state_pattern.fullmatch(fname)

        if m is None:
            continue

        size = int(m.group("size"))
        seed = int(m.group("seed"))
        dr_str = m.group("dr")
        tau_str = m.group("tauinit")
        B_str = m.group("B")

        Bv = float(B_str)

        if size != 30 or not np.isclose(float(dr_str), 0.25):
            continue

        ckpt_name = (
            f"sharedDQN_SIZE30_SEED{seed}_Dr{dr_str}"
            f"_tauinit_{tau_str}"
            f"_taufinal_0.1_anneal_95000_B_{B_str}.pt"
        )

        ckpt_path = CHECKPOINT_DIR / ckpt_name

        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

        S = np.load(state_path).astype(np.float32)

        if S.ndim != 2 or S.shape[1] != 5:
            raise ValueError(
                f"Invalid state shape: {fname}, {S.shape}"
            )

        if not np.all((S == 0) | (S == 1)):
            raise ValueError(
                f"Non-binary states found: {fname}"
            )

        S_t = torch.from_numpy(S)

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=True
        )

        model = Net(
            state_dim=5,
            hidden_dim=96,
            action_dim=2
        )

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()


        # ========================================================
        # 2. Extract hidden activations and greedy-action labels
        # ========================================================

        with torch.no_grad():

            # Original sampled observations
            Q_raw = model(S_t).numpy()
            y_raw = np.argmax(Q_raw, axis=1)

            # All 32 possible states
            H = model.forward_hidden(ALL_STATES_T).numpy()
            Q = model(ALL_STATES_T).numpy()
            y = np.argmax(Q, axis=1)

        # Near-ties validation across all 32 states

        if not np.isfinite(Q).all():
            raise ValueError(
                f"Non-finite Q-values: B={Bv}, seed={seed}"
            )

        q_gap = Q[:, 0] - Q[:, 1]
        abs_q_gap = np.abs(q_gap)

        n_near_ties = int(np.sum(abs_q_gap <= TIE_TOL))
        min_abs_Q_gap = float(np.min(abs_q_gap))
        mean_abs_Q_gap = float(np.mean(abs_q_gap))

        # ========================================================
        # 3. Class balance
        # ========================================================

        n_C_raw = int(np.sum(y_raw == 0))
        n_D_raw = int(np.sum(y_raw == 1))

        n_C_all = int(np.sum(y == 0))
        n_D_all = int(np.sum(y == 1))

        n_unique_observed = int(
            len(np.unique(S, axis=0))
        )

        min_class = min(n_C_all, n_D_all)

        single_class = (
            n_C_all == 0 or n_D_all == 0
        )


        # ========================================================
        # 4. Action-label silhouette
        # ========================================================

        action_silhouette = np.nan

        if not single_class:

            action_silhouette = float(
                silhouette_score(
                    H,
                    y,
                    metric="euclidean"
                )
            )


        # ========================================================
        # 5. Linear Probe ROC-AUC
        # ========================================================

        probe_auc = np.nan
        auc_sd_folds = np.nan
        n_folds_used = 0

        if min_class >= 2:

            n_folds_used = min(
                CV_FOLDS,
                min_class
            )

            cv = StratifiedKFold(
                n_splits=n_folds_used,
                shuffle=True,
                random_state=CV_SEED
            )

            fold_aucs = []

            for train_idx, test_idx in cv.split(H, y):

                X_train = H[train_idx]
                X_test = H[test_idx]

                y_train = y[train_idx]
                y_test = y[test_idx]

                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=CV_SEED
                    )
                )

                clf.fit(X_train, y_train)

                y_score = clf.predict_proba(X_test)[:, 1]

                fold_auc = roc_auc_score(
                    y_test,
                    y_score
                )

                fold_aucs.append(float(fold_auc))

            probe_auc = float(np.mean(fold_aucs))

            if len(fold_aucs) >= 2:
                auc_sd_folds = float(
                    np.std(fold_aucs, ddof=1)
                )


        # ========================================================
        # 6. Save per-checkpoint results
        # ========================================================

        records.append({
            "B": Bv,
            "seed": seed,

            "n_near_ties": n_near_ties,
            "min_abs_Q_gap": min_abs_Q_gap,
            "mean_abs_Q_gap": mean_abs_Q_gap,

            "n_samples": len(S),
            "n_unique_observed": n_unique_observed,
            "n_all_states": len(ALL_STATES),

            "n_C_raw": n_C_raw,
            "n_D_raw": n_D_raw,

            "n_C_all": n_C_all,
            "n_D_all": n_D_all,

            "single_class": single_class,

            "action_silhouette": action_silhouette,

            "probe_auc": probe_auc,
            "auc_sd_folds": auc_sd_folds,
            "cv_folds": n_folds_used
        })


    # ============================================================
    # 7. Per-checkpoint results
    # ============================================================

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            "No valid state/checkpoint pairs found."
        )

    df = df.sort_values(
        ["B", "seed"]
    ).reset_index(drop=True)

    if df.duplicated(["B", "seed"]).any():
        raise ValueError(
            "Duplicate B/seed configurations detected."
        )

    print("\n========== Per-Checkpoint Results ==========")

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ============================================================
    # 9. Aggregate across seeds for each B
    # ============================================================

    summary_records = []

    for Bv, group in df.groupby("B", sort=True):

        (
            sil_mean,
            sil_sd,
            sil_lo,
            sil_hi,
            sil_n
        ) = mean_ci95(group["action_silhouette"])

        (
            auc_mean,
            auc_sd,
            auc_lo,
            auc_hi,
            auc_n
        ) = mean_ci95(group["probe_auc"])

        summary_records.append({

            "B": Bv,
            "n_seeds": group["seed"].nunique(),

            "sil_mean": sil_mean,
            "sil_sd": sil_sd,
            "sil_ci_low": sil_lo,
            "sil_ci_high": sil_hi,
            "sil_valid": sil_n,

            "auc_mean": auc_mean,
            "auc_sd": auc_sd,
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
            "auc_valid": auc_n,

            "total_C_raw": int(group["n_C_raw"].sum()),
            "total_D_raw": int(group["n_D_raw"].sum()),

            "single_class_seeds": int(
                group["single_class"].sum()
            ),

            "mean_unique_observed": float(
                group["n_unique_observed"].mean()
            )

        })

    summary = pd.DataFrame(summary_records)


    # ============================================================
    # 10. Seed validation
    # ============================================================

    print("\n========== Seed Validation ==========")

    expected_set = set(EXPECTED_SEEDS)

    for Bv, group in df.groupby("B", sort=True):

        seeds = sorted(
            group["seed"].unique().tolist()
        )

        status = (
            "PASS"
            if set(seeds) == expected_set
            else "WARNING"
        )

        print(
            f"B={Bv:.6f} | "
            f"Seeds={len(seeds)} | "
            f"Status={status} | "
            f"Seed IDs={seeds}"
        )

    B_count = df["B"].nunique()

    print(
        "\nExpected B count:",
        EXPECTED_B_COUNT
    )

    print(
        "Actual B count:",
        B_count
    )

    print(
        "B count status:",
        "PASS" if B_count == EXPECTED_B_COUNT
        else "WARNING"
    )

    seed_sets = [
        set(group["seed"])
        for _, group in df.groupby("B", sort=True)
    ]

    print(
        "Identical seed sets across B:",
        all(s == seed_sets[0] for s in seed_sets)
    )


    # ============================================================
    # 11. Statistical summary
    # ============================================================

    print("\n========== Per-B Statistical Summary ==========")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # ============================================================
    # 12. Diagnostics
    # ============================================================

    print("\n========== Diagnostics ==========")

    print("Total checkpoints:", len(df))

    print(
        "Checkpoints with one greedy-action class:",
        int(df["single_class"].sum())
    )

    print(
        "Undefined action-label silhouette:",
        int(df["action_silhouette"].isna().sum())
    )

    print(
        "Undefined ROC-AUC:",
        int(df["probe_auc"].isna().sum())
    )

    print(
        "Checkpoints with fewer than 2 states in either class:",
        int(
            (df[["n_C_all", "n_D_all"]].min(axis=1) < 2).sum()
        )
    )

    print(
        "Minimum observed unique states:",
        int(df["n_unique_observed"].min())
    )

    print(
        "Maximum observed unique states:",
        int(df["n_unique_observed"].max())
    )

    print(
        "Number of evaluated states per checkpoint:",
        len(ALL_STATES)
    )

    print("\n========== Near-Ties Validation ==========")

    print("Threshold:", TIE_TOL)

    print(
        "Checkpoints with near-ties:",
        int((df["n_near_ties"] > 0).sum()),
        "/", len(df)
    )

    print(
        "Total near-tie states:",
        int(df["n_near_ties"].sum()),
        "/", len(df) * 32
    )

    print(
        "Global minimum absolute Q-gap:",
        df["min_abs_Q_gap"].min()
    )

    if (df["n_near_ties"] > 0).any():
        print("\nCheckpoints with near-ties:")
        print(
            df.loc[
                df["n_near_ties"] > 0,
                ["B", "seed", "n_near_ties", "min_abs_Q_gap"]
            ].to_string(index=False)
        )
    else:
        print("Near-ties validation: PASS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "greedy_32state_checkpoints.csv", index=False)


if __name__ == "__main__":
    main()
