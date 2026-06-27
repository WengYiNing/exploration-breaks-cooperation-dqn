from pathlib import Path
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

GROUPED_FILE = Path("figure2_grouped_DQN.txt")
SHARED_FILE = Path("figure1_shared_DQN.txt")

BOUNDARY_SHARED  = 0.55
BOUNDARY_GROUPED = 0.15
DR_LO, DR_HI     = 0.10, 0.40     

SELECTED_B = [0.120250, 0.219999, 0.319748, 0.420248, 0.519997, 0.619746]
B_TOL = 1e-5

LINE_GROUPED = re.compile(
    r"Dr:\s*([0-9.]+).*?"
    r"SEED:\s*(\d+).*?"
    r"group_size:\s*(\d+).*?"
    r"B:\s*([0-9.]+).*?"
    r"result:\s*([0-9.]+)",
    re.IGNORECASE
)
LINE_SHARED = re.compile(
    r"Dr:\s*([0-9.]+).*?"
    r"SEED:\s*(\d+).*?"
    r"B:\s*([0-9.]+).*?"
    r"result:\s*([0-9.]+)",
    re.IGNORECASE
)

def read_grouped_rows(path):
    rows = []  

    if not path.exists():
        raise FileNotFoundError(f"Can't find Grouped file：{path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_GROUPED.search(line)
            if not m:
                continue

            dr, seed, gsz, b, res = (
                float(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
            )

            rows.append((b, dr, seed, gsz, res))
    return rows

def read_shared_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Can't Shared file：{path}")
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_SHARED.search(line)
            if not m:
                continue
            dr, seed, b, res = (
                float(m.group(1)),
                int(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
            )
            rows.append((b, dr, seed, res))

    return rows

def average_by_B_Dr_grouped(rows):

    bd_to_vals = defaultdict(list)

    for b, dr, seed, gsz, res in rows:
        bd_to_vals[(b, dr)].append(res)

    b_to_series = defaultdict(list)
    for (b, dr), vals in bd_to_vals.items():
        b_to_series[b].append((dr, float(np.mean(vals))))

    return b_to_series

def average_by_B_Dr_shared(rows):
    bd_to_vals = defaultdict(list)
    for b, dr, seed, res in rows:
        bd_to_vals[(b, dr)].append(res)
    b_to_series = defaultdict(list)
    for (b, dr), vals in bd_to_vals.items():
        b_to_series[b].append((dr, float(np.mean(vals))))
    return b_to_series

def filter_b_to_series(b_to_series, selected_b, tol=1e-5):
    filtered = defaultdict(list)

    for target_b in selected_b:
        matches = [b for b in b_to_series.keys() if abs(b - target_b) <= tol]
        if not matches:
            print(f"Warning: no B matched target B = {target_b}")
            continue

        b_match = matches[0]
        filtered[b_match] = b_to_series[b_match]

    return filtered

def dr_star_with_censor(series_pairs, boundary):
    if not series_pairs:
        return (np.nan, "left_censored")

    data = sorted(series_pairs, key=lambda x: x[0])
    drs = np.array([d for d, _ in data], dtype=float)
    co  = np.array([c for _, c in data], dtype=float)

    diff = co - boundary
    sign = np.sign(diff)

    cross_idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if cross_idx.size > 0:
        i = int(cross_idx[0])
        d1, d2 = drs[i], drs[i+1]
        c1, c2 = co[i],  co[i+1]
        t = (boundary - c1) / (c2 - c1)
        dr_cross = float(d1 + t * (d2 - d1))

        if dr_cross < DR_LO:
            return (dr_cross, "left_censored")
        elif dr_cross > DR_HI:
            return (dr_cross, "right_censored")
        else:
            return (dr_cross, "ok")

    if np.all(diff >= 0):
        return (float(drs.max()), "right_censored")
    if np.all(diff <= 0):
        return (float(drs.min()), "left_censored")

    if np.any(diff == 0):
        if not np.any(diff < 0):  
            return (float(drs.max()), "right_censored")
        if not np.any(diff > 0):  
            return (float(drs.min()), "left_censored")
        idx0 = int(np.where(diff == 0)[0][0])
        dr0 = float(drs[idx0])
        if dr0 < DR_LO:
            return (dr0, "left_censored")
        elif dr0 > DR_HI:
            return (dr0, "right_censored")
        else:
            return (dr0, "ok")


def build_df_drstar(b_to_series, boundary):
    rows_star = []

    for b in sorted(b_to_series.keys()):
        dr_star, status = dr_star_with_censor(b_to_series[b], boundary)
        rows_star.append((b, dr_star, status))

    df_star = (
        pd.DataFrame(rows_star, columns=["B", "Dr_star", "status_raw"])
        .sort_values("B")
    )

    df_star["Dr_star_clamped"] = df_star["Dr_star"].clip(DR_LO, DR_HI)
    df_star["status"] = np.where(
        df_star["status_raw"] == "left_censored",
        "below",
        np.where(df_star["status_raw"] == "right_censored", "above", "in"),
    )

    return df_star.reset_index(drop=True)

def compute_all():
    rows_grouped = read_grouped_rows(GROUPED_FILE)
    rows_shared  = read_shared_rows(SHARED_FILE)

    b2s_grouped = average_by_B_Dr_grouped(rows_grouped)
    b2s_shared  = average_by_B_Dr_shared(rows_shared)

    b2s_grouped = filter_b_to_series(b2s_grouped, SELECTED_B, B_TOL)
    b2s_shared  = filter_b_to_series(b2s_shared,  SELECTED_B, B_TOL)

    df_s_star = build_df_drstar(
        b2s_shared, BOUNDARY_SHARED
    )

    df_g_star = build_df_drstar(
        b2s_grouped, BOUNDARY_GROUPED
    )

    return {
        "df_s_star": df_s_star,
        "df_g_star": df_g_star,
    }

if __name__ == "__main__":
    data = compute_all(return_debug=True)

    df_s_star = data["df_s_star"]
    df_g_star = data["df_g_star"]

    print("\n========== Shared DQN ==========")
    print(df_s_star.to_string(index=False))

    print("\n========== Grouped DQN ==========")
    print(df_g_star.to_string(index=False))



data = compute_all()

df_s_star = data["df_s_star"]
df_g_star = data["df_g_star"]

fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex='col', sharey='row')

ax = axs[0]
m_in = df_s_star["status"] == "in"
m_lo = df_s_star["status"] == "below"
m_hi = df_s_star["status"] == "above"

ax.scatter(
    df_s_star.loc[m_in, "B"],
    df_s_star.loc[m_in, "Dr_star_clamped"],
    s=250,
    color="tab:blue",
    linewidths=4,
    zorder=10,
)

ax.scatter(
    df_s_star.loc[m_lo, "B"],
    df_s_star.loc[m_lo, "Dr_star_clamped"],
    s=300,
    facecolors="none",
    edgecolors="tab:blue",
    linewidths=4,
    zorder=10,
)

ax.scatter(
    df_s_star.loc[m_hi, "B"],
    df_s_star.loc[m_hi, "Dr_star_clamped"],
    s=300,
    facecolors="none",
    edgecolors="tab:blue",
    linewidths=4,
    zorder=10,
)

ax.plot(
    df_s_star["B"],
    df_s_star["Dr_star_clamped"],
    linewidth=8,
    color="tab:blue",
    zorder=5,
)

ax.axhline(DR_LO, linestyle="--", linewidth=4, color="gray", alpha=0.4, zorder=1)
ax.axhline(DR_HI, linestyle="--", linewidth=4, color="gray", alpha=0.4, zorder=1)

ax.set_ylabel("Dr*", fontsize=30)

ax = axs[1]

m_in = df_g_star["status"] == "in"
m_lo = df_g_star["status"] == "below"
m_hi = df_g_star["status"] == "above"

ax.scatter(
    df_g_star.loc[m_in, "B"],
    df_g_star.loc[m_in, "Dr_star_clamped"],
    s=250,
    color="tab:blue",
    linewidths=4,
    zorder=10,
)

ax.scatter(
    df_g_star.loc[m_lo, "B"],
    df_g_star.loc[m_lo, "Dr_star_clamped"],
    s=300,
    facecolors="none",
    edgecolors="tab:blue",
    linewidths=4,
    zorder=10,
)

ax.scatter(
    df_g_star.loc[m_hi, "B"],
    df_g_star.loc[m_hi, "Dr_star_clamped"],
    s=300,
    facecolors="none",
    edgecolors="tab:blue",
    linewidths=3,
    zorder=10,
)

ax.plot(
    df_g_star["B"],
    df_g_star["Dr_star_clamped"],
    linewidth=8,
    color="tab:blue",
    zorder=5,
)

ax.axhline(DR_LO, linestyle="--", linewidth=4, color="gray", alpha=0.4, zorder=1)
ax.axhline(DR_HI, linestyle="--", linewidth=4, color="gray", alpha=0.4, zorder=1)


for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=30)

axs[0].text(
    0.02, 1.12, "(a)",
    transform=axs[0].transAxes,
    fontsize=30,
    va="top",
    ha="left",
)

axs[1].text(
    0.02, 1.12, "(b)",
    transform=axs[1].transAxes,
    fontsize=30,
    va="top",
    ha="left",
)


fig.supxlabel("B", fontsize=30)

fig.tight_layout()
OUTPUT_FILE = Path("figures/figure2_boundary.png")
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
