import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("figure3")

INPUT_FILES = {
    "τ + Anneal (7 states)": BASE_DIR / "state size=7, tau+anneal.txt",
    "τ (6 states)": BASE_DIR / "state size=6, tau.txt",
    "Anneal (6 states)": BASE_DIR / "state size=6, anneal.txt",
    "Baseline": BASE_DIR / "baseline.txt"
}

pattern = re.compile(
    r"B:\s*([-+]?\d*\.?\d+).*?result:\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE
)

plt.figure(figsize=(14, 7), dpi=150)

for label, file_path in INPUT_FILES.items():
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                b = round(float(m.group(1)), 2)  
                result = float(m.group(2))
                records.append({"B": b, "result": result})

    df = pd.DataFrame(records)

    agg = (
        df.groupby("B", as_index=False)
        .agg(
            coop_mean=("result", "mean"),
            coop_std=("result", "std"),
            n=("result", "count")  
        )
        .sort_values("B")
    )

    plt.plot(
        agg["B"],
        agg["coop_mean"],
        marker="o",
        linewidth=7,
        markersize=16,
        label=label,
        zorder=3,
    )
    agg["coop_ci95"] = 1.96 * agg["coop_std"] / (agg["n"] ** 0.5)

    lower = (agg["coop_mean"] - agg["coop_ci95"]).clip(lower=0.0)
    upper = (agg["coop_mean"] + agg["coop_ci95"]).clip(upper=1.0)

    plt.fill_between(
        agg["B"],
        lower,
        upper,
        alpha=0.05,
        zorder=1
    )
    print(f"\n[{label}] n per B:")
    print(agg[["B", "n"]].to_string(index=False))
    print(f"[{label}] avg n across B = {agg['n'].mean():.2f}, min n = {agg['n'].min()}, max n = {agg['n'].max()}")

plt.xlabel("B", fontsize=26)
plt.ylabel("Cooperation Level", fontsize=26)
plt.legend(
    fontsize=22,
    loc="upper left",
    bbox_to_anchor=(1.02, 0.32),
    borderaxespad=0.
)

plt.tight_layout()
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.savefig("figure3_state_augmentation.png", dpi=300, bbox_inches="tight")
plt.savefig("figure3_state_augmentation.pdf", bbox_inches="tight")
plt.show()
