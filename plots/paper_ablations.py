"""Generate ablation study figures and LaTeX tables.

Usage:
    python plots/paper_ablations.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
    "legend.fontsize": 12,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
})

RESULTS_DIR = Path("results/ablations")
PLOTS_DIR = Path("plots/paper")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COV_QVALUE_COLOR = "#C44E52"
RANDOM_COLOR = "#7F7F7F"


# ---------------------------------------------------------------------------
# Figure: Budget + S matched (2 panels)
# ---------------------------------------------------------------------------

def fig_ablation_budget_and_s():
    """Two-panel figure: (a) coverage vs budget, (b) coverage vs S."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): Budget ---
    data = json.load(open(RESULTS_DIR / "ablation_exec_budget.json"))
    results = data["results"]
    budgets = sorted(set(r["exec_budget"] for r in results))

    qv_means, qv_ses, rand_means = [], [], []
    for b in budgets:
        b_results = [r for r in results if r["exec_budget"] == b]
        qv = [r["cov_qvalue"]["final"] for r in b_results]
        rn = [r["random"]["final"] for r in b_results]
        qv_means.append(np.mean(qv))
        qv_ses.append(np.std(qv) / np.sqrt(len(qv)))
        rand_means.append(np.mean(rn))

    ax1.plot(budgets, qv_means, "o-", color=COV_QVALUE_COLOR, linewidth=2.5,
             markersize=10, label="CovQValue", zorder=3)
    ax1.fill_between(budgets,
                      [m - s for m, s in zip(qv_means, qv_ses)],
                      [m + s for m, s in zip(qv_means, qv_ses)],
                      color=COV_QVALUE_COLOR, alpha=0.15)
    ax1.plot(budgets, rand_means, "s--", color=RANDOM_COLOR, linewidth=2,
             markersize=8, label="Random", zorder=3)

    ax1.set_xlabel("Execution Budget $N$")
    ax1.set_ylabel("Mean Branch Coverage")
    ax1.set_title("(a) Budget Scaling")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(budgets)

    # --- Panel (b): S matched rounds ---
    data_s = json.load(open(RESULTS_DIR / "ablation_S_matched.json"))
    results_s = data_s["results"]
    s_values = [1, 3, 5]

    s_means, s_ses = [], []
    for s in s_values:
        vals = [r[f"S={s}"]["final"] for r in results_s]
        s_means.append(np.mean(vals))
        s_ses.append(np.std(vals) / np.sqrt(len(vals)))

    x = np.arange(len(s_values))
    bars = ax2.bar(x, s_means, 0.5, yerr=s_ses, color=COV_QVALUE_COLOR,
                    alpha=0.88, capsize=6, edgecolor="white", linewidth=0.5)

    # Budget labels on bars
    for i, (s, m, se) in enumerate(zip(s_values, s_means, s_ses)):
        budget = 8 * s
        ax2.annotate(f"$N$={budget}", (i, m + se + 1.5),
                    ha="center", fontsize=11, color="#555555")

    ax2.set_xlabel("Plan Length $S$")
    ax2.set_ylabel("Mean Branch Coverage")
    ax2.set_title("(b) Plan Length (8 rounds each)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in s_values])
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_ablations.pdf")
    plt.savefig(PLOTS_DIR / "fig_ablations.png")
    plt.close()
    print("Saved fig_ablations")


# ---------------------------------------------------------------------------
# LaTeX table: γ, K, diversity decomposition
# ---------------------------------------------------------------------------

def table_ablations():
    """Compact ablation table for γ, K, and diversity decomposition."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation studies on RepoExploreBench (93 targets, Gemini Flash). "
                 r"Top: hyperparameter sensitivity. Bottom: contribution of each component. "
                 r"Default values in bold.}")
    lines.append(r"\label{tab:ablations}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Ablation} & \textbf{Setting} & \textbf{Coverage} \\")
    lines.append(r"\midrule")

    # γ ablation
    data_g = json.load(open(RESULTS_DIR / "ablation_gamma.json"))
    gamma_vals = {0.0: [], 0.5: [], 1.0: []}
    for r in data_g["results"]:
        gamma_vals[r["gamma"]].append(r["cov_qvalue"]["final"])

    lines.append(r"$\gamma$ (discount) & 0.0 & " +
                 f"{np.mean(gamma_vals[0.0]):.1f}" + r" \\")
    lines.append(r" & \textbf{0.5} & \textbf{" +
                 f"{np.mean(gamma_vals[0.5]):.1f}" + r"} \\")
    lines.append(r" & 1.0 & " +
                 f"{np.mean(gamma_vals[1.0]):.1f}" + r" \\")
    lines.append(r"\addlinespace")

    # K ablation
    data_k = json.load(open(RESULTS_DIR / "ablation_K_plans.json"))
    k_vals = {1: [], 3: [], 5: []}
    for r in data_k["results"]:
        k_vals[r["K"]].append(r["cov_qvalue"]["final"])

    lines.append(r"$K$ (candidates) & 1 & " +
                 f"{np.mean(k_vals[1]):.1f}" + r" \\")
    lines.append(r" & \textbf{3} & \textbf{" +
                 f"{np.mean(k_vals[3]):.1f}" + r"} \\")
    lines.append(r" & 5 & " +
                 f"{np.mean(k_vals[5]):.1f}" + r" \\")

    lines.append(r"\midrule")

    # Diversity decomposition
    data_d = json.load(open(RESULTS_DIR / "ablation_diversity.json"))
    strats = {"cov_greedy": [], "cov_diverse": [],
              "cov_nodiversity": [], "cov_qvalue": []}
    for r in data_d["results"]:
        for s in strats:
            strats[s].append(r[s]["final"])

    lines.append(r"Component & Coverage map only & " +
                 f"{np.mean(strats['cov_greedy']):.1f}" + r" \\")
    lines.append(r" & + Diversity hints & " +
                 f"{np.mean(strats['cov_diverse']):.1f}" + r" \\")
    lines.append(r" & + Q-value scoring & " +
                 f"{np.mean(strats['cov_nodiversity']):.1f}" + r" \\")
    lines.append(r" & + Both (CovQValue) & \textbf{" +
                 f"{np.mean(strats['cov_qvalue']):.1f}" + r"} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    with open(PLOTS_DIR / "table_ablations.tex", "w") as f:
        f.write(table)
    print("Saved table_ablations.tex")
    print(table)


def main():
    fig_ablation_budget_and_s()
    print()
    table_ablations()


if __name__ == "__main__":
    main()
