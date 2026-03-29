"""Generate ablation study figures and LaTeX table.

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
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = Path("results/ablations")
PLOTS_DIR = Path("plots/paper")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_FILES = {
    "gamma": "ablation_gamma.json",
    "K": "ablation_K_plans.json",
    "S": "ablation_S_plan_length.json",
    "budget": "ablation_exec_budget.json",
}

ABLATION_LABELS = {
    "gamma": r"$\gamma$ (discount factor)",
    "K": "$K$ (candidate plans)",
    "S": "$S$ (scripts per plan)",
    "budget": "Execution budget $N$",
}

ABLATION_PARAM_KEYS = {
    "gamma": "gamma",
    "K": "K",
    "S": "plan_length",
    "budget": "exec_budget",
}


def load_ablation(name):
    path = RESULTS_DIR / ABLATION_FILES[name]
    with open(path) as f:
        return json.load(f)


def get_values_and_means(data, param_key):
    """Extract parameter values and mean coverage for random + cov_qvalue."""
    results = data["results"]
    values = sorted(set(r[param_key] for r in results))

    random_means = []
    qvalue_means = []
    qvalue_ses = []
    deltas = []
    wins_list = []

    for v in values:
        v_results = [r for r in results if r[param_key] == v]
        rand = [r["random"]["final"] for r in v_results]
        qv = [r["cov_qvalue"]["final"] for r in v_results]
        d = [q - r for q, r in zip(qv, rand)]

        random_means.append(np.mean(rand))
        qvalue_means.append(np.mean(qv))
        qvalue_ses.append(np.std(qv) / np.sqrt(len(qv)))
        deltas.append(np.mean(d))
        wins_list.append(sum(1 for x in d if x > 0))

    return values, random_means, qvalue_means, qvalue_ses, deltas, wins_list


# ---------------------------------------------------------------------------
# Figure: 2×2 ablation grid
# ---------------------------------------------------------------------------

def fig_ablation_grid():
    """2×2 grid showing all ablations: bars for CovQValue, dashed line for Random."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (name, file) in zip(axes.flat, ABLATION_FILES.items()):
        data = load_ablation(name)
        param_key = ABLATION_PARAM_KEYS[name]
        values, rand_means, qv_means, qv_ses, deltas, wins = \
            get_values_and_means(data, param_key)

        x = np.arange(len(values))
        x_labels = [str(v) for v in values]

        # CovQValue bars
        bars = ax.bar(x, qv_means, 0.5, yerr=qv_ses, color="#E84C4C",
                      alpha=0.85, capsize=5, label="CovQValue")

        # Random dashed line
        ax.plot(x, rand_means, "k--o", markersize=6, linewidth=1.5,
                label="Random", zorder=5)

        # Delta annotations on bars
        for i, (qv, d) in enumerate(zip(qv_means, deltas)):
            ax.annotate(f"+{d:.0f}", (i, qv + qv_ses[i] + 1),
                       ha="center", fontsize=9, color="#B03030")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(ABLATION_LABELS[name])
        ax.set_ylabel("Mean Branch Coverage")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_ablations.pdf")
    plt.savefig(PLOTS_DIR / "fig_ablations.png")
    plt.close()
    print("Saved fig_ablations")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def table_ablations():
    """Generate ablation LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation studies on RepoExploreBench (93 targets, Gemini Flash). "
                 r"Default values: $\gamma{=}0.5$, $K{=}3$, $S{=}3$, $N{=}24$. "
                 r"Bold indicates the default setting.}")
    lines.append(r"\label{tab:ablations}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Ablation} & \textbf{Value} & \textbf{Random} & "
                 r"\textbf{CovQValue} & \textbf{$\Delta$} & \textbf{Wins} \\")
    lines.append(r"\midrule")

    defaults = {"gamma": 0.5, "K": 3, "S": 3, "budget": 24}
    default_param = {"gamma": "gamma", "K": "K", "S": "plan_length", "budget": "exec_budget"}

    for name, label_short in [("gamma", r"$\gamma$"), ("K", "$K$"),
                                ("S", "$S$"), ("budget", "$N$")]:
        data = load_ablation(name)
        param_key = ABLATION_PARAM_KEYS[name]
        values, rand_means, qv_means, _, deltas, wins = \
            get_values_and_means(data, param_key)

        for i, v in enumerate(values):
            is_default = (v == defaults[name])
            name_col = label_short if i == 0 else ""
            v_str = f"\\textbf{{{v}}}" if is_default else str(v)
            qv_str = f"\\textbf{{{qv_means[i]:.1f}}}" if is_default else f"{qv_means[i]:.1f}"

            lines.append(f"  {name_col} & {v_str} & {rand_means[i]:.1f} & "
                        f"{qv_str} & +{deltas[i]:.1f} & {wins[i]}/93 \\\\")

        if name != "budget":
            lines.append(r"  \addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    with open(PLOTS_DIR / "table_ablations.tex", "w") as f:
        f.write(table)
    print("Saved table_ablations.tex")
    print(table)


def main():
    fig_ablation_grid()
    print()
    table_ablations()


if __name__ == "__main__":
    main()
