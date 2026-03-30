"""Generate compact case study figure for corridor navigation.

Usage:
    python plots/paper_case_studies.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
})

PLOTS_DIR = Path("plots/paper")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "random": "#7F7F7F",
    "greedy": "#5975A4",
    "cov_greedy": "#5F9E6E",
    "cov_qvalue": "#C44E52",
}

LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "cov_greedy": "CovGreedy",
    "cov_qvalue": "CovQValue",
}

COMPACT_STYLES = {
    "random": {"linestyle": "--", "marker": "s", "markersize": 8, "markevery": 5},
    "greedy": {"linestyle": "-.", "marker": "^", "markersize": 8, "markevery": 5},
    "cov_greedy": {"linestyle": "-", "marker": "D", "markersize": 8, "markevery": 5},
    "cov_qvalue": {"linestyle": "-", "marker": "o", "markersize": 9, "markevery": 5, "linewidth": 5},
}

COMPACT_RC = {
    "font.size": 22,
    "axes.labelsize": 24,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.5,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "lines.linewidth": 4.0,
    "grid.linewidth": 1.2,
}

MODULES = [
    ("flask.app", r"$\tt{flask.app}$"),
    ("werkzeug.http", r"$\tt{werkzeug.http}$"),
    ("requests.models", r"$\tt{requests.models}$"),
    ("jinja2.ext", r"$\tt{jinja2.ext}$"),
]


def main():
    with open("results/repo_explore_bench/full_run_gemini.json") as f:
        data = json.load(f)

    targets = {r["module"]: r for r in data["results"]}

    with matplotlib.rc_context(COMPACT_RC):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

        for ax, (module, title) in zip(axes, MODULES):
            strategies = targets[module]["strategies"]
            for strat in ["random", "greedy", "cov_greedy", "cov_qvalue"]:
                curve = strategies[strat]["branch_curve"]
                steps = np.arange(1, len(curve) + 1)
                ax.plot(steps, curve,
                        color=COLORS[strat],
                        label=LABELS[strat],
                        **COMPACT_STYLES[strat])
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, 24)
            ax.set_xticks([1, 12, 24])
            ax.set_xlabel("Step")

        # Only first panel gets y-label
        axes[0].set_ylabel("Branches")

        # Single legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   fontsize=20, bbox_to_anchor=(0.5, -0.06),
                   columnspacing=1.2, handletextpad=0.5)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, wspace=0.3)
        plt.savefig(PLOTS_DIR / "fig_case_studies.pdf")
        plt.savefig(PLOTS_DIR / "fig_case_studies.png")
        plt.close()
        print("Saved fig_case_studies (4-panel compact)")


if __name__ == "__main__":
    main()
