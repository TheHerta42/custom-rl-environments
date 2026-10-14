"""Plot the Pac-Man DQN results.

Data comes from CSVs extracted out of the training notebook's saved cell
outputs (see results/dqn_*.csv). Produces results/dqn_results.png:

  (a) training curve — the only real training log, 500 episodes of
      thesecond.ipynb, printed every 20 episodes (12 recorded points)
  (b) evaluation: dots eaten per episode across the three evaluation runs
  (c) evaluation: per-episode reward distribution, which exposes how much the
      runs differ in *variance*, not just in mean

Run from the repository root:
    python tools/plot_dqn.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

TOTAL_DOTS = 135  # board size; used as the "cleared the board" reference line


def read_csv(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot():
    train = read_csv("dqn_training.csv")
    te = [int(r["episode"]) for r in train]
    tr_rew = [float(r["avg_reward"]) for r in train]
    tr_dots = [float(r["avg_dots"]) for r in train]

    runs = {
        "My_practice_cell11": "My_practice (cell 11)\nreward-model01.pth",
        "My_practice_cell14": "My_practice (cell 14)\nreward-model01.pth",
        "thefirst_cell13": "thefirst (cell 13)\nthreech1.pth (3-ch)",
    }
    evals = {k: read_csv(f"dqn_eval_{k}.csv") for k in runs}
    with open(os.path.join(RES, "dqn_eval_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # ---- (a) training curve ----
    ax = axes[0]
    ax.plot(te, tr_rew, "o-", color="#4C72B0", label="avg reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("avg reward (last 20 eps)", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(te, tr_dots, "s--", color="#55A868", label="avg dots eaten")
    ax2.axhline(TOTAL_DOTS, color="grey", lw=1, ls=":", alpha=0.7)
    ax2.set_ylabel("avg dots eaten", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")
    ax2.set_ylim(min(tr_dots) - 3, TOTAL_DOTS + 2)
    # mark the degradation point
    worst = int(np.argmin(tr_rew))
    ax.annotate(f"ep {te[worst]}\ndegrades",
                xy=(te[worst], tr_rew[worst]), xytext=(-40, -28),
                textcoords="offset points", fontsize=8, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1))
    ax.set_title("(a) Training — 500 eps, 12 logged points\n(thesecond.ipynb)")

    # ---- (b) dots eaten ----
    ax = axes[1]
    names = list(runs)
    means = [summary[k]["avg_dots"] for k in names]
    sds = [np.std([int(r["dots"]) for r in evals[k]]) for k in names]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    ax.bar(range(len(names)), means, yerr=sds, capsize=5, color=colors, alpha=0.85)
    ax.axhline(TOTAL_DOTS, color="grey", lw=1, ls=":", alpha=0.7)
    ax.text(len(names) - 0.5, TOTAL_DOTS + 0.5, "board size = 135",
            fontsize=8, color="grey", ha="right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([runs[n] for n in names], fontsize=8)
    ax.set_ylabel("dots eaten per episode")
    ax.set_ylim(0, 145)
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + 2, f"{m:.1f}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("(b) Evaluation — 2000 eps each\nmean ± sd")

    # ---- (c) reward distribution ----
    ax = axes[2]
    for k, c in zip(names, colors):
        rew = [float(r["reward"]) for r in evals[k]]
        ax.hist(rew, bins=60, range=(-60, 145), histtype="step", lw=1.8,
                color=c, label=f"{k}\nsd={np.std(rew):.2f}")
    ax.set_xlabel("episode reward")
    ax.set_ylabel("episodes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("(c) Reward spread — same weights, different variance")

    fig.suptitle("Pac-Man DQN (Dueling + Double + PER + NoisyNet + n-step + CNN)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(RES, "dqn_results.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    plot()
