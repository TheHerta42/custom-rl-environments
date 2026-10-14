"""Evaluate a trained PPO policy on BossDuelEnv and count win / loss / draw.

Why this exists: the TensorBoard logs for this project only record the *mean*
reward per evaluation batch, and that number is nearly uninterpretable here.
The reward is bimodal — +200 for a win, -200 for a loss, roughly 0 for a
timeout — so "mean reward +32, sd 92" tells you nothing about how often the
policy actually wins. This script plays N episodes and counts outcomes.

Usage (from the repository root):
    python tools/eval_boss_duel.py --model /path/to/ppo_treth.zip -n 200
    python tools/eval_boss_duel.py --model ... -n 200 --out results/boss_duel_outcomes.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "boss_duel"))

from BossDuelEnv import BossDuelEnv  # noqa: E402


def run_episode(env, model, deterministic=True):
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    if env.agent_health <= 0:
        # BossDuelEnv.step() tests agent death BEFORE boss death when assigning
        # the +-200 terminal reward, so a mutual kill counts as a LOSS. Match
        # that precedence here or those episodes get mislabelled as wins.
        outcome = "loss" if env.boss_health > 0 else "mutual"
    elif env.boss_health <= 0:
        outcome = "win"
    else:
        outcome = "draw"
    return outcome, total, steps, env.agent_health, env.boss_health


def make_plot(rows, out_png):
    """Outcome mix + reward distribution.

    The second panel is the point: the reward is bimodal, so any single 'mean
    reward' figure thrown around for this environment is close to meaningless.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    order = ["win", "mutual", "loss", "draw"]
    labels = {"win": "win\n(clean)", "mutual": "mutual kill\n(both died)",
              "loss": "loss", "draw": "draw\n(50-step cap)"}
    colors = {"win": "#55A868", "mutual": "#C44E52", "loss": "#4C72B0", "draw": "#999999"}
    counts = {k: sum(r["outcome"] == k for r in rows) for k in order}
    n = len(rows)
    rew = [float(r["reward"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ks = [k for k in order if counts[k] > 0]
    ax.bar([labels[k] for k in ks], [counts[k] for k in ks],
           color=[colors[k] for k in ks])
    for i, k in enumerate(ks):
        ax.text(i, counts[k] + n * 0.012, f"{counts[k]}\n{counts[k]/n*100:.1f}%",
                ha="center", fontsize=9)
    ax.set_ylabel("episodes")
    ax.set_ylim(0, max(counts.values()) * 1.25)
    ax.set_title(f"Outcome mix over {n} episodes\n"
                 "mutual kill is scored as a loss by the env")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.hist(rew, bins=60, color="#4C72B0", alpha=0.8)
    mean = sum(rew) / len(rew)
    ax.axvline(mean, color="#C44E52", lw=2,
               label=f"mean {mean:.1f}  (describes no actual episode)")
    ax.set_xlabel("episode reward")
    ax.set_ylabel("episodes")
    ax.set_title("Reward is bimodal: ±200 for win/loss, ~0 for a timeout")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the SB3 .zip")
    ap.add_argument("-n", "--episodes", type=int, default=200)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of taking the argmax")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "boss_duel_outcomes.csv"))
    ap.add_argument("--out-png", default=os.path.join(ROOT, "results", "boss_duel_outcomes.png"),
                    help="outcome chart; pass '' to skip")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    if not os.path.exists(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    env = BossDuelEnv()
    model = PPO.load(args.model, device="cpu")
    print(f"model   : {os.path.basename(args.model)}")
    print(f"env     : BossDuelEnv.py  boss_hp={env.boss_health_max}  "
          f"probs={env.boss_probs}  ult={env.agent_ult_damage}")
    print(f"policy  : {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"episodes: {args.episodes}\n")

    rows = []
    for i in range(args.episodes):
        np.random.seed(10_000 + i)  # BossDuelEnv draws from global np.random
        outcome, rew, steps, ahp, bhp = run_episode(env, model,
                                                    deterministic=not args.stochastic)
        rows.append(dict(episode=i, outcome=outcome, reward=rew, steps=steps,
                         agent_hp=ahp, boss_hp=bhp))

    n = len(rows)
    wins = sum(r["outcome"] == "win" for r in rows)
    losses = sum(r["outcome"] == "loss" for r in rows)
    draws = sum(r["outcome"] == "draw" for r in rows)
    mutuals = sum(r["outcome"] == "mutual" for r in rows)
    rew = np.array([r["reward"] for r in rows])
    steps = np.array([r["steps"] for r in rows])

    print("=" * 58)
    print(f"  win     {wins:4}  {wins/n*100:5.1f}%")
    print(f"  loss    {losses:4}  {losses/n*100:5.1f}%")
    print(f"  mutual  {mutuals:4}  {mutuals/n*100:5.1f}%   "
          f"(both died — counted as a loss, see below)")
    print(f"  draw    {draws:4}  {draws/n*100:5.1f}%   (hit the 50-step cap)")
    print("=" * 58)
    print(f"  effective win rate  {wins/n*100:5.1f}%   "
          f"loss rate {(losses+mutuals)/n*100:5.1f}%")
    print(f"  mean reward {rew.mean():8.2f}  (sd {rew.std():.2f}, "
          f"min {rew.min():.2f}, max {rew.max():.2f})")
    print(f"  mean steps  {steps.mean():8.2f}  (min {steps.min()}, max {steps.max()})")

    # A mean reward is only meaningful once you know the outcome mix.
    print(f"\n  => the 'mean reward' of {rew.mean():.2f} is "
          f"{wins/n*100:.0f}% wins against {(losses+mutuals)/n*100:.0f}% losses, "
          f"not a typical episode.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "outcome", "reward",
                                          "steps", "agent_hp", "boss_hp"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {args.out}")

    if args.out_png:
        make_plot(rows, args.out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
