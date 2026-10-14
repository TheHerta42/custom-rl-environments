"""Plot an SB3 PPO TensorBoard log.

Merges every event shard under the chosen directory by step (later writes win),
then draws three panels:

  1. eval/mean_reward and rollout/ep_rew_mean — both, because they can disagree
     in sign and reading only one gives the wrong conclusion
  2. eval/mean_ep_length and rollout/ep_len_mean
  3. train/explained_variance — whether the value function actually learned

Faint lines are raw points, bold lines are a rolling mean. The raw eval series
is very noisy (boss-duel's has a standard deviation of ~92), so the script also
prints **segment means**, which is what the READMEs quote.

Usage (from the repository root):
    python tools/plot_ppo.py --which pacman
    python tools/plot_ppo.py --which boss_duel
    python tools/plot_ppo.py --pattern "some/other/*.tfevents.*" -o out.png

Requires the `tensorboard` package.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Named log locations. Both live in this repo, so auto-detection would be
# ambiguous — pick one explicitly.
TARGETS = {
    "pacman": {
        "pattern": "pacman/ppo/pacman_tensorboard/PPO_*/*.tfevents.*",
        "out": "results/pacman_ppo_curve.png",
        "title": "PPO on self-built Pac-Man",
    },
    "boss_duel": {
        "pattern": "boss_duel/tensorboard/*.tfevents.*",
        "out": "results/boss_duel_curve.png",
        "title": "PPO on self-built Boss Duel env",
    },
}

C_EVAL = "#4C72B0"
C_ROLL = "#DD8452"


def merged(pattern, tag):
    """Merge every shard's scalar for `tag`, keyed by step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    m = {}
    for f in glob.glob(pattern):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        for e in acc.Scalars(tag):
            m[e.step] = e.value
    if not m:
        return None, None
    ks = sorted(m)
    return np.array(ks), np.array([m[k] for k in ks])


def thin(x, y, n=4000):
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx]


def rolling(y, w):
    """Centred rolling mean; returns an array the same length as y."""
    if len(y) < w:
        return y
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(padded, np.ones(w) / w, mode="valid")[: len(y)]


def report_segments(pattern, tag, name, n_segments=5):
    """Raw first/last points are useless when the eval series is noisy — the
    last value is just wherever the series happened to land. Print segment
    means, which is what should actually be quoted."""
    x, y = merged(pattern, tag)
    if y is None:
        return
    segs = np.array_split(np.arange(len(y)), n_segments)
    means = [y[s].mean() for s in segs]
    print(f"  {name}: n={len(y)} sd={y.std():.2f} "
          f"min={y.min():.2f} max={y.max():.2f}")
    print(f"      raw first={y[0]:.2f} last={y[-1]:.2f}   "
          f"(endpoints — do NOT quote these)")
    print(f"      segment means: " + "  ".join(f"{m:7.2f}" for m in means))
    print(f"      => early {means[0]:.2f}  ->  late {means[-1]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=sorted(TARGETS),
                    help="which log to plot (or use --pattern)")
    ap.add_argument("--pattern", help="glob for event files; overrides --which")
    ap.add_argument("-o", "--out", help="output PNG")
    ap.add_argument("--title", help="figure title")
    args = ap.parse_args()

    if args.pattern:
        pattern = args.pattern if os.path.isabs(args.pattern) \
            else os.path.join(ROOT, args.pattern)
        out = args.out or os.path.join(ROOT, "results", "ppo_curve.png")
        title = args.title or "PPO training"
    elif args.which:
        spec = TARGETS[args.which]
        pattern = os.path.join(ROOT, spec["pattern"])
        out = args.out or os.path.join(ROOT, spec["out"])
        title = args.title or spec["title"]
    else:
        ap.error("pass --which pacman / --which boss_duel, or --pattern")

    # The TARGETS entries use forward slashes; joining them onto a Windows ROOT
    # yields a mixed-separator path that matplotlib's PNG writer rejects with
    # OSError 22 on some Windows/Pillow combinations. Normalise it.
    out = os.path.normpath(out)

    files = glob.glob(pattern)
    if not files:
        print(f"Pattern matched no event files: {pattern}", file=sys.stderr)
        return 1
    print(f"{len(files)} event file(s) from {os.path.relpath(pattern, ROOT)}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # ---- panel 1: rewards ----
    ax = axes[0]
    for tag, c, lbl in [("eval/mean_reward", C_EVAL, "eval/mean_reward"),
                        ("rollout/ep_rew_mean", C_ROLL, "rollout/ep_rew_mean")]:
        x, y = merged(pattern, tag)
        if x is None:
            continue
        x, y = thin(x, y)
        ax.plot(x, y, color=c, lw=0.8, alpha=0.30)
        ax.plot(x, rolling(y, max(5, len(y) // 12)), color=c, lw=2.2, label=lbl)
    ax.axhline(0, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("reward")
    ax.set_title("Reward — eval vs. rollout\n(faint = raw, bold = rolling mean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- panel 2: episode length ----
    ax = axes[1]
    for tag, c, lbl in [("eval/mean_ep_length", C_EVAL, "eval/mean_ep_length"),
                        ("rollout/ep_len_mean", C_ROLL, "rollout/ep_len_mean")]:
        x, y = merged(pattern, tag)
        if x is None:
            continue
        x, y = thin(x, y)
        ax.plot(x, y, color=c, lw=0.8, alpha=0.30)
        ax.plot(x, rolling(y, max(5, len(y) // 12)), color=c, lw=2.2, label=lbl)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("episode length")
    ax.set_title("Episode length\n(faint = raw, bold = rolling mean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    # The shape of this curve is the most informative thing in the figure: a
    # policy that first learns to *survive* (length rises toward the cap) and
    # only later learns to *win* (length falls again as the boss dies sooner).
    xl, yl = merged(pattern, "eval/mean_ep_length")
    if xl is not None and len(yl) > 12:
        xl, yl = thin(xl, yl)
        sml = rolling(yl, max(5, len(yl) // 12))
        i = int(np.argmax(sml))
        ax.annotate(
            f"peak {sml[i]:.1f} steps @ {xl[i]/1e6:.1f}M\n(learning to survive)",
            xy=(xl[i], sml[i]), xytext=(6, 14), textcoords="offset points",
            fontsize=7.5, color=C_EVAL,
            arrowprops=dict(arrowstyle="->", color=C_EVAL, lw=1))
        ax.annotate(
            f"ends {sml[-1]:.1f}  (learning to win\n=> decisive fights end sooner)",
            xy=(xl[-1], sml[-1]), xytext=(-10, -34), textcoords="offset points",
            fontsize=7.5, color=C_EVAL, ha="right",
            arrowprops=dict(arrowstyle="->", color=C_EVAL, lw=1))

    # ---- panel 3: explained variance ----
    ax = axes[2]
    x, y = merged(pattern, "train/explained_variance")
    if x is not None:
        x, y = thin(x, y)
        sm = rolling(y, max(5, len(y) // 12))
        ax.plot(x, y, color="#55A868", lw=0.8, alpha=0.30)
        ax.plot(x, sm, color="#55A868", lw=2.2)
        ax.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.axhline(0.0, color="grey", lw=0.8, ls=":", alpha=0.6)
        # The raw final value is as unreliable here as anywhere else, so quote
        # the rolling-mean endpoint and the segment mean instead.
        segs = np.array_split(np.arange(len(y)), 5)
        seg_mean = float(np.mean([y[s].mean() for s in segs]))
        ax.text(0.02, 0.97,
                f"rolling-mean end {sm[-1]:.3f}\n"
                f"segment mean {seg_mean:.3f}\n"
                f"(raw end {y[-1]:.3f} — not quotable)",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    ax.set_xlabel("timesteps")
    ax.set_ylabel("explained variance")
    ax.set_title("Value function fit\n(faint = raw, bold = rolling mean)")
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    try:
        fig.savefig(out, dpi=140)
    except OSError as exc:
        plt.close(fig)
        print(f"\n[ERROR] could not write {out}\n        {exc}", file=sys.stderr)
        print("\nMost likely causes:\n"
              "  - the file is open in an image viewer (close it and retry)\n"
              "  - the directory is not writable\n"
              "  - unusual characters in the path\n"
              "Try another output path with:  -o some/other/name.png",
              file=sys.stderr)
        return 1
    plt.close(fig)
    print(f"[saved] {out}")

    print("\nRaw first/last points are unreliable when the eval series is noisy;")
    print("quote the segment means instead:")
    report_segments(pattern, "eval/mean_reward", "eval/mean_reward")
    report_segments(pattern, "rollout/ep_rew_mean", "rollout/ep_rew_mean")
    report_segments(pattern, "eval/mean_ep_length", "eval/mean_ep_length")
    return 0


if __name__ == "__main__":
    sys.exit(main())
