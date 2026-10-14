#!/bin/sh
# Regenerate every figure in custom-rl-environments and print the segment means
# that the README quotes. Run from the repository root:
#     sh tools/regen_all.sh
set -e
cd "$(dirname "$0")/.."
PY="${PY:-D:/TheHerta/MUJOCO/mujoco_env/Scripts/python.exe}"
"$PY" tools/plot_dqn.py
"$PY" tools/plot_ppo.py --which pacman
"$PY" tools/plot_ppo.py --which boss_duel
echo "All figures regenerated."
