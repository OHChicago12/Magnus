"""
eval_visualize.py — Evaluate and compare multiple Magnus PPO checkpoints.

Usage (CLI):
    python eval_visualize.py --checkpoints checkpoints/iter_20 checkpoints/iter_60 \
        --episodes 10 --out plots/

Usage (notebook):
    from eval_visualize import evaluate_checkpoints, plot_comparison
    results = evaluate_checkpoints(["checkpoints/iter_20", "checkpoints/iter_60"], n_episodes=10)
    plot_comparison(results, out_dir="plots/")

Assumes:
    - envs/jsbsim_env.py defines JSBSimEnv with the info dict populated as in
      the version updated alongside this script (altitude_ft, reward_components).
    - Checkpoints are Ray RLlib PPO checkpoints saved via algo.save(), matching
      train_ppo_scaled.py (edit `load_policy` if using a different framework).
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Silence Ray/RLlib's own console spam (env worker startup logs, deprecation
# warnings) so only this script's own print statements show up.
os.environ.setdefault("RAY_DEDUP_LOGS", "1")

# Add project root to sys.path so `envs.jsbsim_env` is importable regardless
# of where this script is invoked from (mirrors train_ppo_scaled.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Project-specific imports ---
from envs.jsbsim_env import JSBSimEnv  # noqa: E402

# Ray/RLlib is optional at import time so this file can be inspected/tested
# without a full Ray install.
try:
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.columns import Columns
    import ray
    ray.init(log_to_driver=False, logging_level="ERROR", include_dashboard=False)
    _HAS_RLLIB = True
except ImportError:
    _HAS_RLLIB = False


# Keys we track per timestep across every episode. "pitch"/"roll" come from
# obs (converted rad -> deg); "airspeed"/"aoa" come straight from obs (already
# in kts/deg); "altitude" isn't in the observation, so it's read from info.
STATE_KEYS = ["pitch", "roll", "airspeed", "aoa", "altitude"]
# Must match the keys in JSBSimEnv._compute_reward's returned components dict.
REWARD_KEYS = ["alpha_shaping", "bank_shaping", "departure_penalty", "terminal_bonus"]

# obs layout, per JSBSimEnv: [phi, theta, psi, p, q, r, alpha, beta, vc]
OBS_PHI_IDX = 0
OBS_THETA_IDX = 1
OBS_ALPHA_IDX = 6
OBS_VC_IDX = 8


def load_policy(checkpoint_path):
    """Load an RLlib PPO policy from a checkpoint directory."""
    if not _HAS_RLLIB:
        raise RuntimeError(
            "ray[rllib] not installed. `pip install 'ray[rllib]'` or adapt "
            "load_policy() to your checkpoint format."
        )
    # Algorithm.from_checkpoint hands the path to pyarrow's FileSystem.from_uri,
    # which errors on relative paths ("URI has empty scheme") -- resolve to an
    # absolute path first so both relative and absolute inputs work.
    checkpoint_path = os.path.abspath(checkpoint_path)
    algo = Algorithm.from_checkpoint(checkpoint_path)
    return algo


def get_action(module, obs, deterministic=True):
    """Compute one action from an RLModule (new RLlib API stack) for a
    single observation. Replaces the deprecated
    Algorithm.compute_single_action / get_policy path, which no longer
    exists on this RLlib version.
    """
    obs_batch = {Columns.OBS: torch.from_numpy(obs).float().unsqueeze(0)}
    fwd_out = module.forward_inference(obs_batch)
    action_dist_cls = module.get_inference_action_dist_cls()
    action_dist = action_dist_cls.from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
    if deterministic:
        action_dist = action_dist.to_deterministic()
    action = action_dist.sample()[0].detach().numpy()
    return action


def rollout_episode(module, env, deterministic=True, max_steps=2000):
    """Run one episode, returning a dict of per-step traces plus outcome info.

    Reads pitch/roll/airspeed/aoa from `obs` (per JSBSimEnv's fixed layout),
    altitude and reward_components from `info` (added in jsbsim_env.py).
    """
    obs, info = env.reset()
    trace = {k: [] for k in STATE_KEYS}
    reward_trace = {k: [] for k in REWARD_KEYS}
    total_reward = 0.0
    termination_reason = "truncated"

    for _ in range(max_steps):
        action = get_action(module, obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        trace["roll"].append(np.rad2deg(obs[OBS_PHI_IDX]))
        trace["pitch"].append(np.rad2deg(obs[OBS_THETA_IDX]))
        trace["aoa"].append(obs[OBS_ALPHA_IDX])
        trace["airspeed"].append(obs[OBS_VC_IDX])
        trace["altitude"].append(info.get("altitude_ft", np.nan))

        rc = info.get("reward_components", {})
        for k in REWARD_KEYS:
            reward_trace[k].append(rc.get(k, 0.0))

        if terminated or truncated:
            if terminated:
                termination_reason = info.get("termination_reason") or "recovered"
            else:
                termination_reason = "truncated"
            break

    return {
        "trace": trace,
        "reward_trace": reward_trace,
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "length": len(trace[STATE_KEYS[0]]),
    }


def evaluate_checkpoints(checkpoint_paths, n_episodes=10, seed=0):
    """Run n_episodes for each checkpoint. Returns {checkpoint_path: {...}}."""
    results = {}
    for ckpt in checkpoint_paths:
        print(f"Evaluating checkpoint: {ckpt}")
        algo = load_policy(ckpt)
        module = algo.get_module()  # default policy's RLModule
        env = JSBSimEnv()

        episodes = []
        for ep in range(n_episodes):
            env.reset(seed=seed + ep)
            episodes.append(rollout_episode(module, env))

        outcome_counts = defaultdict(int)
        for ep in episodes:
            outcome_counts[ep["termination_reason"]] += 1

        results[ckpt] = {
            "episodes": episodes,
            "outcome_counts": dict(outcome_counts),
            "n_episodes": n_episodes,
        }
        print(f"  outcomes: {dict(outcome_counts)}")

        algo.stop()
        env.close()

    return results


def _best_worst_episode(episodes):
    by_reward = sorted(episodes, key=lambda e: e["total_reward"])
    return by_reward[-1], by_reward[0]  # best, worst


def plot_comparison(results, out_dir="plots"):
    """Generate and save two summary figures instead of one-per-checkpoint:
    an outcome bar chart, and a single overlaid trajectory plot (worst
    episode per checkpoint) so there's a small, fixed number of files to
    look through regardless of how many checkpoints you pass in."""
    os.makedirs(out_dir, exist_ok=True)
    figures = {}

    # 1. Outcome summary bar chart across checkpoints
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(results.keys())
    reasons = sorted({r for res in results.values() for r in res["outcome_counts"]})
    x = np.arange(len(labels))
    width = 0.8 / max(len(reasons), 1)
    for i, reason in enumerate(reasons):
        counts = [results[ck]["outcome_counts"].get(reason, 0) for ck in labels]
        ax.bar(x + i * width, counts, width, label=reason)
    ax.set_xticks(x + width * (len(reasons) - 1) / 2)
    ax.set_xticklabels([os.path.basename(l) for l in labels], rotation=20, ha="right")
    ax.set_ylabel("Episode count")
    ax.set_title("Outcome distribution by checkpoint")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "outcome_summary.png"), dpi=150)
    figures["outcome_summary"] = fig

    # 2. One overlaid trajectory plot: worst episode per checkpoint, all on
    # the same axes per state variable, so checkpoints are directly comparable.
    fig, axes = plt.subplots(len(STATE_KEYS), 1, figsize=(9, 11), sharex=True)
    for ck, res in results.items():
        _, worst = _best_worst_episode(res["episodes"])
        t = np.arange(worst["length"])
        label = f"{os.path.basename(ck)} ({worst['termination_reason']})"
        for ax, key in zip(axes, STATE_KEYS):
            ax.plot(t, worst["trace"][key], label=label)
            ax.set_ylabel(key)
            ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("Worst episode per checkpoint, overlaid")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "worst_episode_comparison.png"), dpi=150)
    figures["worst_episode_comparison"] = fig

    print(f"Saved {len(figures)} figures to {out_dir}/")
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="One or more checkpoint directory paths to compare")
    parser.add_argument("--episodes", type=int, default=10,
                         help="Episodes to roll out per checkpoint")
    parser.add_argument("--out", default="plots", help="Output directory for PNGs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = evaluate_checkpoints(args.checkpoints, n_episodes=args.episodes, seed=args.seed)
    plot_comparison(results, out_dir=args.out)


if __name__ == "__main__":
    main()