"""
Visualize policy vs do-nothing performance: outcome bar chart across many
seeds, plus per-step state traces (AoA, bank, altitude) for a few sample
episodes so you can see recovery behavior directly, not just summary stats.

Run from the project root:
    python scripts/eval_plot.py
    python scripts/eval_plot.py checkpoints/iter_10

Requires matplotlib (add to requirements.txt: matplotlib).
Saves two PNGs into eval_plots/: outcomes.png and traces.png
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
from ray.rllib.core.rl_module.rl_module import RLModule
from envs.jsbsim_env import JSBSimEnv

N_SEEDS = 100
SEED_OFFSET = 1000  # held out from training seeds
N_TRACE_EPISODES = 3  # how many individual episodes to plot in detail
OUT_DIR = os.path.join(ROOT, "eval_plots")

checkpoint = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "checkpoints", "final")
checkpoint = os.path.abspath(checkpoint)
module_path = os.path.join(checkpoint, "learner_group", "learner", "rl_module", "default_policy")

if not os.path.isdir(module_path):
    print(f"ERROR: could not find the policy at {module_path}")
    sys.exit(1)

module = RLModule.from_checkpoint(module_path)


def policy_action(obs):
    with torch.no_grad():
        out = module.forward_inference({"obs": torch.from_numpy(obs).float().unsqueeze(0)})
    dist_inputs = out["action_dist_inputs"][0].numpy()
    action = np.clip(dist_inputs[:4], -1, 1)
    action[3] = (action[3] + 1) / 2
    return action.astype(np.float32)


def do_nothing_action(obs):
    return np.array([0.0, 0.0, 0.0, 0.85], dtype=np.float32)


def run_episode(env, action_fn, seed, record_trace=False):
    obs, _ = env.reset(seed=seed, options={"departed": True})
    total = 0.0
    trace = {"alpha_deg": [], "phi_deg": [], "altitude_ft": []} if record_trace else None
    for step in range(2000):
        if record_trace:
            trace["alpha_deg"].append(env.fdm["aero/alpha-deg"])
            trace["phi_deg"].append(env.fdm["attitude/phi-deg"])
            trace["altitude_ft"].append(env.fdm["position/h-sl-ft"])
        obs, reward, terminated, truncated, info = env.step(action_fn(obs))
        total += reward
        if terminated:
            return info["termination_reason"], total, step + 1, trace
        if truncated:
            return "timeout", total, step + 1, trace
    return "timeout", total, step + 1, trace


def evaluate(env, action_fn):
    seeds = list(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))
    results = [run_episode(env, action_fn, s) for s in seeds]
    reasons = Counter(str(r[0]) for r in results)
    returns = [r[1] for r in results]
    steps = [r[2] for r in results]
    return reasons, np.mean(returns), np.std(returns), np.mean(steps)


def collect_traces(env, action_fn, n_episodes):
    seeds = list(range(SEED_OFFSET, SEED_OFFSET + n_episodes))
    traces = []
    for s in seeds:
        reason, total, steps, trace = run_episode(env, action_fn, s, record_trace=True)
        traces.append((s, reason, trace))
    return traces


os.makedirs(OUT_DIR, exist_ok=True)

# --- Part 1: outcome comparison across many seeds ---
summary = {}
for name, fn in [("policy", policy_action), ("do-nothing", do_nothing_action)]:
    env = JSBSimEnv()  # fresh env per baseline -- avoids state leakage between runs
    reasons, mean_ret, std_ret, mean_steps = evaluate(env, fn)
    summary[name] = reasons
    print(f"RESULT {name}: return_mean={mean_ret:.0f} (std {std_ret:.0f}) "
          f"steps_mean={mean_steps:.0f} outcomes={dict(reasons)}")

all_outcomes = sorted(set().union(*[set(r.keys()) for r in summary.values()]))
x = np.arange(len(all_outcomes))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
for i, (name, reasons) in enumerate(summary.items()):
    counts = [reasons.get(o, 0) for o in all_outcomes]
    ax.bar(x + (i - 0.5) * width, counts, width, label=name)

ax.set_xticks(x)
ax.set_xticklabels(all_outcomes)
ax.set_ylabel(f"count (of {N_SEEDS} episodes)")
ax.set_title(f"Outcomes: policy vs do-nothing ({N_SEEDS} held-out seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "outcomes.png"), dpi=150)
print(f"Saved {os.path.join(OUT_DIR, 'outcomes.png')}")

# --- Part 2: per-step traces for a few sample episodes ---
policy_env = JSBSimEnv()  # fresh env per baseline -- avoids state leakage between runs
baseline_env = JSBSimEnv()
policy_traces = collect_traces(policy_env, policy_action, N_TRACE_EPISODES)
baseline_traces = collect_traces(baseline_env, do_nothing_action, N_TRACE_EPISODES)

fig, axes = plt.subplots(3, N_TRACE_EPISODES, figsize=(5 * N_TRACE_EPISODES, 10), sharex=False)
if N_TRACE_EPISODES == 1:
    axes = axes.reshape(3, 1)

field_labels = [("alpha_deg", "AoA (deg)"), ("phi_deg", "Bank (deg)"), ("altitude_ft", "Altitude (ft)")]

for col, (seed, reason, trace) in enumerate(policy_traces):
    for row, (field, label) in enumerate(field_labels):
        ax = axes[row][col]
        ax.plot(trace[field], label="policy", color="tab:blue")
        b_seed, b_reason, b_trace = baseline_traces[col]
        ax.plot(b_trace[field], label="do-nothing", color="tab:orange", alpha=0.7)
        if row == 0:
            ax.set_title(f"seed {seed}\npolicy={reason} / do-nothing={b_reason}")
        if col == 0:
            ax.set_ylabel(label)
        if row == 2:
            ax.set_xlabel("step")
        if row == 0 and col == 0:
            ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "traces.png"), dpi=150)
print(f"Saved {os.path.join(OUT_DIR, 'traces.png')}")