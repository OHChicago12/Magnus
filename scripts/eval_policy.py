"""
Load a saved PPO checkpoint, run one departed-start episode with the policy
(deterministic) and one with a do-nothing baseline from the SAME starting
conditions, then plot both.

Run from the project root:
    python scripts/eval_policy.py
    python scripts/eval_policy.py checkpoints/iter_10
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ray.rllib.core.rl_module.rl_module import RLModule
from envs.jsbsim_env import JSBSimEnv

SEED = 0
AGENT_HZ = 20

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
    action = np.clip(dist_inputs[:4], -1, 1)  # first 4 numbers = the mean action
    action[3] = (action[3] + 1) / 2           # throttle: [-1, 1] -> [0, 1]
    return action.astype(np.float32)


def do_nothing_action(obs):
    return np.array([0.0, 0.0, 0.0, 0.85], dtype=np.float32)


def run_episode(env, action_fn):
    obs, _ = env.reset(seed=SEED, options={"departed": True})
    trace = {"t": [], "roll": [], "pitch": [], "alpha": [], "airspeed": []}
    total = 0.0
    for step in range(2000):
        obs, reward, terminated, truncated, info = env.step(action_fn(obs))
        total += reward
        trace["t"].append((step + 1) / AGENT_HZ)
        trace["roll"].append(np.degrees(obs[0]))
        trace["pitch"].append(np.degrees(obs[1]))
        trace["alpha"].append(obs[6])
        trace["airspeed"].append(obs[8])
        if terminated or truncated:
            break
    return trace, info["termination_reason"], total, step + 1


env = JSBSimEnv()
policy_trace, p_reason, p_return, p_steps = run_episode(env, policy_action)
base_trace, b_reason, b_return, b_steps = run_episode(env, do_nothing_action)

print(f"RESULT policy:     steps={p_steps} reason={p_reason} return={p_return:.0f}")
print(f"RESULT do-nothing: steps={b_steps} reason={b_reason} return={b_return:.0f}")

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
panels = [
    ("roll", "Bank angle (deg)"),
    ("pitch", "Pitch (deg)"),
    ("alpha", "Angle of attack (deg)"),
    ("airspeed", "Airspeed (kts)"),
]
for ax, (key, label) in zip(axes.flat, panels):
    ax.plot(base_trace["t"], base_trace[key], label="do-nothing", alpha=0.7)
    ax.plot(policy_trace["t"], policy_trace[key], label="policy")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[1][0].set_xlabel("Time (s)")
axes[1][1].set_xlabel("Time (s)")
axes[0][0].legend()
fig.suptitle(f"Departed start, seed {SEED} - {os.path.basename(checkpoint)}")
fig.tight_layout()

out_path = os.path.join(ROOT, "eval_plot.png")
fig.savefig(out_path, dpi=120)
print(f"RESULT plot saved to {out_path}")