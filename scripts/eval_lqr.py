"""
Stage 3 of the LQR baseline: run the LQR controller through the same
departed-start evaluation as eval_many.py, alongside the PPO policy and
do-nothing baseline for a direct three-way comparison.

Run from the project root:
    python scripts/eval_lqr.py
    python scripts/eval_lqr.py checkpoints/iter_10
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from collections import Counter
import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from envs.jsbsim_env import JSBSimEnv

N_SEEDS = 100
SEED_OFFSET = 1000  # same held-out seeds used by eval_many.py

checkpoint = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "checkpoints", "final")
checkpoint = os.path.abspath(checkpoint)
module_path = os.path.join(checkpoint, "learner_group", "learner", "rl_module", "default_policy")

if not os.path.isdir(module_path):
    print(f"ERROR: could not find the policy at {module_path}")
    sys.exit(1)

module = RLModule.from_checkpoint(module_path)

lqr = np.load(os.path.join(ROOT, "lqr_gains.npz"))
lqr_x0 = lqr["x0"]
lqr_u0 = lqr["u0"]
lqr_K = lqr["K"]


def policy_action(obs):
    with torch.no_grad():
        out = module.forward_inference({"obs": torch.from_numpy(obs).float().unsqueeze(0)})
    dist_inputs = out["action_dist_inputs"][0].numpy()
    action = np.clip(dist_inputs[:4], -1, 1)
    action[3] = (action[3] + 1) / 2
    return action.astype(np.float32)


def do_nothing_action(obs):
    return np.array([0.0, 0.0, 0.0, 0.85], dtype=np.float32)


def lqr_action(obs):
    dx = obs - lqr_x0
    # Heading (index 2) can wrap around a full circle during a departure;
    # the LQR gain on it is ~0 by design (Q weight 0), but wrap it anyway
    # so a huge raw difference can't sneak through via any small coupling term.
    dx[2] = (dx[2] + np.pi) % (2 * np.pi) - np.pi

    u = lqr_u0 - lqr_K @ dx
    aileron = np.clip(u[0], -1, 1)
    elevator = np.clip(u[1], -1, 1)
    rudder = np.clip(u[2], -1, 1)
    throttle = np.clip(u[3], 0.0, 1.0)
    return np.array([aileron, elevator, rudder, throttle], dtype=np.float32)


def run_episode(env, action_fn, seed):
    obs, _ = env.reset(seed=seed, options={"departed": True})
    total = 0.0
    for step in range(2000):
        obs, reward, terminated, truncated, info = env.step(action_fn(obs))
        total += reward
        if terminated:
            return info["termination_reason"], total, step + 1
        if truncated:
            return "timeout", total, step + 1
    return "timeout", total, step + 1


def evaluate(env, action_fn):
    seeds = range(SEED_OFFSET, SEED_OFFSET + N_SEEDS)
    results = [run_episode(env, action_fn, s) for s in seeds]
    reasons = Counter(str(r[0]) for r in results)
    returns = [r[1] for r in results]
    steps = [r[2] for r in results]
    return reasons, np.mean(returns), np.std(returns), np.mean(steps)


for name, fn in [("do-nothing", do_nothing_action), ("policy", policy_action), ("lqr", lqr_action)]:
    env = JSBSimEnv()  # fresh env per baseline -- avoids state leakage between runs (see chat history)
    reasons, mean_ret, std_ret, mean_steps = evaluate(env, fn)
    print(f"RESULT {name}: return_mean={mean_ret:.0f} (std {std_ret:.0f}) "
          f"steps_mean={mean_steps:.0f} outcomes={dict(reasons)}")