"""
Robustness test: control (no memory) vs memory8 (last 8 steps) on the unseen planes,
with delayed, noisy, and spiky sensor readings. Both policies get identical
starts and identical noise, so differences come from the policy only.

Run from repo root:  PYTHONPATH=. python scripts/eval_robustness.py
Quick test:          PYTHONPATH=. python scripts/eval_robustness.py --episodes 5
"""
import argparse, collections, json, os, sys
import numpy as np
import gymnasium as gym
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

sys.path.insert(0, os.getcwd())
from train_memory_experiment import HistoryWrapper, HISTORY, ENV_KWARGS, TEST_PLANES
from envs.multi_plane_env import MultiPlaneEnv

CONDITIONS = {
    "clean": {},
    "latency1": {"latency": 1},
    "latency2": {"latency": 2},
    "latency3": {"latency": 3},
    "latency4": {"latency": 4},
    "latency5": {"latency": 5},
    "latency6": {"latency": 6},
    "latency8": {"latency": 8},
    "noise0.3": {"noise": 0.3},
    "noise0.6": {"noise": 0.6},
    "noise1.0": {"noise": 1.0},
    "noise2.0": {"noise": 2.0},
    "noise4.0": {"noise": 4.0},
    "spikes5%": {"spike_p": 0.05},
    "spikes10%": {"spike_p": 0.10},
    "spikes20%": {"spike_p": 0.20},
}


class Perturb(gym.Wrapper):
    """Corrupts the sensor readings the policy sees (not the plane itself)."""

    def __init__(self, env, scale, latency=0, noise=0.0, spike_p=0.0, spike_size=5.0):
        super().__init__(env)
        self.scale = scale
        self.k, self.noise, self.p, self.size = latency, noise, spike_p, spike_size
        self.rng = np.random.default_rng(0)
        self.q = collections.deque(maxlen=latency + 1)

    def _out(self):
        o = np.array(self.q[0], dtype=np.float32)
        if self.noise:
            o += (self.rng.normal(0, self.noise, o.shape) * self.scale).astype(np.float32)
        if self.p and self.rng.random() < self.p:
            i = self.rng.integers(len(o))
            o[i] += self.size * self.scale[i] * self.rng.choice([-1, 1])
        return o

    def reset(self, seed=None, **kw):
        obs, info = self.env.reset(seed=seed, **kw)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.q.clear()
        for _ in range(self.k + 1):
            self.q.append(obs)
        return self._out(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.q.append(obs)
        return self._out(), r, term, trunc, info

def calibrate(module, hist, planes, eval_seed, episodes=3):
    """Typical spread (std) of each sensor channel during clean flights."""
    seen = []
    for plane in planes:
        env = HistoryWrapper(MultiPlaneEnv({**ENV_KWARGS, "planes": [plane]}), hist)
        low, high, ad, od = env.action_space.low, env.action_space.high, env.ad, env.od
        for ep in range(episodes):
            obs, _ = env.reset(seed=eval_seed + ep)
            done = False
            while not done:
                seen.append(obs[-(od + ad):-ad].copy())
                with torch.no_grad():
                    o = module.forward_inference({"obs": torch.from_numpy(obs)[None]})
                obs, _, term, trunc, _ = env.step(np.clip(o["action_dist_inputs"][0].numpy()[:ad], low, high))
                done = term or trunc
    return np.maximum(np.std(seen, axis=0), 1e-3)

def run(module, hist, plane, cond, episodes, eval_seed, scale):
    inner = MultiPlaneEnv({**ENV_KWARGS, "planes": [plane]})
    env = HistoryWrapper(Perturb(inner, scale, **cond), hist)
    low, high, ad = env.action_space.low, env.action_space.high, env.ad
    ok = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=eval_seed + ep)
        done, info = False, {}
        while not done:
            with torch.no_grad():
                o = module.forward_inference({"obs": torch.from_numpy(obs)[None]})
            act = o["action_dist_inputs"][0].numpy()[:ad]
            obs, _, term, trunc, info = env.step(np.clip(act, low, high))
            done = term or trunc
        ok += info.get("termination_reason") == "recovered"
    return int(ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--control", default="checkpoints_memexp/control/iter_1200")
    p.add_argument("--memory8", default="checkpoints_memexp/memory8/iter_1200")
    a = p.parse_args()

    policies = {}
    for name, ckpt in [("control", a.control), ("memory8", a.memory8)]:
        path = os.path.abspath(os.path.join(ckpt, "learner_group/learner/rl_module/default_policy"))
        policies[name] = (RLModule.from_checkpoint(path), HISTORY[name])

    scale = calibrate(policies["control"][0], 1, TEST_PLANES, a.eval_seed)
    print("channel spread:", np.round(scale, 4))

    results = {}
    print(f"{'plane':10s} {'condition':10s} {'control':>8s} {'memory8':>8s}   (recovered / {a.episodes})")
    for plane in TEST_PLANES:
        for cname, cond in CONDITIONS.items():
            row = {n: run(m, h, plane, cond, a.episodes, a.eval_seed, scale) for n, (m, h) in policies.items()}
            results[f"{plane}/{cname}"] = row
            print(f"{plane:10s} {cname:10s} {row['control']:>8d} {row['memory8']:>8d}", flush=True)

    with open("robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTotals across all unseen planes, per condition")
    for cname in CONDITIONS:
        tot = {n: sum(results[f"{pl}/{cname}"][n] for pl in TEST_PLANES) for n in policies}
        print(f"{cname:10s} control {tot['control']:>4d}  memory8 {tot['memory8']:>4d}  (of {a.episodes * len(TEST_PLANES)})")


if __name__ == "__main__":
    main()