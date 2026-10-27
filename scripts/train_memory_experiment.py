"""
Multi-plane memory experiment.

Train on c172p_rc, c182, pa28. Test on J3Cub, c310, t6texan2 (never seen in training).
Two runs, identical except for memory:
  control: policy sees current obs + the action that led to it (history = 1)
  memory : policy sees the last 32 (obs, action) pairs (history = 32)

Usage (from repo root):
  PYTHONPATH=. python scripts/train_memory_experiment.py --run both
  PYTHONPATH=. python scripts/train_memory_experiment.py --run memory --iterations 50 --episodes 10   # smoke test
"""
import argparse, collections, json, os, sys
import numpy as np
import gymnasium as gym
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, os.getcwd())
from envs.multi_plane_env import MultiPlaneEnv

TRAIN_PLANES = ["c172p_rc", "c182", "pa28", "pc7"]
TEST_PLANES = ["J3Cub", "c310", "t6texan2"]
HISTORY = {"control": 1, "memory": 32, "memory8": 8}

# Copy the rest of your v4 settings from train_ppo_rc.py (servo lag, spawn ranges, etc.)
ENV_KWARGS = dict(departed_start_prob=1.0)


class HistoryWrapper(gym.Wrapper):
    """Plain rolling buffer of the last n steps. Each step = obs + the action that produced it.
    Oldest first; empty slots at episode start are zeros."""

    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        self.od = env.observation_space.shape[0]
        self.ad = env.action_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n * (self.od + self.ad),), np.float32)
        self.buf = collections.deque(maxlen=n)

    def _flat(self):
        return np.concatenate(list(self.buf)).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.buf.clear()
        for _ in range(self.n - 1):
            self.buf.append(np.zeros(self.od + self.ad, np.float32))
        self.buf.append(np.concatenate([obs, np.zeros(self.ad)]).astype(np.float32))
        return self._flat(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.buf.append(np.concatenate([obs, np.asarray(action).ravel()]).astype(np.float32))
        return self._flat(), r, term, trunc, info

SCALE = [0.6356, 0.1794, 2.6394, 2.0373, 1.162, 0.9199, 6.2644, 16.4452, 7.3381]  # channel spread from your sweep


class RandomPerturb(gym.Wrapper):
    """Each episode: delay readings by 0..max_latency steps and add noise up to max_noise x channel spread."""

    def __init__(self, env, max_latency=2, max_noise=1.0):
        super().__init__(env)
        self.scale = np.asarray(SCALE, np.float32)
        self.max_lat, self.max_noise = max_latency, max_noise
        self.rng = np.random.default_rng()
        self.q = collections.deque(maxlen=max_latency + 1)
        self.k, self.noise = 0, 0.0

    def _out(self):
        o = np.array(self.q[-1 - self.k], np.float32)
        return o + (self.rng.normal(0, self.noise, o.shape) * self.scale).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        if kw.get("seed") is not None:
            self.rng = np.random.default_rng(kw["seed"])
        self.k = int(self.rng.integers(0, self.max_lat + 1))
        self.noise = float(self.rng.uniform(0, self.max_noise))
        self.q.clear()
        for _ in range(self.max_lat + 1):
            self.q.append(obs)
        return self._out(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.q.append(obs)
        return self._out(), r, term, trunc, info

def make_env(cfg):
    inner = MultiPlaneEnv({**ENV_KWARGS, "planes": cfg["planes"]})
    if cfg.get("perturb"):
        inner = RandomPerturb(inner, **cfg["perturb"])
    return HistoryWrapper(inner, cfg["history"])


register_env("magnus_multi", make_env)


def train(name, args):
    hist = HISTORY[name]
    config = (
        PPOConfig()
        .environment("magnus_multi", env_config={
            "planes": TRAIN_PLANES,
            "history": hist,
            "perturb": {"max_latency": 2, "max_noise": 1.0} if args.perturb else None,
        })
        .env_runners(num_env_runners=args.workers)
        .training(train_batch_size=args.batch, lr=3e-4)
        .rl_module(model_config={"fcnet_hiddens": [256, 256]})
        .debugging(seed=args.seed)
    )
    algo = config.build()
    ckpt_dir = os.path.abspath(f"checkpoints_memexp/{name}{args.tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for i in range(1, args.iterations + 1):
        res = algo.train()
        if i % 20 == 0:
            ret = res.get("env_runners", {}).get("episode_return_mean")
            print(f"[{name}] iter {i} return {ret}", flush=True)
        if i % 100 == 0 or i == args.iterations:
            algo.save_to_path(os.path.join(ckpt_dir, f"iter_{i}"))
    return algo


def evaluate(algo, name, planes, args):
    module = algo.get_module()
    hist = HISTORY[name]
    out = {}
    for plane in planes:
        env = make_env({"planes": [plane], "history": hist,
                "perturb": {"max_latency": 2, "max_noise": 1.0} if args.perturb else None})
        low, high = env.action_space.low, env.action_space.high
        ad = env.ad
        counts = collections.Counter()
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.eval_seed + ep)  # same spawns for both runs
            done, info = False, {}
            while not done:
                with torch.no_grad():
                    o = module.forward_inference({"obs": torch.from_numpy(obs)[None]})
                mean = o["action_dist_inputs"][0].numpy()[:ad]
                obs, _, term, trunc, info = env.step(np.clip(mean, low, high))
                done = term or trunc
            reason = info.get("termination_reason") or ("truncated" if trunc else "unknown")
            counts[reason] += 1
        out[plane] = {"recovered": counts["recovered"], "n": args.episodes, "outcomes": dict(counts)}
        print(f"[{name}] {plane}: {counts['recovered']}/{args.episodes} recovered  {dict(counts)}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["control", "memory", "memory8", "both"], default="both")
    p.add_argument("--iterations", type=int, default=1200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=8000)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    ray.init(ignore_reinit_error=True)
    runs = ["control", "memory", "memory8"] if args.run == "both" else [args.run]
    results = {}
    for name in runs:
        algo = train(name, args)
        results[name] = {
            "train_planes": evaluate(algo, name, TRAIN_PLANES, args),
            "test_planes": evaluate(algo, name, TEST_PLANES, args),
        }
        algo.stop()
        with open("memexp_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\nRecovered / n (test planes = never seen in training)")
    print(f"{'plane':10s} " + " ".join(f"{r:>10s}" for r in runs))
    for group in ["train_planes", "test_planes"]:
        for plane in results[runs[0]][group]:
            row = " ".join(f"{results[r][group][plane]['recovered']:>6d}/{args.episodes:<3d}" for r in runs)
            print(f"{plane:10s} {row}  ({group.split('_')[0]})")


if __name__ == "__main__":
    main()
