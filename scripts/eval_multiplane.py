"""eval_multiplane.py - test Magnus checkpoints on several planes, each rescaled
to RC size (see envs/multi_plane_env.py). Reuses eval_visualize.py, which stays untouched.

Usage (from the repo root):
    python scripts/eval_multiplane.py --checkpoints checkpoints/<your_v4>/iter_XXX \
        --planes c172p_rc c182 pa28 J3Cub c310 --episodes 50

Add --baseline zero to ignore the policy and send zero controls (a "do nothing" floor).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_visualize as ev  # noqa: E402  (reuses load_policy / rollout_episode)
from envs.multi_plane_env import MultiPlaneEnv  # noqa: E402

# RC-plane settings, same as eval_visualize.py's `--rc similar` preset
# (60 Hz, RC-sized speeds and altitude). Override with --env-kwargs.
RC_BASE = dict(
    agent_hz=60,
    initial_altitude_ft=500, altitude_randomization_ft=100,
    initial_airspeed_kts=30, airspeed_randomization_kts=4,
    departed_airspeed_range_kts=(12, 18),
    departed_yaw_rate_range=(1.5, 3.0),
    wind_randomization_fps=4,
)


def run_plane(module, plane, n_episodes, seed, departed_prob, env_kwargs):
    cfg = {"planes": [plane], "departed_start_prob": departed_prob, **RC_BASE, **env_kwargs}
    env = MultiPlaneEnv(cfg)
    episodes = []
    for i in range(n_episodes):
        env.reset(seed=seed + i)
        episodes.append(ev.rollout_episode(module, env))
    tau = env.taus[plane]
    env.close()
    return episodes, tau


def summarize(episodes, tau):
    departed = [e for e in episodes if e["started_departed"]]
    recovered = [e for e in departed if e["termination_reason"] == "recovered"]
    times = [e["recovery_time_steps"] for e in recovered if e["recovery_time_steps"] is not None]
    # altitude lost is divided by tau^2 so planes are comparable in RC-sized feet
    alt = [(e["trace"]["altitude"][0] - min(e["trace"]["altitude"])) / tau ** 2 for e in recovered]
    pitch = [min(e["trace"]["pitch"]) for e in recovered]
    outcomes = defaultdict(int)
    for e in episodes:
        outcomes[e["termination_reason"]] += 1
    med = lambda x: float(np.median(x)) if x else None
    return {
        "departed": len(departed), "recovered": len(recovered),
        "t_med": med(times), "alt_med": med(alt), "pitch_med": med(pitch),
        "outcomes": dict(outcomes),
    }


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--planes", nargs="+", default=["c172p_rc", "c182", "pa28", "J3Cub", "c310"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--departed-start-prob", type=float, default=1.0,
                    help="fraction of episodes that spawn already departed (default 1.0)")
    ap.add_argument("--env-kwargs", default="{}", help="extra env settings as JSON")
    ap.add_argument("--baseline", choices=["zero"], default=None)
    args = ap.parse_args()
    ev.BASELINE = args.baseline
    env_kwargs = json.loads(args.env_kwargs)

    for ckpt in args.checkpoints:
        print(f"\nCheckpoint: {ckpt}")
        algo = ev.load_policy(ckpt)
        module = algo.get_module()
        rows = {}
        for plane in args.planes:
            eps, tau = run_plane(module, plane, args.episodes, args.seed,
                                 args.departed_start_prob, env_kwargs)
            rows[plane] = summarize(eps, tau)
        algo.stop()

        print(f"{'plane':9} {'recovered':>11} {'rec_steps':>10} {'alt_lost_ft':>12} {'min_pitch':>10}  outcomes")
        for plane, r in rows.items():
            print(f"{plane:9} {r['recovered']:>5}/{r['departed']:<5} {fmt(r['t_med'], '.0f'):>10} "
                  f"{fmt(r['alt_med'], '.0f'):>12} {fmt(r['pitch_med'], '.0f'):>10}  {r['outcomes']}")


if __name__ == "__main__":
    main()