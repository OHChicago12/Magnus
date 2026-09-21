"""
Random-policy sanity test for JSBSimEnv.

Steps the environment with random actions for a fixed number of steps
and checks for the basic failure modes that should be caught BEFORE
any RL training is attempted: crashes, NaNs, and observations that
blow past their declared bounds.

This test does NOT check whether behavior is good -- random actions
will fly the plane badly on purpose. It only checks that the
environment itself is well-behaved as a piece of software.

Run from the magnus/ project root:
    python scripts/test_random_policy.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.jsbsim_env import JSBSimEnv

N_STEPS = 500


def main():
    env = JSBSimEnv(aircraft="c172p", agent_hz=20)
    obs, _ = env.reset()

    nan_count = 0
    out_of_bounds_count = 0
    terminated_at = None

    for step in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if np.any(np.isnan(obs)):
            nan_count += 1
            print(f"Step {step}: NaN detected in observation: {obs}")

        if np.any(obs < env.observation_space.low) or np.any(obs > env.observation_space.high):
            out_of_bounds_count += 1
            print(f"Step {step}: observation out of declared bounds: {obs}")

        if terminated:
            terminated_at = step
            print(f"Step {step}: episode terminated (sim_time={info['sim_time_s']:.1f}s)")
            break

        if truncated:
            print(f"Step {step}: episode truncated (max steps reached)")
            break

    print("\n--- Summary ---")
    print(f"Steps run: {step + 1} / {N_STEPS}")
    print(f"NaN occurrences: {nan_count}")
    print(f"Out-of-bounds occurrences: {out_of_bounds_count}")
    print(f"Terminated early: {'yes at step ' + str(terminated_at) if terminated_at is not None else 'no'}")

    if nan_count == 0 and out_of_bounds_count == 0:
        print("PASS: environment produced clean observations throughout the run.")
    else:
        print("ATTENTION: environment produced NaNs or out-of-bounds values -- investigate before RL training.")


if __name__ == "__main__":
    main()