"""
Minimal PPO smoke test against JSBSimEnv.

This does NOT attempt to train a good policy. The only goal is to
confirm that Ray/RLlib can construct the environment, run env_runners,
and complete a few training iterations without crashing.

Run from the magnus/ project root:
    python scripts/test_ppo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray.rllib.algorithms.ppo import PPOConfig
from envs.jsbsim_env import JSBSimEnv


class RLlibJSBSimEnv(JSBSimEnv):
    """
    Thin wrapper so RLlib can construct JSBSimEnv via its expected
    single-`config`-argument constructor, without modifying JSBSimEnv
    itself (which takes explicit named arguments).
    """

    def __init__(self, config=None):
        config = config or {}
        super().__init__(
            aircraft=config.get("aircraft", "c172p"),
            agent_hz=config.get("agent_hz", 20),
        )


def main():
    config = (
        PPOConfig()
        .environment(RLlibJSBSimEnv)
        .env_runners(num_env_runners=1)
    )

    algo = config.build_algo()

    print("Algorithm built successfully. Starting training iterations...\n")

    for i in range(3):
        result = algo.train()
        reward_mean = result.get("env_runners", {}).get("episode_return_mean", None)
        print(f"Iteration {i}: episode_return_mean = {reward_mean}")

    print("\nPASS: PPO ran 3 iterations against JSBSimEnv without crashing.")


if __name__ == "__main__":
    main()