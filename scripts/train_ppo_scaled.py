"""
Scaled-up PPO run against JSBSimEnv, using 8 parallel env runners and
more iterations than the smoke test, to start actually observing
whether episode_return_mean trends in a sensible direction.

This is still NOT a real training run to convergence -- 20 iterations
is enough to see an early trend, not enough to produce a usable policy.
Real training will need far more iterations, likely on the home
machine once the Ray cluster is set up.

Run from the magnus/ project root:
    python scripts/train_ppo_scaled.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray.rllib.algorithms.ppo import PPOConfig
from envs.jsbsim_env import JSBSimEnv

N_ITERATIONS = 20
NUM_ENV_RUNNERS = 8

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"
)
SAVE_EVERY = 5

class RLlibJSBSimEnv(JSBSimEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(
            aircraft=config.get("aircraft", "c172p"),
            agent_hz=config.get("agent_hz", 20),
            departed_start_prob=config.get("departed_start_prob", 0.0),
        )

def main():
    config = (
        PPOConfig()
        .environment(RLlibJSBSimEnv, env_config={"departed_start_prob": 0.5})
        .env_runners(num_env_runners=NUM_ENV_RUNNERS)
    )

    algo = config.build_algo()

    print(f"Algorithm built. Running {N_ITERATIONS} iterations with {NUM_ENV_RUNNERS} env runners...\n")

    returns = []
    for i in range(N_ITERATIONS):
        result = algo.train()
        reward_mean = result.get("env_runners", {}).get("episode_return_mean", None)
        returns.append(reward_mean)
        print(f"Iteration {i}: episode_return_mean = {reward_mean}")
        if (i + 1) % SAVE_EVERY == 0:
            saved = algo.save(os.path.join(CHECKPOINT_DIR, f"iter_{i + 1}"))
            print(f"Checkpoint saved: {saved.checkpoint.path}")

    print("\n--- Summary ---")
    valid_returns = [r for r in returns if r is not None]
    if len(valid_returns) >= 2:
        print(f"First valid return: {valid_returns[0]:.2f}")
        print(f"Last valid return:  {valid_returns[-1]:.2f}")
        print(f"Change: {valid_returns[-1] - valid_returns[0]:.2f}")
    else:
        print("Not enough valid (non-None) returns to compare trend.")

    final = algo.save(os.path.join(CHECKPOINT_DIR, "final"))
    print(f"Final checkpoint saved: {final.checkpoint.path}")

    print("\nPASS: completed all iterations without crashing.")


if __name__ == "__main__":
    main()