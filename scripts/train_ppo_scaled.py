"""
Medium-length PPO run against JSBSimEnv, using 8 parallel env runners.

Bumped up from the 20-iteration smoke test to actually see whether
episode_return_mean trends in a sensible direction over a longer run.
Still not a full training-to-convergence run -- that's the multi-day
job planned for the home machine + Ray cluster once that's validated.

Run from the magnus/ project root:
    python scripts/train_ppo_scaled.py

To resume a run that got interrupted (crash, reboot, closed lid):
    python scripts/train_ppo_scaled.py --resume checkpoints/iter_150

To survive your Mac going to sleep during a long run, wrap it in caffeinate:
    caffeinate -i python scripts/train_ppo_scaled.py
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from envs.jsbsim_env import JSBSimEnv

N_ITERATIONS = 300
NUM_ENV_RUNNERS = 8

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"
)
SAVE_EVERY = 10  # more frequent than the original 5-iteration smoke test's
                 # SAVE_EVERY=5, since a longer run has more to lose per crash

class RLlibJSBSimEnv(JSBSimEnv):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(
            aircraft=config.get("aircraft", "c172p"),
            agent_hz=config.get("agent_hz", 20),
            departed_start_prob=config.get("departed_start_prob", 0.0),
        )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint dir to resume from (e.g. checkpoints/iter_150). "
             "Omit to start a fresh run."
    )
    parser.add_argument(
        "--iterations", type=int, default=N_ITERATIONS,
        help=f"Total iterations to run (default {N_ITERATIONS})"
    )
    parser.add_argument(
        "--checkpoint-dir", default=CHECKPOINT_DIR,
        help=f"Directory to save checkpoints under (default {CHECKPOINT_DIR}). "
             "Use a fresh directory (e.g. checkpoints_v2) when the reward "
             "function or env changed, so old and new checkpoints don't mix."
    )
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        algo = Algorithm.from_checkpoint(os.path.abspath(args.resume))
        # iteration count so far is tracked by the algo itself
        start_iter = algo.iteration
    else:
        config = (
            PPOConfig()
            .environment(RLlibJSBSimEnv, env_config={"departed_start_prob": 0.5})
            .env_runners(num_env_runners=NUM_ENV_RUNNERS)
        )
        algo = config.build_algo()
        start_iter = 0

    print(f"Starting from iteration {start_iter}. Running to iteration {args.iterations} "
          f"with {NUM_ENV_RUNNERS} env runners...\n")

    returns = []
    try:
        for i in range(start_iter, args.iterations):
            result = algo.train()
            reward_mean = result.get("env_runners", {}).get("episode_return_mean", None)
            returns.append(reward_mean)
            print(f"Iteration {i}: episode_return_mean = {reward_mean}")
            if (i + 1) % SAVE_EVERY == 0:
                saved = algo.save(os.path.join(checkpoint_dir, f"iter_{i + 1}"))
                print(f"Checkpoint saved: {saved.checkpoint.path}")
    except KeyboardInterrupt:
        # Save on manual interrupt (Ctrl-C) so a deliberate stop doesn't lose
        # progress since the last SAVE_EVERY boundary.
        print("\nInterrupted -- saving checkpoint before exit...")
        saved = algo.save(os.path.join(checkpoint_dir, f"iter_{start_iter + len(returns)}_interrupted"))
        print(f"Checkpoint saved: {saved.checkpoint.path}")
        return

    print("\n--- Summary ---")
    valid_returns = [r for r in returns if r is not None]
    if len(valid_returns) >= 2:
        print(f"First valid return: {valid_returns[0]:.2f}")
        print(f"Last valid return:  {valid_returns[-1]:.2f}")
        print(f"Change: {valid_returns[-1] - valid_returns[0]:.2f}")
    else:
        print("Not enough valid (non-None) returns to compare trend.")

    final = algo.save(os.path.join(checkpoint_dir, "final"))
    print(f"Final checkpoint saved: {final.checkpoint.path}")

    print("\nPASS: completed all iterations without crashing.")


if __name__ == "__main__":
    main()