# Magnus

An AI flight-control copilot for custom RC aircraft, designed to
assist pilots and help recover control in unusual or unstable flight
regimes -- including high angle-of-attack, stalls, and spins. The
human pilot retains primary authority; Magnus is an assistance and
recovery layer, not a replacement.

## Status

Phase 0 complete. Phase 1 (RL foundation) in progress.

- JSBSim <-> Gymnasium <-> RLlib/PPO pipeline confirmed working
  end-to-end (see scripts/test_ppo.py)
- Domain randomization implemented in envs/jsbsim_env.py's reset(),
  standing in for a physical test fleet
- Termination logic distinguishes crashed / unrecoverable_attitude /
  recovered outcomes, with sustained-streak requirements on both
  departure entry and recovery exit (see envs/jsbsim_env.py)
- Reward function includes per-step AoA/bank shaping, a per-step
  penalty while departed, and large terminal bonuses/penalties tied
  to how the episode ended
- Physical airframe not yet selected/built

See docs/ for design notes, jsbsim-learnings for Phase 0 debugging
history (kept outside this repo, in project notes).

## Architecture

Hybrid classical + learned control: classical PID/LQR for normal
flight, PPO-based RL reserved for post-stall/spin/departure dynamics
where classical methods struggle. Two authority modes: Assisted
(envelope protection always live) and Auto-recovery (Magnus takes
primary control during a departure). See docs/mode_switching_design.md
for the full design.

## Repo layout

- `envs/` -- Gymnasium environments wrapping JSBSim: `jsbsim_env.py`
  (base env), `rc_env.py` (airspeed corruption, smoothness penalty),
  `airframe_dr.py` (randomized airframe XML per reset)
- `controllers/` -- classical control code (`pid.py`)
- `scripts/` -- run directly from the project root:
  - Training: `train_ppo_scaled.py`, `train_ppo_rc.py`
  - Evaluation: `eval_visualize.py`, `eval_policy.py`, `eval_many.py`,
    `eval_plot.py`, `eval_lqr.py`, `check_paired.py`, `replay_fail.py`
  - LQR baseline: `lqr_find_trim.py`, `lqr_compute_gain.py`
  - Tests: `test_pid_hold.py`, `test_random_policy.py`, `test_ppo.py`,
    `test_departed_start.py`, `test_departed_episode.py`
- `docs/` -- design notes (mode switching, Ray cluster setup)
- `logs/` -- `training/` and `eval/` text logs
- `plots/` -- figures; per-run output folders are in `plots/runs/`
- `archive/` -- applied patch scripts, probe scripts, old script backups
- `checkpoints*/` -- trained policies (not committed; copy manually)

## Running the test scripts

Activate the virtual environment first:
    source venv/bin/activate

Then run any script from the project root, e.g.:
    python scripts/test_random_policy.py

## Setup

Requires a Python virtual environment with: jsbsim, gymnasium,
matplotlib, numpy, ray[rllib], torch. Create your own environment with
`python3 -m venv venv`, then `pip install -r requirements.txt`.
(`venv/` is not committed.)