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

- `envs/` -- Gymnasium environment wrapping JSBSim (`jsbsim_env.py`)
- `controllers/` -- classical control code (`pid.py`)
- `scripts/` -- standalone test/validation scripts, run directly:
  - `test_pid_hold.py` -- classical PID attitude-hold validation
  - `test_random_policy.py` -- environment sanity check (crashes,
    NaNs, observation bounds) under random actions
  - `test_ppo.py` -- minimal PPO smoke test against the environment
- `docs/` -- design notes (mode switching, Ray cluster setup)

## Running the test scripts

Activate the virtual environment first:
    source venv/bin/activate

Then run any script from the project root, e.g.:
    python scripts/test_random_policy.py

## Setup

Requires a Python virtual environment with: jsbsim, gymnasium,
matplotlib, numpy, ray[rllib], torch. See venv/ (not committed --
create your own with `python3 -m venv venv` and pip install the above).