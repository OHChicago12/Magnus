"""
Stage 1 of the LQR baseline: find a steady trim point (straight-and-level
flight) by running the existing PID hold controller for a while and letting
it settle. We'll linearize around this trim point in stage 2.

Run from the project root:
    python scripts/lqr_find_trim.py

Saves the trim state/action to trim_point.npz for stage 2 to load.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from envs.jsbsim_env import JSBSimEnv
from controllers.pid import PID

# Softer gains than the Phase 0 hold-test values -- the rate limiter below
# adds lag to the loop, so the original gains cause a persistent gentle
# oscillation instead of settling. This script only needs a quiet trim
# point, not fast tracking, so gentler gains are fine here.
roll_pid = PID(kp=0.5, ki=0.03, kd=0.15)
pitch_pid = PID(kp=1.2, ki=0.08, kd=0.5)
# New: rudder holds sideslip near zero so it (not aileron alone) counters the
# c172p's P-factor/torque roll tendency -- see jsbsim-learnings.md "roll-off"
# entry. Gains are a rough starting guess, not tuned.
rudder_pid = PID(kp=0.05, ki=0.01, kd=0.02)

env = JSBSimEnv(
    initial_altitude_ft=5000,
    initial_airspeed_kts=120,
    altitude_randomization_ft=0,
    airspeed_randomization_kts=0,
    attitude_randomization_deg=0,
    wind_randomization_fps=0,
)

obs, _ = env.reset(seed=0, options={"departed": False})

dt_agent = 1.0 / 20  # matches agent_hz default
N_SETTLE_STEPS = 1000  # ~50s -- softer gains need more time to settle

ROLL_SETPOINT_DEG = 0.0
PITCH_SETPOINT_DEG = 0.0
THROTTLE = 0.85

history = []
prev_action = np.array([0.0, 0.0, 0.0, THROTTLE], dtype=np.float32)
MAX_DELTA_PER_STEP = 0.05  # rate limit -- prevents bang-bang oscillation between saturation limits

for step in range(N_SETTLE_STEPS):
    phi_deg = np.rad2deg(obs[0])
    theta_deg = np.rad2deg(obs[1])
    beta_deg = obs[7]

    aileron_raw = roll_pid.update(ROLL_SETPOINT_DEG, phi_deg, dt_agent)
    elevator_raw = -pitch_pid.update(PITCH_SETPOINT_DEG, theta_deg, dt_agent)  # sign flip, see jsbsim-learnings.md
    rudder_raw = rudder_pid.update(0.0, beta_deg, dt_agent)
    raw_action = np.array([aileron_raw, elevator_raw, rudder_raw, THROTTLE], dtype=np.float32)

    # Rate-limit each channel relative to the previous commanded action.
    delta = np.clip(raw_action - prev_action, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)
    action = prev_action + delta
    prev_action = action.copy()

    obs, reward, terminated, truncated, info = env.step(action)
    history.append((phi_deg, theta_deg, action.copy(), obs.copy()))
    if terminated or truncated:
        print(f"WARNING: episode ended early at step {step} ({info.get('termination_reason')})")
        break

print("Last 10 steps (phi_deg, theta_deg, [aileron, elevator, rudder, throttle]):")
for phi_deg, theta_deg, action, _ in history[-10:]:
    print(f"  phi={phi_deg:7.3f}  theta={theta_deg:7.3f}  action={np.round(action, 3)}")

# Average state/action over the last full oscillation cycle (~10 steps observed)
# rather than a single snapshot, to filter out the residual lightly-damped
# oscillation instead of chasing a perfectly still trim point.
AVERAGE_OVER_STEPS = 100
recent = history[-AVERAGE_OVER_STEPS:]
x0 = np.mean([r[3] for r in recent], axis=0)
u0 = np.mean([r[2] for r in recent], axis=0)

print("Trim state (x0):")
labels = ["phi_rad", "theta_rad", "psi_rad", "p", "q", "r", "alpha_deg", "beta_deg", "vc_kts"]
for label, val in zip(labels, x0):
    print(f"  {label}: {val:.4f}")

print("Trim action (u0):")
for label, val in zip(["aileron", "elevator", "rudder", "throttle"], u0):
    print(f"  {label}: {val:.4f}")

np.savez(os.path.join(ROOT, "trim_point.npz"), x0=x0, u0=u0)
print(f"Saved trim point to {os.path.join(ROOT, 'trim_point.npz')}")