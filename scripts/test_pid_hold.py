"""
Phase 0 validation script.

Runs a stock JSBSim aircraft (c172p) with two independent PID loops
holding a commanded roll and pitch angle. This is a plumbing test only:
no aerodynamic tuning for your actual airframe, no envelope protection,
no mode switching. The goal is just to confirm the env <-> controller
loop behaves sanely before you build anything Magnus-specific on top.

Run from the magnus/ project root:
    python scripts/test_pid_hold.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from envs.jsbsim_env import JSBSimEnv
from controllers.pid import PID

AGENT_HZ = 20
DT = 1.0 / AGENT_HZ

ROLL_SETPOINT_DEG = 15.0
PITCH_SETPOINT_DEG = 5.0
DURATION_S = 30


def main():
    env = JSBSimEnv(aircraft="c172p", agent_hz=AGENT_HZ)
    obs, _ = env.reset()

    roll_pid = PID(kp=1.2, ki=0.1, kd=0.3, output_limits=(-1, 1))
    pitch_pid = PID(kp=3.0, ki=0.25, kd=1.0, output_limits=(-1, 1))
    print("prev_measurement attr exists:", hasattr(pitch_pid, "_prev_measurement"))

    roll_sp = np.deg2rad(ROLL_SETPOINT_DEG)
    pitch_sp = np.deg2rad(PITCH_SETPOINT_DEG)

    history = {"t": [], "phi": [], "theta": [], "psi": [], "beta": [], "vc": [], "elevator": []}

    n_steps = int(DURATION_S * AGENT_HZ)
    for i in range(n_steps):
        t_now = i * DT
        phi, theta = obs[0], obs[1]
        psi = obs[2]
        beta = obs[7]

        aileron = roll_pid.update(roll_sp, phi, DT)

        ramp_duration_s = 1.0
        ramp_frac = min(t_now / ramp_duration_s, 1.0)
        pitch_sp_ramped = ramp_frac * pitch_sp

        # NOTE: elevator sign convention (nose-up = negative or positive
        # command) depends on the aircraft's control surface definition.
        # Verify against your model rather than trusting this blindly --
        # if pitch runs away from the setpoint instead of toward it, flip this sign.
        elevator = -pitch_pid.update(pitch_sp_ramped, theta, DT)

        action = np.array([aileron, elevator, 0.0, 0.85])  # fixed throttle for now
        obs, reward, terminated, truncated, info = env.step(action)

        history["t"].append(info["sim_time_s"])
        history["phi"].append(np.rad2deg(obs[0]))
        history["theta"].append(np.rad2deg(obs[1]))
        history["vc"].append(obs[8])
        history["elevator"].append(elevator)
        history["psi"].append(np.rad2deg(psi))
        history["beta"].append(beta)  # beta is already in degrees per the obs docstring

        if terminated:
            print(f"Terminated early at t={info['sim_time_s']:.1f}s -- check bounds/tuning")
            break

    fig, axes = plt.subplots(6, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(history["t"], history["phi"], label="roll (deg)")
    axes[0].axhline(ROLL_SETPOINT_DEG, color="r", linestyle="--", label="setpoint")
    axes[0].legend()
    axes[0].set_ylabel("Roll (deg)")

    axes[1].plot(history["t"], history["theta"], label="pitch (deg)")
    axes[1].axhline(PITCH_SETPOINT_DEG, color="r", linestyle="--", label="setpoint")
    axes[1].legend()
    axes[1].set_ylabel("Pitch (deg)")

    axes[2].plot(history["t"], history["vc"], label="airspeed (kts)")
    axes[2].legend()
    axes[2].set_ylabel("Airspeed (kts)")
    axes[2].set_xlabel("Time (s)")

    axes[3].plot(history["t"], history["elevator"], label="elevator cmd")
    axes[3].legend()
    axes[3].set_ylabel("Elevator")
    axes[3].set_xlabel("Time (s)")

    axes[4].plot(history["t"], history["psi"], label="heading (deg)")
    axes[4].legend()
    axes[4].set_ylabel("Heading (deg)")

    axes[5].plot(history["t"], history["beta"], label="sideslip (deg)")
    axes[5].legend()
    axes[5].set_ylabel("Sideslip (deg)")
    axes[5].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("pid_hold_test.png", dpi=150)
    print(f"Final airspeed: {history['vc'][-1]:.1f} kts")
    print("Saved plot to pid_hold_test.png")


if __name__ == "__main__":
    main()
