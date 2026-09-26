"""
Stage 2 of the LQR baseline: numerically linearize the JSBSim c172p around
the trim point found by lqr_find_trim.py, then compute an LQR gain matrix.

Run from the project root (after lqr_find_trim.py has produced trim_point.npz):
    python scripts/lqr_compute_gain.py

Saves x0, u0, K to lqr_gains.npz for eval_many.py-style evaluation.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.linalg import solve_discrete_are
from envs.jsbsim_env import JSBSimEnv

trim = np.load(os.path.join(ROOT, "trim_point.npz"))
x0 = trim["x0"]  # [phi_rad, theta_rad, psi_rad, p, q, r, alpha_deg, beta_deg, vc_kts]
u0 = trim["u0"]  # [aileron, elevator, rudder, throttle]

TRIM_ALTITUDE_FT = 5000  # must match lqr_find_trim.py's env altitude

env = JSBSimEnv(initial_altitude_ft=TRIM_ALTITUDE_FT, initial_airspeed_kts=120)
env.reset(seed=0, options={"departed": False})  # just to initialize fdm/np_random


def apply_ic(x, use_alpha=False):
    """
    Set JSBSim's initial conditions directly from a 9-element state vector,
    bypassing env's randomized reset -- gives us exact, repeatable states
    for finite-difference perturbation.

    use_alpha=True sets ic/alpha-deg as the independent pitch input instead
    of ic/theta-deg (mirrors the mutually-exclusive alpha/theta choice in
    envs/jsbsim_env.py's _set_initial_conditions -- setting both at once is
    not supported cleanly by JSBSim's IC solver).
    """
    phi_rad, theta_rad, psi_rad, p, q, r, alpha_deg, beta_deg, vc_kts = x

    env.fdm.reset_to_initial_conditions(0)
    env.fdm["ic/h-sl-ft"] = TRIM_ALTITUDE_FT
    env.fdm["ic/vc-kts"] = vc_kts
    env.fdm["ic/gamma-deg"] = 0
    env.fdm["ic/phi-deg"] = np.rad2deg(phi_rad)
    if use_alpha:
        env.fdm["ic/alpha-deg"] = alpha_deg
    else:
        env.fdm["ic/theta-deg"] = np.rad2deg(theta_rad)
    env.fdm["ic/psi-true-deg"] = np.rad2deg(psi_rad)
    env.fdm["ic/p-rad_sec"] = p
    env.fdm["ic/q-rad_sec"] = q
    env.fdm["ic/r-rad_sec"] = r
    env.fdm["ic/beta-deg"] = beta_deg
    env.fdm.run_ic()
    env.fdm.get_propulsion().init_running(-1)
    env.fdm["fcs/mixture-cmd-norm[0]"] = 1.0


def step_from(x, u, use_alpha=False):
    """Set state x, apply action u for one agent step, return resulting obs."""
    apply_ic(x, use_alpha=use_alpha)
    obs, reward, terminated, truncated, info = env.step(u)
    return obs


N_STATES = 9
N_ACTIONS = 4

# Per-dimension perturbation sizes, in the state vector's own units
# (phi/theta/psi: rad, p/q/r: rad/s, alpha/beta: deg, vc: kts).
eps_x = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 2.0])
eps_u = np.array([0.02, 0.02, 0.02, 0.02])

print("Linearizing state matrix A (perturbing each state dimension)...")
A = np.zeros((N_STATES, N_STATES))
for i in range(N_STATES):
    use_alpha = (i == 6)  # only when perturbing alpha itself do we drive IC via alpha
    x_plus = x0.copy()
    x_plus[i] += eps_x[i]
    x_minus = x0.copy()
    x_minus[i] -= eps_x[i]

    obs_plus = step_from(x_plus, u0, use_alpha=use_alpha)
    obs_minus = step_from(x_minus, u0, use_alpha=use_alpha)

    A[:, i] = (obs_plus - obs_minus) / (2 * eps_x[i])
    print(f"  state {i} done")

print("Linearizing control matrix B (perturbing each action dimension)...")
B = np.zeros((N_STATES, N_ACTIONS))
for j in range(N_ACTIONS):
    u_plus = u0.copy()
    u_plus[j] += eps_u[j]
    u_minus = u0.copy()
    u_minus[j] -= eps_u[j]

    obs_plus = step_from(x0, u_plus)
    obs_minus = step_from(x0, u_minus)

    B[:, j] = (obs_plus - obs_minus) / (2 * eps_u[j])
    print(f"  action {j} done")

print("A matrix:")
print(np.round(A, 3))
print("B matrix:")
print(np.round(B, 3))

# LQR cost weights -- placeholder magnitudes, not tuned yet. Heavily
# penalize bank/pitch/AoA (safety-relevant), lightly penalize rates and
# sideslip, ignore heading (psi) entirely since holding a heading isn't
# this baseline's job (the PID hold doesn't do it either).
Q = np.diag([50.0, 50.0, 0.0, 1.0, 1.0, 1.0, 20.0, 5.0, 0.1])
R = np.diag([1.0, 1.0, 1.0, 1.0])

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

print("LQR gain matrix K:")
print(np.round(K, 4))

np.savez(os.path.join(ROOT, "lqr_gains.npz"), x0=x0, u0=u0, K=K)
print(f"Saved LQR gain to {os.path.join(ROOT, 'lqr_gains.npz')}")