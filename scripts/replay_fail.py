import sys, numpy as np, matplotlib.pyplot as plt
import eval_visualize as ev
from envs.jsbsim_env import JSBSimEnv

EPS = [120, 177]
SEED = 1000
env_kwargs = dict(initial_altitude_ft=500, altitude_randomization_ft=100,
                  initial_airspeed_kts=30, airspeed_randomization_kts=4,
                  departed_airspeed_range_kts=(12, 18),
                  departed_yaw_rate_range=(1.5, 3.0),
                  wind_randomization_fps=4, agent_hz=60, servo_tau_s=0.05)
gain = np.array([1, 1, 1, 0.425, 0.425, 0.425, 1, 1, 3.6], dtype=np.float32)

algo = ev.load_policy("checkpoints_v2/iter_2500")
module = algo.get_module()
env = JSBSimEnv(aircraft="c172p_rc", departed_start_prob=1.0,
                departed_alpha_range_deg=(16, 35),
                departed_bank_range_deg=(0, 90), **env_kwargs)

fig, axes = plt.subplots(len(ev.STATE_KEYS), 1, figsize=(9, 11), sharex=True)
for ep in EPS:
    env.reset(seed=SEED + ep)
    r = ev.rollout_episode(module, env, obs_gain=gain)
    tr = r["trace"]
    print(f"ep {ep}: {r['termination_reason']}, len={r['length']}")
    for i in range(0, min(r["length"], 50), 2):
        print(f"  step {i}: pitch={tr['pitch'][i]:.0f} alpha={tr['aoa'][i]:.0f} "
              f"q={tr['q'][i]:.0f} v={tr['airspeed'][i]:.0f} elev={tr['elevator'][i]:.2f}")
    for ax, k in zip(axes, ev.STATE_KEYS):
        ax.plot(tr[k], label=f"ep {ep}"); ax.set_ylabel(k); ax.grid(alpha=0.3)
axes[0].legend(); axes[-1].set_xlabel("step (60 per second)")
fig.savefig("replay_fail.png", dpi=150)