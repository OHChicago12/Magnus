import numpy as np
from envs.jsbsim_env import JSBSimEnv

env = JSBSimEnv(departed_start_prob=0.5,
                departed_alpha_range_deg=(16, 35),
                departed_bank_range_deg=(0, 90))
for seed in [4, 139, 12, 79]:
    env.reset(seed=seed)
    obs, info = env.reset()          # same double reset the eval does
    sp = info["spawn_params"]
    f = env.fdm
    print(f"seed {seed}: departed={sp['departed']} phi0={sp['phi_deg']:.0f} "
          f"alpha0={sp['alpha_deg']} gamma0={sp['gamma_deg']:.0f} "
          f"v0={sp['airspeed_kts']:.0f} p0={sp['p_rad']:.2f} r0={sp['r_rad']:.2f}")
    print(f"   at IC:        theta={f['attitude/theta-deg']:.0f} phi={f['attitude/phi-deg']:.0f} "
          f"alpha={f['aero/alpha-deg']:.0f}")
    obs, *_ = env.step(np.array([0, 0, 0, 0.6], dtype=np.float32))
    print(f"   after 1 step: theta={np.rad2deg(obs[1]):.0f} phi={np.rad2deg(obs[0]):.0f} "
          f"alpha={obs[6]:.0f}")