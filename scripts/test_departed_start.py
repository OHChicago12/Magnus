import numpy as np
from envs.jsbsim_env import JSBSimEnv

env = JSBSimEnv()

def show(tag, i, f):
    print(f"ep {i} {tag}: alpha={f['aero/alpha-deg']:.1f} beta={f['aero/beta-deg']:.1f} "
          f"bank={f['attitude/phi-deg']:.1f} pitch={f['attitude/theta-deg']:.1f} "
          f"gamma={f['flight-path/gamma-deg']:.1f} airspeed={f['velocities/vc-kts']:.1f}")

for i in range(5):
    env.reset(seed=i, options={"departed": True})
    show("start", i, env.fdm)
    env.step(np.array([0.0, 0.0, 0.0, 0.85], dtype=np.float32))
    show("after1", i, env.fdm)