import numpy as np
from envs.jsbsim_env import JSBSimEnv

def show(tag, obs):
    print(f"{tag}: alpha={obs[6]:.1f}  theta={np.rad2deg(obs[1]):.1f}  "
          f"phi={np.rad2deg(obs[0]):.1f}  vc={obs[8]:.0f}  p/q/r={obs[3]:.2f}/{obs[4]:.2f}/{obs[5]:.2f}")

env = JSBSimEnv(departed_start_prob=0.0)
env.reset(seed=1, options={"departed": True})
obs, _ = env.reset(seed=2)                      # normal start right after a departed one
show("normal AFTER departed", obs)

fresh = JSBSimEnv(departed_start_prob=0.0)
obs, _ = fresh.reset(seed=2)                    # same seed, fresh env
show("normal on FRESH env  ", obs)