import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.multi_plane_env import MultiPlaneEnv

env = MultiPlaneEnv({"departed_start_prob": 1.0, "planes": ["pc7"]})
for ep in range(20):
    obs, _ = env.reset(seed=ep)
    for t in range(2000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if not np.all(np.isfinite(obs)) or np.abs(obs).max() > 1e6:
            print(f"ep {ep} step {t}: worst channel {np.abs(obs).argmax()} value {np.abs(obs).max():.3g}")
            break
        if term or trunc:
            break