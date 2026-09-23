import numpy as np
from envs.jsbsim_env import JSBSimEnv

env = JSBSimEnv(departed_start_prob=1.0)

for i in range(5):
    env.reset(seed=i)
    total = 0.0
    for step in range(2000):
        # do-nothing policy: neutral controls, 85% throttle
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 0.85], dtype=np.float32))
        total += r
        if term or trunc:
            break
    print(f"EP {i}: steps={step + 1} reason={info['termination_reason']} "
            f"return={total:.0f} altitude={env.fdm['position/h-sl-ft']:.0f}")