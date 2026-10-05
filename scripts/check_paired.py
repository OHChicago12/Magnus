import sys; sys.path.insert(0, "envs")
from jsbsim_env import JSBSimEnv # type: ignore

kw = dict(aircraft="c172p_rc", departed_start_prob=1.0,
          departed_alpha_range_deg=(16, 35), departed_bank_range_deg=(0, 90))
a = JSBSimEnv(**kw)
b = JSBSimEnv(**kw, airframe_dr=True)

bad = 0
for s in range(1000, 1200):
    sa = a.reset(seed=s)[1]["spawn_params"]
    sb = b.reset(seed=s)[1]["spawn_params"]
    sb = {k: v for k, v in sb.items() if k != "airframe"}
    bad += sa != sb
print("mismatched spawns:", bad, "of 200")