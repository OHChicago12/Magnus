"""Multi-plane env: each episode flies one randomly chosen plane, rescaled so it
looks RC-sized to the policy.

Scaling rule (dynamic similarity), with s = plane wingspan / RC wingspan and
tau = sqrt(s), rounded so the agent step is a whole number of sim steps:
    speeds   (airspeed, wind, spawn airspeed)  x tau      (policy sees them / tau)
    lengths  (start altitude and its spread)   x tau^2
    rates    (p, q, r, spawn rates)            / tau      (policy sees them x tau)
    time     (agent step, servo lag)           x tau
    angles   (roll, pitch, heading, alpha, beta) unchanged
The RC plane itself (c172p_rc) is never changed: tau = 1.

Usage:
    env = MultiPlaneEnv({"planes": ["c172p_rc", "c182", "pa28"], **your_rc_env_config})
Put `"planes": ["J3Cub"]` (or ["c310"]) for a held-out test on one plane.
Run `python envs/multi_plane_env.py` for a quick smoke test.
"""
import math
import os

import gymnasium as gym
import jsbsim
import numpy as np

try:
    from .rc_env import RCEnv
except ImportError:
    from rc_env import RCEnv

RC_PLANE = "c172p_rc"
RC_SPAN_FT = 4.27

# JSBSimEnv defaults, used when your config doesn't set a value
_DEFAULTS = {
    "initial_altitude_ft": 5000,
    "altitude_randomization_ft": 1000,
    "initial_airspeed_kts": 120,
    "airspeed_randomization_kts": 15,
    "wind_randomization_fps": 15,
    "departed_airspeed_range_kts": (40, 60),
    "departed_yaw_rate_range": (0.6, 1.2),
    "departed_roll_rate_max": 1.0,
    "servo_tau_s": 0.0,
    "servo_rate_per_s": 0.0,
}
_TIMES_TAU = ("initial_airspeed_kts", "airspeed_randomization_kts", "wind_randomization_fps",
              "departed_airspeed_range_kts", "servo_tau_s")
_OVER_TAU = ("departed_yaw_rate_range", "departed_roll_rate_max", "servo_rate_per_s")
_TIMES_TAU_SQ = ("initial_altitude_ft", "altitude_randomization_ft")   # lengths scale as tau^2

_span_cache = {}


def plane_span_ft(name, root_dir=None):
    """Wingspan of a JSBSim plane in feet (measured once, then cached)."""
    if name not in _span_cache:
        fdm = jsbsim.FGFDMExec(root_dir or jsbsim.get_default_root_dir())
        fdm.set_debug_level(0)
        fdm.load_model(name)
        fdm["ic/h-sl-ft"] = 3000
        fdm["ic/vc-kts"] = 60
        fdm.run_ic()
        _span_cache[name] = float(fdm["metrics/bw-ft"])
    return _span_cache[name]


def _mul(value, f):
    if isinstance(value, (tuple, list)):
        return tuple(float(v) * f for v in value)
    return float(value) * f


def plane_config(name, base):
    """Returns (config for this plane, tau, sim steps per agent step)."""
    dt = base.get("dt", 1.0 / 120.0)
    ref_steps = max(1, int(round(1.0 / (base.get("agent_hz", 60) * dt))))
    cfg = dict(base)
    cfg["aircraft"] = name
    if name == RC_PLANE:
        return cfg, 1.0, ref_steps
    s = max(1.0, plane_span_ft(name, base.get("root_dir")) / RC_SPAN_FT)
    steps = max(ref_steps, int(round(ref_steps * math.sqrt(s))))
    tau = steps / ref_steps
    # tiny reduction so int() inside JSBSimEnv can't round steps down
    cfg["agent_hz"] = (1.0 / dt) / steps * (1.0 - 1e-6)
    for k in _TIMES_TAU:
        cfg[k] = _mul(base.get(k, _DEFAULTS[k]), tau)
    for k in _OVER_TAU:
        cfg[k] = _mul(base.get(k, _DEFAULTS[k]), 1.0 / tau)
    for k in _TIMES_TAU_SQ:
        cfg[k] = _mul(base.get(k, _DEFAULTS[k]), tau * tau)
    cfg["airframe_dr"] = None   # airframe randomization is only set up for the RC plane
    return cfg, tau, steps


class _ScaledPlaneEnv(RCEnv):
    """RCEnv for one plane, with the true observation rescaled to RC size
    (before RCEnv's sensor corruption, so stuck/noisy readings stay RC-sized)."""

    def __init__(self, config, tau):
        self.tau = float(tau)
        self._obs_scale = np.array(
            [1, 1, 1, self.tau, self.tau, self.tau, 1, 1, 1.0 / self.tau], dtype=np.float32
        )
        super().__init__(config)

    def _get_obs(self):
        return super()._get_obs() * self._obs_scale


class MultiPlaneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        cfg = dict(config or {})
        self.planes = list(cfg.pop("planes", [RC_PLANE, "c182", "pa28"]))
        self.envs, self.taus, self.steps = {}, {}, {}
        for name in self.planes:
            pcfg, tau, steps = plane_config(name, cfg)
            self.envs[name] = _ScaledPlaneEnv(pcfg, tau)
            self.taus[name], self.steps[name] = tau, steps
        first = self.envs[self.planes[0]]
        self.observation_space = first.observation_space
        self.action_space = first.action_space
        self._cur = None
        self.plane = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.plane = self.planes[int(self.np_random.integers(len(self.planes)))]
        self._cur = self.envs[self.plane]
        obs, info = self._cur.reset(seed=seed, options=options)
        info["plane"] = self.plane
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._cur.step(action)
        info["plane"] = self.plane
        return obs, reward, terminated, truncated, info

    def close(self):
        for e in self.envs.values():
            e.close()

    def __getattr__(self, name):
        # eval/logging code can read things like env.fdm from the current plane
        cur = self.__dict__.get("_cur")
        if name.startswith("__") or cur is None:
            raise AttributeError(name)
        return getattr(cur, name)


if __name__ == "__main__":
    # Demo settings only (RC-sized numbers). In training, pass YOUR RC env config.
    demo = {
        "agent_hz": 60,
        "initial_airspeed_kts": 25,
        "airspeed_randomization_kts": 4,
        "wind_randomization_fps": 3,
        "departed_start_prob": 1.0,
        "departed_airspeed_range_kts": (12, 18),
        "vc_drop_prob": 0.0,
    }
    names = [n for n in ["c172p_rc", "c182", "pa28", "J3Cub", "c310"]]
    print(f"{'plane':9} {'span_ft':>8} {'steps':>6} {'tau':>5} {'start_vc':>9} {'engines':>8}  thrust_lb_per_engine")
    for name in names:
        if not os.path.isdir(os.path.join(jsbsim.get_default_root_dir(), "aircraft", name)):
            print(f"{name:9} skipped (not installed)")
            continue
        env = MultiPlaneEnv({"planes": [name], **demo})
        obs, _ = env.reset(seed=0)
        for _ in range(60):
            obs, *_ = env.step(np.array([0, 0, 0, 0.85], dtype=np.float32))
        n_eng = env.fdm.get_propulsion().get_num_engines()
        thrust = [round(env.fdm[f"propulsion/engine[{i}]/thrust-lbs"]) for i in range(n_eng)]
        span = plane_span_ft(name)
        print(f"{name:9} {span:8.2f} {env.steps[name]:6d} {env.taus[name]:5.2f} "
              f"{env.reset(seed=0)[0][8]:9.1f} {n_eng:8d}  {thrust}")