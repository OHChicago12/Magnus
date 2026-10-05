"""Training env for the RC-scale airframe.

Adds two things on top of JSBSimEnv, both applied only here (evals use
JSBSimEnv directly, so they are unaffected):

  * airspeed-sensor corruption, sampled once per episode: a gain error, a
    bias, and with some probability a dropout where the airspeed reading is
    replaced by a constant. The true airspeed still drives termination and
    reward.
  * a small penalty on how much the three control-surface commands change
    from one step to the next (reduces jitter and the periodic roll kick).
"""
import numpy as np

try:
    from .jsbsim_env import JSBSimEnv
except ImportError:
    from jsbsim_env import JSBSimEnv

VC = 8  # index of airspeed in the observation


class RCEnv(JSBSimEnv):
    def __init__(self, config=None):
        cfg = dict(config or {})
        self.vc_drop_prob = float(cfg.pop("vc_drop_prob", 0.15))
        self.vc_gain_range = tuple(cfg.pop("vc_gain_range", (0.6, 1.4)))
        self.vc_bias_kts = float(cfg.pop("vc_bias_kts", 3.0))
        self.vc_const_range = tuple(cfg.pop("vc_const_range", (0.0, 40.0)))
        self.smooth_coef = float(cfg.pop("smooth_coef", 0.1))
        super().__init__(**cfg)
        self._prev_action = None
        self._vc_gain, self._vc_bias, self._vc_drop, self._vc_const = 1.0, 0.0, False, 0.0

    def _corrupt(self, obs):
        obs = np.array(obs, dtype=np.float32, copy=True)
        if self._vc_drop:
            obs[VC] = self._vc_const
        else:
            obs[VC] = max(0.0, obs[VC] * self._vc_gain + self._vc_bias)
        return obs

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        r = self.np_random
        self._vc_gain = float(r.uniform(*self.vc_gain_range))
        self._vc_bias = float(r.uniform(-self.vc_bias_kts, self.vc_bias_kts))
        self._vc_drop = bool(r.random() < self.vc_drop_prob)
        self._vc_const = float(r.uniform(*self.vc_const_range))
        self._prev_action = None
        return self._corrupt(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        a = np.clip(np.asarray(action, dtype=np.float32)[:3], -1.0, 1.0)
        if self._prev_action is not None and self.smooth_coef > 0:
            reward -= self.smooth_coef * float(np.mean(np.abs(a - self._prev_action)))
        self._prev_action = a
        return self._corrupt(obs), reward, terminated, truncated, info
