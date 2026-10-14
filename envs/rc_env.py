"""Training env for the RC-scale airframe.

Adds these things on top of JSBSimEnv (applied whenever RCEnv is used, which
is in training and in evals whose --env-kwargs contain vc_* or smooth_coef):

  * airspeed-sensor corruption, sampled once per episode: a gain error, a
    bias, and with some probability a dropout where the airspeed reading is
    replaced by a constant. The true airspeed still drives termination and
    reward.
  * a small penalty on how much the three control-surface commands change
    from one step to the next (reduces jitter and the periodic roll kick).
  * (v4) a random action delay per episode: the action actually applied to
    the plane is the one the policy chose `latency` steps earlier.
  * (v4) random sensor masking per episode: with some probability alpha,
    beta and airspeed read as 0, and independently heading (psi) reads as 0.

The v4 options default to off (max_latency_steps=0, mask probabilities 0),
and when off they draw no random numbers, so old runs behave exactly as before.
"""
from collections import deque

import numpy as np

try:
    from .jsbsim_env import JSBSimEnv
except ImportError:
    from jsbsim_env import JSBSimEnv

PSI, ALPHA, BETA, VC = 2, 6, 7, 8  # indices in the observation


class RCEnv(JSBSimEnv):
    def __init__(self, config=None):
        cfg = dict(config or {})
        self.vc_drop_prob = float(cfg.pop("vc_drop_prob", 0.15))
        self.vc_gain_range = tuple(cfg.pop("vc_gain_range", (0.6, 1.4)))
        self.vc_bias_kts = float(cfg.pop("vc_bias_kts", 3.0))
        self.vc_const_range = tuple(cfg.pop("vc_const_range", (0.0, 40.0)))
        self.smooth_coef = float(cfg.pop("smooth_coef", 0.1))
        self.max_latency_steps = int(cfg.pop("max_latency_steps", 0))
        self.mask_abv_prob = float(cfg.pop("mask_abv_prob", 0.0))
        self.mask_psi_prob = float(cfg.pop("mask_psi_prob", 0.0))
        super().__init__(**cfg)
        self._prev_action = None
        self._vc_gain, self._vc_bias, self._vc_drop, self._vc_const = 1.0, 0.0, False, 0.0
        self._latency = 0
        self._action_queue = None
        self._mask_abv = False
        self._mask_psi = False

    def _corrupt(self, obs):
        obs = np.array(obs, dtype=np.float32, copy=True)
        if self._vc_drop:
            obs[VC] = self._vc_const
        else:
            obs[VC] = max(0.0, obs[VC] * self._vc_gain + self._vc_bias)
        if self._mask_abv:
            obs[ALPHA] = 0.0
            obs[BETA] = 0.0
            obs[VC] = 0.0
        if self._mask_psi:
            obs[PSI] = 0.0
        return obs

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        r = self.np_random
        self._vc_gain = float(r.uniform(*self.vc_gain_range))
        self._vc_bias = float(r.uniform(-self.vc_bias_kts, self.vc_bias_kts))
        self._vc_drop = bool(r.random() < self.vc_drop_prob)
        self._vc_const = float(r.uniform(*self.vc_const_range))
        # v4 draws come after the old ones and are skipped when off
        self._latency = int(r.integers(0, self.max_latency_steps + 1)) if self.max_latency_steps > 0 else 0
        self._mask_abv = bool(r.random() < self.mask_abv_prob) if self.mask_abv_prob > 0 else False
        self._mask_psi = bool(r.random() < self.mask_psi_prob) if self.mask_psi_prob > 0 else False
        self._action_queue = None
        self._prev_action = None
        return self._corrupt(obs), info

    def step(self, action):
        a_cmd = np.asarray(action, dtype=np.float32)
        a_exec = a_cmd
        if self._latency > 0:
            # same scheme as eval_visualize.py: delay by `latency` steps,
            # holding the first command until the queue fills
            if self._action_queue is None:
                self._action_queue = deque([a_cmd] * self._latency)
            self._action_queue.append(a_cmd)
            a_exec = self._action_queue.popleft()
        obs, reward, terminated, truncated, info = super().step(a_exec)
        a = np.clip(a_cmd[:3], -1.0, 1.0)
        if self._prev_action is not None and self.smooth_coef > 0:
            reward -= self.smooth_coef * float(np.mean(np.abs(a - self._prev_action)))
        self._prev_action = a
        return self._corrupt(obs), reward, terminated, truncated, info
