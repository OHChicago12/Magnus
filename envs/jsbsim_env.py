"""
Gymnasium-compatible wrapper around JSBSim for Magnus development.

This environment exposes JSBSim's six-DOF flight dynamics as a standard
Gym interface so classical controllers (and later, RL agents) can drive
the same simulation loop without caring what's underneath.

Phase 0 goal: validate the sim <-> controller plumbing using a stock
JSBSim aircraft (c172p) BEFORE building Magnus's own airframe model.
Do not spend time tuning against this aircraft's dynamics -- it's
a placeholder to prove the loop works.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim


class JSBSimEnv(gym.Env):
    """
    Minimal JSBSim environment for attitude-hold development.

    Observation (9,):
        [phi (roll, rad), theta (pitch, rad), psi (heading, rad),
         p, q, r (body rates, rad/s),
         alpha (AoA, deg), beta (sideslip, deg),
         vc (calibrated airspeed, kts)]

    Action (4,): [aileron, elevator, rudder, throttle], each in [-1, 1]
        (throttle is internally remapped to [0, 1] before being sent to JSBSim)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        aircraft="c172p",
        dt=1.0 / 120.0,
        agent_hz=20,
        initial_altitude_ft=5000,
        initial_airspeed_kts=120,
        root_dir=None,
        # Domain randomization spreads -- each reset() samples a fresh
        # initial condition within these +/- ranges. This substitutes for
        # variation you'd otherwise get from a physical test fleet.
        altitude_randomization_ft=1000,
        airspeed_randomization_kts=15,
        attitude_randomization_deg=5,
        wind_randomization_fps=15,
    ):
        super().__init__()

        self.aircraft = aircraft
        self.dt = dt
        # How many JSBSim integration steps per controller ("agent") step.
        # JSBSim integrates fast (e.g. 120 Hz) while your controller/RL
        # policy typically runs slower (e.g. 20 Hz) -- this ratio bridges them.
        self.steps_per_action = max(1, int((1.0 / agent_hz) / dt))
        self.initial_altitude_ft = initial_altitude_ft
        self.initial_airspeed_kts = initial_airspeed_kts
        self.altitude_randomization_ft = altitude_randomization_ft
        self.airspeed_randomization_kts = airspeed_randomization_kts
        self.attitude_randomization_deg = attitude_randomization_deg
        self.wind_randomization_fps = wind_randomization_fps

        root_dir = root_dir or jsbsim.get_default_root_dir()
        self.fdm = jsbsim.FGFDMExec(jsbsim.get_default_root_dir())
        self.fdm.set_dt(self.dt)
        self.fdm.load_model(self.aircraft)

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )

        # Generous bounds -- tighten once you know your actual flight envelope.
        obs_high = np.array(
            [np.pi, np.pi, 2 * np.pi, 10, 10, 10, 90, 90, 400], dtype=np.float32
        )
        obs_low = np.array(
            [-np.pi, -np.pi, 0.0, -10, -10, -10, -90, -90, -400], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        self._steps = 0
        self._max_steps = 2000  # ~100s of sim time at agent_hz=20

    def _set_initial_conditions(self):
        # Sample randomized initial conditions. self.np_random is seeded by
        # gymnasium via super().reset(seed=seed), called before this method.
        altitude_ft = self.initial_altitude_ft + self.np_random.uniform(
            -self.altitude_randomization_ft, self.altitude_randomization_ft
        )
        airspeed_kts = self.initial_airspeed_kts + self.np_random.uniform(
            -self.airspeed_randomization_kts, self.airspeed_randomization_kts
        )
        phi_deg = self.np_random.uniform(
            -self.attitude_randomization_deg, self.attitude_randomization_deg
        )
        theta_deg = self.np_random.uniform(
            -self.attitude_randomization_deg, self.attitude_randomization_deg
        )

        self.fdm["ic/h-sl-ft"] = altitude_ft
        self.fdm["ic/vc-kts"] = airspeed_kts
        self.fdm["ic/gamma-deg"] = 0      # flight path angle
        self.fdm["ic/phi-deg"] = phi_deg  # roll, randomized
        self.fdm["ic/theta-deg"] = theta_deg  # pitch, randomized
        self.fdm["ic/psi-true-deg"] = 0   # heading
        self.fdm.run_ic()
        self.fdm.get_propulsion().init_running(-1)
        self.fdm["fcs/mixture-cmd-norm[0]"] = 1.0

        # Randomized wind, set AFTER run_ic() so it doesn't perturb trim.
        wind_north_fps = self.np_random.uniform(
            -self.wind_randomization_fps, self.wind_randomization_fps
        )
        wind_east_fps = self.np_random.uniform(
            -self.wind_randomization_fps, self.wind_randomization_fps
        )
        self.fdm["atmosphere/wind-north-fps"] = wind_north_fps
        self.fdm["atmosphere/wind-east-fps"] = wind_east_fps

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.fdm.set_dt(self.dt)
        self._set_initial_conditions()
        self._steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        fdm = self.fdm
        return np.array(
            [
                fdm["attitude/phi-rad"],
                fdm["attitude/theta-rad"],
                fdm["attitude/psi-rad"],
                fdm["velocities/p-rad_sec"],
                fdm["velocities/q-rad_sec"],
                fdm["velocities/r-rad_sec"],
                fdm["aero/alpha-deg"],
                fdm["aero/beta-deg"],
                fdm["velocities/vc-kts"],
            ],
            dtype=np.float32,
        )

    def step(self, action):
        aileron, elevator, rudder, throttle = action
        aileron = float(np.clip(aileron, -1, 1))
        elevator = float(np.clip(elevator, -1, 1))
        rudder = float(np.clip(rudder, -1, 1))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        self.fdm["fcs/aileron-cmd-norm"] = aileron
        self.fdm["fcs/elevator-cmd-norm"] = elevator
        self.fdm["fcs/rudder-cmd-norm"] = rudder
        self.fdm["fcs/throttle-cmd-norm[0]"] = throttle

        for _ in range(self.steps_per_action):
            keep_running = self.fdm.run()
            if not keep_running:
                break

        self._steps += 1
        obs = self._get_obs()

        terminated = self._check_termination(obs)
        truncated = self._steps >= self._max_steps
        reward = self._compute_reward(obs)

        info = {"sim_time_s": self.fdm.get_sim_time()}
        return obs, reward, terminated, truncated, info

    def _check_termination(self, obs):
        phi, theta = obs[0], obs[1]
        altitude = self.fdm["position/h-sl-ft"]
        # Placeholder safety bounds -- these are NOT your real envelope limits,
        # just enough to stop a runaway sim during Phase 0 testing.
        if altitude < 100:
            return True
        if abs(phi) > np.deg2rad(160) or abs(theta) > np.deg2rad(80):
            return True
        return False

    def _compute_reward(self, obs):
        alpha = obs[6]   # AoA, degrees
        phi = obs[0]     # roll, radians

        safe_alpha_deg = 8.0    # rough safe AoA target for c172p, well below stall
        alpha_penalty = -abs(alpha - safe_alpha_deg) * 0.1
        bank_penalty = -abs(phi) * 0.5

        return alpha_penalty + bank_penalty

    def close(self):
        pass
