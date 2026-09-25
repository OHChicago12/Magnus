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
        departed_start_prob=0.0,
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
        self.departed_start_prob = departed_start_prob

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

    def _set_initial_conditions(self, departed=False):
        self.fdm.reset_to_initial_conditions(0)

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

        alpha_deg = None
        gamma_deg = 0
        p_rad = 0.0
        r_rad = 0.0
        if departed:
            airspeed_kts = self.np_random.uniform(40, 60)
            alpha_deg = self.np_random.uniform(16, 25)
            gamma_deg = self.np_random.uniform(-10, 0)
            phi_deg = self.np_random.choice([-1, 1]) * self.np_random.uniform(0, 30)
            p_rad = self.np_random.uniform(-1.0, 1.0)
            r_rad = self.np_random.choice([-1, 1]) * self.np_random.uniform(0.6, 1.2)

        self.fdm["ic/h-sl-ft"] = altitude_ft
        self.fdm["ic/vc-kts"] = airspeed_kts
        self.fdm["ic/gamma-deg"] = gamma_deg
        self.fdm["ic/phi-deg"] = phi_deg
        if alpha_deg is not None:
            self.fdm["ic/alpha-deg"] = alpha_deg
        else:
            self.fdm["ic/theta-deg"] = theta_deg
        self.fdm["ic/psi-true-deg"] = 0
        self.fdm["ic/p-rad_sec"] = p_rad
        self.fdm["ic/r-rad_sec"] = r_rad
        self.fdm.run_ic()
        self.fdm.get_propulsion().init_running(-1)
        self.fdm["fcs/mixture-cmd-norm[0]"] = 1.0
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

        departed = bool((options or {}).get("departed", False))
        if not departed and self.departed_start_prob > 0:
            departed = bool(self.np_random.random() < self.departed_start_prob)

        self._set_initial_conditions(departed=departed)
        self._steps = 0
        self._was_departed = departed  # departed spawns count as departed from step 0
        self._recovery_streak = 0
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

        terminated, termination_reason = self._check_termination(obs)
        truncated = self._steps >= self._max_steps
        reward = self._compute_reward(obs, termination_reason)

        info = {
            "sim_time_s": self.fdm.get_sim_time(),
            "termination_reason": termination_reason,
        }
        return obs, reward, terminated, truncated, info

    def _check_termination(self, obs):
        phi, theta = obs[0], obs[1]
        alpha = obs[6]
        altitude = self.fdm["position/h-sl-ft"]

        # Placeholder thresholds -- NOT validated against a real flight
        # envelope yet. Expect to revisit these once real airframe data
        # is available.
        DEPARTURE_ALPHA_DEG = 15.0
        DEPARTURE_BANK_DEG = 60.0
        RECOVERY_ALPHA_DEG = 8.0
        RECOVERY_BANK_DEG = 20.0
        CRASH_ALTITUDE_FT = 50
        UNRECOVERABLE_BANK_DEG = 150
        UNRECOVERABLE_PITCH_DEG = 85

        # Track whether the aircraft has departed controlled flight at any
        # point this episode. Sticky for the rest of the episode -- once
        # departed, always departed, so "recovered" below only fires after
        # a genuine departure, not trivially at t=0.
        if abs(alpha) > DEPARTURE_ALPHA_DEG or abs(phi) > np.deg2rad(DEPARTURE_BANK_DEG):
            self._was_departed = True

        # Failure: hit the ground.
        if altitude < CRASH_ALTITUDE_FT:
            return True, "crashed"

        # Failure: attitude so extreme, recovery is not realistic.
        if abs(phi) > np.deg2rad(UNRECOVERABLE_BANK_DEG) or abs(theta) > np.deg2rad(UNRECOVERABLE_PITCH_DEG):
            return True, "unrecoverable_attitude"

        # Success: controlled flight regained after a genuine departure,
        # sustained for RECOVERY_STREAK_STEPS consecutive steps -- a single
        # instant inside the safe zone (e.g. randomly passing through it
        # while still tumbling) does not count as recovery.
        RECOVERY_STREAK_STEPS = 20  # ~1s at agent_hz=20

        in_recovery_zone = (
            abs(alpha) < RECOVERY_ALPHA_DEG and abs(phi) < np.deg2rad(RECOVERY_BANK_DEG)
        )

        if self._was_departed and in_recovery_zone:
            self._recovery_streak += 1
        else:
            self._recovery_streak = 0

        if self._was_departed and self._recovery_streak >= RECOVERY_STREAK_STEPS:
            return True, "recovered"

        return False, None

    def _compute_reward(self, obs, termination_reason):
        alpha = obs[6]   # AoA, degrees
        phi = obs[0]     # roll, radians

        # Placeholder magnitudes -- not validated, just enough to give PPO
        # a directionally sensible signal. Expect to retune these once
        # real training results are in.
        safe_alpha_deg = 8.0
        alpha_penalty = -abs(alpha - safe_alpha_deg) * 0.1
        bank_penalty = -abs(phi) * 0.5
        shaping_reward = alpha_penalty + bank_penalty

        # Per-step cost while departed -- rewards recovering FASTER, not
        # just eventually. Zero once back in controlled flight.
        departed_penalty = -1.0 if self._was_departed else 0.0

        # Large terminal signals tied to how the episode actually ended.
        terminal_bonus = 0.0
        if termination_reason == "recovered":
            terminal_bonus = 200.0
        elif termination_reason in ("crashed", "unrecoverable_attitude"):
            terminal_bonus = -200.0

        return shaping_reward + departed_penalty + terminal_bonus

    def close(self):
        pass
