"""Simple PID controller with output clamping and basic anti-windup.

This is the baseline building block for Magnus's classical inner loop
(manual/assisted modes). Nothing fancy -- get this right and boring
before reaching for LQR or dynamic inversion.
"""


class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lo, self.hi = output_limits
        self._integral = 0.0
        self._prev_measurement = None

    def reset(self):
        self._integral = 0.0
        self._prev_measurement = None

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self._integral += error * dt

        if self._prev_measurement is None:
            derivative = 0.0
        else:
            derivative = -(measurement - self._prev_measurement) / dt
        self._prev_measurement = measurement

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        clamped = max(self.lo, min(self.hi, output))

        # Basic anti-windup: undo the integral accumulation if we're saturated,
        # so the integral term doesn't keep growing while the output is clipped.
        if clamped != output:
            self._integral -= error * dt

        return clamped