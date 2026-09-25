"""
Unit tests for controllers/pid.py.

Run from the project root:
    pytest tests/test_pid.py -v

(add pytest to requirements.txt if it isn't installed: pip install pytest)
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest
from controllers.pid import PID


def test_proportional_only_response():
    # ki=0, kd=0 isolates the P term: output should be exactly kp * error.
    # output_limits widened so the expected value (12.0) isn't clamped.
    pid = PID(kp=2.0, ki=0.0, kd=0.0, output_limits=(-100.0, 100.0))
    output = pid.update(setpoint=10.0, measurement=4.0, dt=0.1)
    assert output == pytest.approx(2.0 * (10.0 - 4.0))


def test_integral_accumulates_over_time():
    # ki alone: with constant error, integral term should grow linearly with
    # accumulated time, so output should increase step over step.
    pid = PID(kp=0.0, ki=1.0, kd=0.0, output_limits=(-100.0, 100.0))
    out1 = pid.update(setpoint=5.0, measurement=0.0, dt=1.0)
    out2 = pid.update(setpoint=5.0, measurement=0.0, dt=1.0)
    out3 = pid.update(setpoint=5.0, measurement=0.0, dt=1.0)
    # error=5 each step, dt=1 -> integral accumulates 5, 10, 15 -> output = ki * integral
    assert out1 == pytest.approx(5.0)
    assert out2 == pytest.approx(10.0)
    assert out3 == pytest.approx(15.0)


def test_derivative_is_zero_on_first_call():
    # No previous measurement yet, so derivative term must not contribute
    # (this is what the ramp-vs-step debugging session was checking for).
    pid = PID(kp=0.0, ki=0.0, kd=5.0)
    output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)
    assert output == pytest.approx(0.0)


def test_derivative_on_measurement_not_error():
    # Derivative-on-measurement: if the setpoint jumps but the measurement
    # hasn't moved between calls, derivative should be ~0 (no setpoint kick).
    # This is the fix from the Phase 0 debugging session -- verifies it stays fixed.
    pid = PID(kp=0.0, ki=0.0, kd=1.0)
    pid.update(setpoint=0.0, measurement=3.0, dt=0.1)  # prime prev_measurement
    output = pid.update(setpoint=50.0, measurement=3.0, dt=0.1)  # setpoint jumps, measurement doesn't
    assert output == pytest.approx(0.0)


def test_derivative_responds_to_measurement_change():
    # With setpoint fixed, if measurement changes, derivative-on-measurement
    # should react (opposite sign of measurement's rate of change).
    # output_limits widened so the expected value (-4.0) isn't clamped.
    pid = PID(kp=0.0, ki=0.0, kd=1.0, output_limits=(-100.0, 100.0))
    pid.update(setpoint=0.0, measurement=0.0, dt=1.0)  # prime prev_measurement
    output = pid.update(setpoint=0.0, measurement=4.0, dt=1.0)  # measurement rose by 4 in 1s
    # derivative = -(4 - 0) / 1 = -4 -> output = kd * -4 = -4
    assert output == pytest.approx(-4.0)


def test_output_is_clamped_to_limits():
    pid = PID(kp=100.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0))
    high = pid.update(setpoint=1000.0, measurement=0.0, dt=0.1)
    pid.reset()
    low = pid.update(setpoint=-1000.0, measurement=0.0, dt=0.1)
    assert high == 1.0
    assert low == -1.0


def test_anti_windup_prevents_integral_growth_while_saturated():
    # With anti-windup, running many saturated steps should NOT leave the
    # integral term growing unboundedly -- once un-saturated, output should
    # respond immediately rather than needing to "unwind" a huge integral.
    pid = PID(kp=1.0, ki=1.0, kd=0.0, output_limits=(-1.0, 1.0))

    # Drive it into saturation for many steps with a huge error.
    for _ in range(50):
        pid.update(setpoint=1000.0, measurement=0.0, dt=0.1)

    # Now measurement catches up to setpoint (error -> 0). If the integral
    # had wound up unboundedly, output would still be pinned at hi despite
    # zero error. With anti-windup it should drop back near 0.
    output = pid.update(setpoint=1000.0, measurement=1000.0, dt=0.1)
    assert output == pytest.approx(0.0, abs=1e-6)


def test_reset_clears_integral_and_derivative_state():
    pid = PID(kp=0.0, ki=1.0, kd=1.0, output_limits=(-1000.0, 1000.0))
    pid.update(setpoint=10.0, measurement=0.0, dt=1.0)  # accumulate integral, set prev_measurement
    pid.reset()

    # After reset, a first call should behave like a fresh controller:
    # zero derivative (no prev_measurement) and zero accumulated integral.
    output = pid.update(setpoint=10.0, measurement=0.0, dt=1.0)
    # Only this single step's integral contribution (10 * 1.0 = 10) should show up,
    # not anything carried over from before reset.
    assert output == pytest.approx(10.0)


def test_negative_error_direction():
    # Sanity check: PID should push in the correct direction when measurement
    # is above setpoint, not just below it.
    pid = PID(kp=1.0, ki=0.0, kd=0.0)
    output = pid.update(setpoint=0.0, measurement=5.0, dt=0.1)
    assert output < 0