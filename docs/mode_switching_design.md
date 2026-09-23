# Mode-Switching Logic Design (draft)

Status: design only, not yet implemented in code. This maps directly
onto concepts already implemented in envs/jsbsim_env.py's
_check_termination (departure detection, recovery streak) -- the same
signals that end an episode in sim should drive mode transitions in
the real system.

## The four modes (from project architecture)

1. Manual -- pilot has full control authority. Default mode.
2. Assisted -- envelope protection active underneath pilot input
   (limits, not overrides). Pilot still primary.
3. Auto-recovery -- aircraft has departed controlled flight. Magnus
   takes primary control authority to execute recovery.
4. Autonomous -- pilot has explicitly ceded control for an extended
   task. NOT safety-triggered -- a deliberate pilot choice, out of
   scope for this design pass.

## Transition triggers (Manual/Assisted <-> Auto-recovery)

This is the safety-critical transition and the main subject of this
design. Reuses the same alpha/bank thresholds as _check_termination:

- ENTER auto-recovery when: abs(alpha) > DEPARTURE_ALPHA_DEG or
  abs(phi) > DEPARTURE_BANK_DEG (same constants as the sim).
- EXIT auto-recovery back to Assisted when: sustained inside the
  RECOVERY_ALPHA_DEG / RECOVERY_BANK_DEG zone for RECOVERY_STREAK_STEPS
  consecutive steps -- the exact same streak logic already validated
  in the sim's termination check. Do NOT exit on a single instant
  inside the safe zone, for the same reason a single-step check was
  wrong for episode termination: noise can pass through the threshold
  in a genuinely unstable aircraft.
- EXIT auto-recovery back to Manual (not Assisted) requires an
  explicit pilot input confirming they're resuming control -- an
  automatic silent handback the instant the streak completes is
  dangerous, since the pilot may not be ready to immediately resume
  manual authority after Magnus was just fighting a departure.

## Manual <-> Assisted transition

RESOLVED: Assisted is the permanent default whenever Magnus is armed.
Manual is not a distinct state -- envelope protection limits are
always live in the background, matching how real fly-by-wire systems
work. There is no separate "raw, zero-intervention" mode; testing the
raw airframe's behavior happens by flying without Magnus armed at
all (hardware/software level), not via an in-flight mode switch.

## Hysteresis / anti-flapping

RESOLVED: asymmetric hysteresis on entry vs exit, matching the very
different costs of a false positive on each side:

- ENTER auto-recovery: short confirmation streak, ~2-3 consecutive
  steps (~0.1s at 20Hz) past the departure threshold. Long enough to
  filter single-sample sensor noise; short enough that a genuine
  departure (which will keep climbing past threshold on the very next
  sample anyway) loses almost no real response time.
- EXIT auto-recovery: long confirmation streak, the existing 20-step
  (~1s) RECOVERY_STREAK_STEPS from _check_termination. A false exit
  (handing back control too early, before flight is genuinely settled)
  is far more dangerous than a slightly delayed exit, so this side
  deliberately requires much stronger evidence of real recovery.

## Explicitly out of scope for this pass

- Autonomous mode entry/exit conditions (pilot-initiated, not
  safety-driven)
- Sensor fusion / state estimation reliability -- this design assumes
  alpha and phi are known accurately; what happens if sensors disagree
  or a pitot tube fails is a separate, important problem
- Actual control blending during the Auto-recovery handback (does
  authority transfer instantly, or ramp over some period so pilot
  inputs aren't abruptly cut off then abruptly restored)

## Next steps
- Resolve the Manual/Assisted open question above
- Decide on ENTER-side hysteresis (or confirm immediate entry is right)
- Prototype this as a simple state machine class once real sensor data
  (or at least sim data with realistic noise) is available to test
  against