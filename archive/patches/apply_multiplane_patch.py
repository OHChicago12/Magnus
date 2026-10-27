"""Patch envs/jsbsim_env.py for multi-plane training.

Two changes (single-engine planes behave exactly as before):
  1. Throttle and mixture go to ALL engines, so twins like the c310 get power on both.
  2. The roll rate of departed spawns becomes a setting (departed_roll_rate_max,
     default 1.0 = the old hard-coded value), so it can be scaled per plane.

Usage (from the repo root):  python apply_multiplane_patch.py
"""
import shutil
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "envs/jsbsim_env.py"
src = open(path).read()

if "departed_roll_rate_max" in src:
    print("Already patched, nothing to do.")
    sys.exit(0)

EDITS = [
    (
        "        departed_spawn_pitch_max_deg=55.0,\n",
        "        departed_spawn_pitch_max_deg=55.0,\n"
        "        departed_roll_rate_max=1.0,   # rad/s, max initial roll rate of departed spawns\n",
    ),
    (
        "        self.departed_spawn_pitch_max_deg = departed_spawn_pitch_max_deg\n",
        "        self.departed_spawn_pitch_max_deg = departed_spawn_pitch_max_deg\n"
        "        self.departed_roll_rate_max = departed_roll_rate_max\n",
    ),
    (
        "            p_rad = self.np_random.uniform(-1.0, 1.0)\n",
        "            p_rad = self.np_random.uniform(-self.departed_roll_rate_max, self.departed_roll_rate_max)\n",
    ),
    (
        '        self.fdm["fcs/mixture-cmd-norm[0]"] = 1.0\n',
        "        for i in range(self.fdm.get_propulsion().get_num_engines()):\n"
        '            self.fdm[f"fcs/mixture-cmd-norm[{i}]"] = 1.0\n',
    ),
    (
        '        self.fdm["fcs/throttle-cmd-norm[0]"] = throttle\n',
        "        for i in range(self.fdm.get_propulsion().get_num_engines()):\n"
        '            self.fdm[f"fcs/throttle-cmd-norm[{i}]"] = throttle\n',
    ),
]

for old, _ in EDITS:
    n = src.count(old)
    if n != 1:
        print(f"Aborted, nothing changed. Expected 1 match, found {n} for:\n{old}")
        sys.exit(1)

shutil.copy(path, path + ".bak_multiplane")
for old, new in EDITS:
    src = src.replace(old, new)
open(path, "w").write(src)
print(f"Patched {path} (backup: {path}.bak_multiplane)")