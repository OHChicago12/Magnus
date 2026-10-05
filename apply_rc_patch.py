"""Adds RC-scale test options to eval_visualize.py (one-time patch).

New options:
  --rc similar   RC-scale airframe, policy sees inputs mapped back to its training scale, 60 Hz
  --rc raw       RC-scale airframe, policy sees raw RC numbers, 20 Hz (naive deployment)
  --env-kwargs   extra JSBSimEnv settings as JSON, e.g. '{"agent_hz": 60}'
  --obs-gain     comma list of 9 multipliers applied to what the policy sees

Usage:  python apply_rc_patch.py scripts/eval_visualize.py
A backup is written to <file>.bak. Nothing is changed if any anchor is missing.
"""
import shutil, sys

path = sys.argv[1] if len(sys.argv) > 1 else "scripts/eval_visualize.py"
src = open(path).read()
if '"--rc"' in src:
    sys.exit("Already patched.")

REPL = [
 ("import argparse\n", "import argparse\nimport json\n"),
 ("departed_bank_range_deg=(0, 30), obs_noise=0.0, latency_steps=0):",
  "departed_bank_range_deg=(0, 30), obs_noise=0.0, latency_steps=0,\n"
  "                          env_kwargs=None, obs_gain=None):"),
 ("            departed_bank_range_deg=departed_bank_range_deg,\n        )",
  "            departed_bank_range_deg=departed_bank_range_deg,\n"
  "            **(env_kwargs or {}),\n        )"),
 ("latency_steps=latency_steps, rng=rng))",
  "latency_steps=latency_steps, rng=rng,\n"
  "                                            obs_gain=obs_gain))"),
 ("obs_noise=0.0, latency_steps=0, rng=None):",
  "obs_noise=0.0, latency_steps=0, rng=None, obs_gain=None):"),
 ("            policy_obs = (obs + rng.normal(0.0, noise_std)).astype(np.float32)\n",
  "            policy_obs = (obs + rng.normal(0.0, noise_std)).astype(np.float32)\n"
  "        if obs_gain is not None:\n"
  "            policy_obs = (policy_obs * obs_gain).astype(np.float32)\n"),
 ("    args = parser.parse_args()\n",
  '''    parser.add_argument("--env-kwargs", default="{}",
                         help="Extra JSBSimEnv settings as JSON, e.g. '{\\"agent_hz\\": 60}'")
    parser.add_argument("--obs-gain", default="",
                         help="9 comma-separated multipliers on the policy's inputs "
                              "[phi,theta,psi,p,q,r,alpha,beta,vc]")
    parser.add_argument("--rc", choices=["raw", "similar"], default=None,
                         help="Preset for the c172p_rc airframe (use with --aircraft c172p_rc)")
    args = parser.parse_args()
    env_kwargs = json.loads(args.env_kwargs)
    obs_gain = [float(x) for x in args.obs_gain.split(",")] if args.obs_gain else None
    if args.rc:
        base = dict(initial_altitude_ft=500, altitude_randomization_ft=100,
                    initial_airspeed_kts=30, airspeed_randomization_kts=4,
                    departed_airspeed_range_kts=(12, 18),
                    departed_yaw_rate_range=(1.5, 3.0),
                    wind_randomization_fps=4)
        if args.rc == "similar":
            base["agent_hz"] = 60
            if obs_gain is None:
                obs_gain = [1, 1, 1, 0.425, 0.425, 0.425, 1, 1, 3.6]
        else:
            base["agent_hz"] = 20
        env_kwargs = {**base, **env_kwargs}
    if obs_gain is not None:
        obs_gain = np.array(obs_gain, dtype=np.float32)
'''),
 ("        latency_steps=args.latency_steps,\n",
  "        latency_steps=args.latency_steps,\n        env_kwargs=env_kwargs,\n        obs_gain=obs_gain,\n"),
]
for old, new in REPL:
    n = src.count(old)
    if n != 1:
        sys.exit(f"Patch not applied: expected 1 match, found {n} for:\n{old}")
    src = src.replace(old, new)
shutil.copy(path, path + ".bak")
open(path, "w").write(src)
print("Patched", path, "(backup:", path + ".bak)")
