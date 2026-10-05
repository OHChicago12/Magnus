import shutil, sys
path = sys.argv[1] if len(sys.argv) > 1 else "scripts/eval_visualize.py"
src = open(path).read()
if '"--baseline"' in src:
    sys.exit("Already patched.")
REPL = [
 ("        action = get_action(module, policy_obs, deterministic=deterministic)\n",
  "        action = get_action(module, policy_obs, deterministic=deterministic)\n"
  "        if globals().get('BASELINE') == 'zero':\n"
  "            action = np.zeros_like(action)\n"),
 ("    args = parser.parse_args()\n",
  "    parser.add_argument('--baseline', choices=['zero'], default=None,\n"
  "                        help='Ignore the policy and send zero actions')\n"
  "    args = parser.parse_args()\n"
  "    globals()['BASELINE'] = args.baseline\n"),
]
for old, new in REPL:
    n = src.count(old)
    if n != 1:
        sys.exit(f"Patch not applied: expected 1 match, found {n} for:\n{old}")
    src = src.replace(old, new)
shutil.copy(path, path + ".bak2")
open(path, "w").write(src)
print("Patched", path, "(backup:", path + ".bak2)")