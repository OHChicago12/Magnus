import math, jsbsim
root = jsbsim.get_default_root_dir()

def make():
    f = jsbsim.FGFDMExec(root); f.set_debug_level(0); f.load_model("c172p")
    f["ic/h-sl-ft"] = 5000; f["ic/vc-kts"] = 48; f["ic/gamma-deg"] = 0
    return f

def theta(f): return round(f["attitude/theta-deg"], 1)

# (a) env order: bank first, then alpha
for phi in (0, 30, 60, 80, 87):
    f = make(); f["ic/phi-deg"] = phi; f["ic/alpha-deg"] = 25; f.run_ic()
    pred = math.degrees(math.atan(math.tan(math.radians(25)) / math.cos(math.radians(phi))))
    print(f"phi {phi}, alpha 25 -> theta {theta(f)}   (formula {pred:.1f})")

# (b) what my cap did: bank changed AFTER alpha was set
f = make(); f["ic/phi-deg"] = 87; f["ic/alpha-deg"] = 25; f.run_ic()
f["ic/phi-deg"] = 9; f.run_ic()
print("bank 87 then 9, alpha not re-set -> theta", theta(f))

# (c) fix: re-set alpha after changing bank
f["ic/alpha-deg"] = 25; f.run_ic()
print("...then alpha re-set -> theta", theta(f))