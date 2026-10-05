import jsbsim

root = jsbsim.get_default_root_dir()

def ic_theta(alpha=25, gamma=0.0, phi=0.0, vc=48, p=0.0, r=0.0, alt=5000):
    fdm = jsbsim.FGFDMExec(root)
    fdm.set_debug_level(0)
    fdm.load_model("c172p")
    fdm["ic/h-sl-ft"] = alt
    fdm["ic/vc-kts"] = vc
    fdm["ic/gamma-deg"] = gamma
    fdm["ic/phi-deg"] = phi
    fdm["ic/alpha-deg"] = alpha
    fdm["ic/psi-true-deg"] = 0
    fdm["ic/p-rad_sec"] = p
    fdm["ic/r-rad_sec"] = r
    fdm.run_ic()
    return round(fdm["attitude/theta-deg"], 1)

print("baseline (alpha 25, gamma 0, vc 48):", ic_theta())
for a in (10, 17, 25, 31):    print("alpha", a, "->", ic_theta(alpha=a))
for g in (-10, -5, 0):        print("gamma", g, "->", ic_theta(gamma=g))
for v in (40, 50, 60, 120):   print("vc", v, "->", ic_theta(vc=v))
for p in (-1, 0, 1):          print("p", p, "->", ic_theta(p=p))
for r in (-1.2, 0, 1.2):      print("r", r, "->", ic_theta(r=r))
