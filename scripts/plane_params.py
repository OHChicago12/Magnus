"""
Compare physical parameters of the training and test planes, read from their JSBSim model files.
Flags any test plane whose value falls outside the range covered by the training planes.

Run from repo root:  python scripts/plane_params.py
Extra search folders: python scripts/plane_params.py --root path/to/aircraft
"""
import argparse, csv, glob, os
import xml.etree.ElementTree as ET

TRAIN = ["c172p_rc", "c182", "pa28"]
TEST = ["J3Cub", "c310", "t6texan2", "p51d", "pc7"]

LEN = {"FT": 0.3048, "M": 1.0, "IN": 0.0254}
AREA = {"FT2": 0.092903, "M2": 1.0}
MASS = {"LBS": 0.453592, "KG": 1.0, "SLUG": 14.5939}
INERT = {"SLUG*FT2": 1.35582, "KG*M2": 1.0}


def val(root, path, table, default_unit):
    el = root.find(path)
    if el is None or not (el.text or "").strip():
        return None
    unit = (el.get("unit") or default_unit).upper().replace(" ", "")
    f = table.get(unit)
    return None if f is None else float(el.text) * f


def find_xml(name, roots):
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", f"{name}.xml"), recursive=True):
            if any(s in p for s in ("/venv/", "/.git/", "node_modules")) and r == ".":
                continue
            try:
                if ET.parse(p).getroot().tag == "fdm_config":
                    return p
            except ET.ParseError:
                pass
    return None


def params(path):
    r = ET.parse(path).getroot()
    S = val(r, "metrics/wingarea", AREA, "FT2")
    b = val(r, "metrics/wingspan", LEN, "FT")
    c = val(r, "metrics/chord", LEN, "FT")
    m = val(r, "mass_balance/emptywt", MASS, "LBS")
    ixx = val(r, "mass_balance/ixx", INERT, "SLUG*FT2")
    iyy = val(r, "mass_balance/iyy", INERT, "SLUG*FT2")
    izz = val(r, "mass_balance/izz", INERT, "SLUG*FT2")
    d = {"mass_kg": m, "wing_area_m2": S, "span_m": b, "chord_m": c, "Ixx": ixx, "Iyy": iyy, "Izz": izz}
    d["wing_load_kg_m2"] = m / S if m and S else None
    d["aspect_ratio"] = b * b / S if b and S else None
    d["Ixx/(m*b^2)"] = ixx / (m * b * b) if ixx and m and b else None
    d["Iyy/(m*c^2)"] = iyy / (m * c * c) if iyy and m and c else None
    return d


def fmt(x):
    return "   n/a" if x is None else f"{x:10.4g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", action="append", default=[])
    a = p.parse_args()
    roots = list(a.root) + ["."]
    try:
        import jsbsim
        roots.append(os.path.join(jsbsim.get_default_root_dir(), "aircraft"))
    except Exception:
        pass

    rows = {}
    for name in TRAIN + TEST:
        path = find_xml(name, roots)
        if path is None:
            print(f"{name}: model file not found (use --root <folder containing it>)")
            continue
        rows[name] = params(path)
        print(f"{name}: {path}")

    keys = list(next(iter(rows.values())).keys())
    print("\n" + " " * 10 + "".join(f"{k:>18s}" for k in keys))
    for name in TRAIN + TEST:
        if name in rows:
            tag = "train" if name in TRAIN else "TEST "
            print(f"{name:10s}" + "".join(f"{fmt(rows[name][k]):>18s}" for k in keys) + f"  {tag}")

    print("\nTest-plane values outside the training range:")
    train_rows = [rows[n] for n in TRAIN if n in rows]
    for name in TEST:
        if name not in rows:
            continue
        out = []
        for k in keys:
            vals = [r[k] for r in train_rows if r[k] is not None]
            v = rows[name][k]
            if v is None or not vals:
                continue
            if v < min(vals):
                out.append(f"{k} below ({v:.3g} vs min {min(vals):.3g})")
            elif v > max(vals):
                out.append(f"{k} above ({v:.3g} vs max {max(vals):.3g})")
        print(f"  {name}: " + ("; ".join(out) if out else "all inside"))

    with open("plane_params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane", "set"] + keys)
        for name in TRAIN + TEST:
            if name in rows:
                w.writerow([name, "train" if name in TRAIN else "test"] + [rows[name][k] for k in keys])


if __name__ == "__main__":
    main()