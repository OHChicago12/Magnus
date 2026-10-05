"""Per-reset randomised copy of a JSBSim aircraft.

JSBSim exposes mass, CG and aero coefficients as read-only properties, so the
randomisation is done by writing a modified copy of the aircraft XML and
loading that copy. Each process gets one variant folder named
<aircraft>_dr_<pid> inside JSBSim's aircraft folder; it is rewritten on every
sample and removed at exit.

Randomised (all multiplicative unless noted, uniform):
  mass         empty weight, and inertias scale with it
  inertia      extra independent factor per axis (Ixx, Iyy, Izz)
  cg           CG x and z shift in inches (additive)
  authority    Cmde, ClDa, Cndr   (control effectiveness)
  damping      Cmq, Clp, Cnr
  stability    Cmalpha, Clb, Cnb  (a constant <value>, or a single 1-D table, is
                                   scaled; 2-D tables and anything else are skipped)
"""
import atexit
import copy
import os
import shutil
import xml.etree.ElementTree as ET

DEFAULTS = {
    "mass": 0.15,
    "inertia": 0.10,
    "cg_x_in": 1.0,
    "cg_z_in": 0.3,
    "authority": 0.25,
    "damping": 0.25,
    "stability": 0.20,
}
AUTHORITY = ["Cmde", "ClDa", "Cndr"]
DAMPING = ["Cmq", "Clp", "Cnr"]
STABILITY = ["Cmalpha", "Clb", "Cnb"]


def _mul(parent, tag, k):
    el = parent.find(tag)
    if el is None or el.text is None:
        return False
    el.text = repr(float(el.text) * k)
    return True


def _scale_table(prod, k):
    """Scale the output column of a single 1-D <table> inside a product."""
    tables = prod.findall("table")
    if len(tables) != 1:
        return False
    td = tables[0].find("tableData")
    if td is None or td.text is None:
        return False
    rows = []
    for line in td.text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            return False  # 2-D table: leave it alone
        rows.append((parts[0], float(parts[1])))
    if not rows:
        return False
    pad = " " * 30
    td.text = "\n" + "\n".join(f"{pad}{x}   {repr(y * k)}" for x, y in rows) + "\n" + " " * 26
    return True


def _scale_function(root, short_name, k):
    """Multiply the single constant <value> of a simple coefficient function."""
    full = "aero/coefficient/" + short_name
    for fn in root.iter("function"):
        if fn.get("name") == full:
            prod = fn.find("product")
            if prod is None:
                return False
            vals = prod.findall("value")
            if len(vals) == 1:
                vals[0].text = repr(float(vals[0].text) * k)
                return True
            if len(vals) == 0:
                return _scale_table(prod, k)
            return False
    return False


class AirframeVariant:
    def __init__(self, root_dir, aircraft):
        self.aircraft = aircraft
        src_dir = os.path.join(root_dir, "aircraft", aircraft)
        self.template = ET.parse(os.path.join(src_dir, aircraft + ".xml")).getroot()
        self.name = f"{aircraft}_dr_{os.getpid()}"
        self.dir = os.path.join(root_dir, "aircraft", self.name)
        os.makedirs(self.dir, exist_ok=True)
        # engines, systems etc. live in sub-folders next to the XML
        for sub in os.listdir(src_dir):
            p = os.path.join(src_dir, sub)
            if os.path.isdir(p):
                dst = os.path.join(self.dir, sub)
                if not os.path.exists(dst):
                    shutil.copytree(p, dst)
        atexit.register(shutil.rmtree, self.dir, True)

    def sample(self, rng, cfg=True):
        c = dict(DEFAULTS)
        if isinstance(cfg, dict):
            c.update(cfg)

        def u(w):
            return float(rng.uniform(1.0 - w, 1.0 + w))

        root = copy.deepcopy(self.template)
        params = {}

        mb = root.find("mass_balance")
        if mb is not None:
            m = u(c["mass"])
            params["mass"] = m
            _mul(mb, "emptywt", m)
            for ax in ("ixx", "iyy", "izz"):
                k = m * u(c["inertia"])
                params[ax] = k
                _mul(mb, ax, k)
            for loc in mb.findall("location"):
                if loc.get("name") == "CG" and loc.get("unit", "IN").upper() == "IN":
                    dx = float(rng.uniform(-c["cg_x_in"], c["cg_x_in"]))
                    dz = float(rng.uniform(-c["cg_z_in"], c["cg_z_in"]))
                    for tag, d in (("x", dx), ("z", dz)):
                        el = loc.find(tag)
                        if el is not None and el.text is not None:
                            el.text = repr(float(el.text) + d)
                    params["cg_dx_in"] = dx
                    params["cg_dz_in"] = dz

        scaled = {}
        for names, key in ((AUTHORITY, "authority"), (DAMPING, "damping"),
                           (STABILITY, "stability")):
            for n in names:
                k = u(c[key])
                if _scale_function(root, n, k):
                    scaled[n] = k
        params["coef"] = scaled

        path = os.path.join(self.dir, self.name + ".xml")
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return self.name, params
