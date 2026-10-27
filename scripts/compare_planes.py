import jsbsim

PLANES = ["c172p_rc", "c182", "pa28", "J3Cub", "c310"]

print(f"{'plane':10} {'dt_s':>7} {'weight_lb':>10} {'span_ft':>8} {'area_sqft':>9} {'engines':>8}")
for name in PLANES:
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.load_model(name)
    fdm.set_property_value("ic/h-sl-ft", 3000)
    fdm.set_property_value("ic/vc-kts", 60)
    fdm.run_ic()
    try:
        engines = fdm.get_propulsion().get_num_engines()
    except Exception:
        engines = "?"
    print(
        f"{name:10} {fdm.get_delta_t():7.4f} "
        f"{fdm.get_property_value('inertia/weight-lbs'):10.1f} "
        f"{fdm.get_property_value('metrics/bw-ft'):8.2f} "
        f"{fdm.get_property_value('metrics/Sw-sqft'):9.1f} "
        f"{engines:>8}"
    )