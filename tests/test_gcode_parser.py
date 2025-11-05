from fastapi.testclient import TestClient
import math

import main


client = TestClient(main.app)


def post_gcode(text: str, **form):
    files = {"file": ("test.gcode", text.encode("utf-8"), "text/plain")}
    data = {}
    data.update({k: str(v) for k, v in form.items() if v is not None})
    resp = client.post("/analyze_gcode", files=files, data=data)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_absolute_vs_relative_extrusion_material_match():
    # Absolute extrusion (M82): total E delta = 2.0 mm
    gcode_abs = "\n".join(
        [
            ";TYPE:WALL-OUTER",
            "M82",
            "G21",
            "G1 X0 Y0 Z0.2 E0",
            "G1 X10 Y0 E1.0",
            "G1 X10 Y10 E2.0",
        ]
    )

    # Relative extrusion (M83): total E delta = 1.0 + 1.0 = 2.0 mm
    gcode_rel = "\n".join(
        [
            ";TYPE:WALL-OUTER",
            "M83",
            "G21",
            "G1 X0 Y0 Z0.2 E0.0",
            "G1 X10 Y0 E1.0",
            "G1 X10 Y10 E1.0",
        ]
    )

    a = post_gcode(gcode_abs)
    b = post_gcode(gcode_rel)

    assert math.isclose(a["material_used_g"], b["material_used_g"], rel_tol=1e-6, abs_tol=1e-6)


def test_inches_vs_mm_material_match():
    # Millimeters: E = 2.0 mm of filament
    gcode_mm = "\n".join(
        [
            ";TYPE:WALL-OUTER",
            "M82",
            "G21",
            "G1 X0 Y0 Z0.2 E0",
            "G1 X10 Y0 E2.0",
        ]
    )

    # Inches: E must be given in inches. 2.0 mm = 0.07874015748 inches
    e_in = 2.0 / 25.4
    gcode_in = "\n".join(
        [
            ";TYPE:WALL-OUTER",
            "M82",
            "G20",
            "G1 X0 Y0 Z0.008 E0",
            f"G1 X0.394 Y0 E{e_in:.8f}",
        ]
    )

    a = post_gcode(gcode_mm)
    b = post_gcode(gcode_in)

    assert math.isclose(a["material_used_g"], b["material_used_g"], rel_tol=1e-4, abs_tol=1e-4)
