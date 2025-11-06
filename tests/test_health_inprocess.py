import pytest


@pytest.mark.skipif("TestClient" not in globals(), reason="TestClient not available")
def test_materials_and_predictions_inprocess(test_client):
    # /materials
    r = test_client.get("/materials")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict) and data
    material_name = next(iter(data.keys()))

    # Kaggle
    kaggle_body = {
        "part_mass_g": 10.0,
        "filament_cost_kg": 25.0,
        "material_name": material_name,
        "layer_height": 0.2,
        "wall_thickness": 1.2,
        "infill_density": 20,
        "infill_pattern": "grid",
        "nozzle_temperature": 205,
        "bed_temperature": 60,
        "print_speed": 50,
        "material": material_name,
        "fan_speed": 100,
    }
    r2 = test_client.post("/predict/kaggle", json=kaggle_body)
    assert r2.status_code == 200
    pred = r2.json()
    for key in [
        "tensile_strength",
        "roughness",
        "elongation",
        "estimated_cost_usd",
        "estimated_print_time_min",
    ]:
        assert key in pred

    # Warpage
    warpage_body = {
        "part_mass_g": 8.0,
        "filament_cost_kg": 25.0,
        "material_name": material_name,
        "Layer_Temperature_C": 210,
        "Infill_Density_percent": 25,
        "First_Layer_Height_mm": 0.25,
        "Other_Layer_Height_mm": 0.2,
    }
    r3 = test_client.post("/predict/warpage", json=warpage_body)
    assert r3.status_code == 200
    wpred = r3.json()
    assert "Warpage_mm" in wpred
