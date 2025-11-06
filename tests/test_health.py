import asyncio
import os

import httpx

BASE_URL = os.getenv("FDM_SIM_BASE", "http://127.0.0.1:8000")


async def fetch_json(client: httpx.AsyncClient, method: str, path: str, **kwargs):
    resp = await client.request(method, f"{BASE_URL}{path}", **kwargs)
    is_json = resp.headers.get("content-type", "").startswith("application/json")
    return resp.status_code, (resp.json() if is_json else None)


async def test_materials_and_kaggle():
    async with httpx.AsyncClient(timeout=5) as client:
        # /materials
        status, data = await fetch_json(client, "GET", "/materials")
        assert status == 200, f"/materials status {status}"
        assert isinstance(data, dict) and data, "Materials response empty"
        # Pick a material
        material_name = next(iter(data.keys()))
        # minimal Kaggle body using allowed ranges
        body = {
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
        status2, pred = await fetch_json(client, "POST", "/predict/kaggle", json=body)
        assert status2 == 200, f"/predict/kaggle status {status2}: {pred}"
        for key in [
            "tensile_strength",
            "roughness",
            "elongation",
            "estimated_cost_usd",
            "estimated_print_time_min",
        ]:
            assert key in pred, f"Missing key '{key}' in prediction"

        # Warpage model test
        warpage_body = {
            "part_mass_g": 8.0,
            "filament_cost_kg": 25.0,
            "material_name": material_name,
            "Layer_Temperature_C": 210,
            "Infill_Density_percent": 25,
            "First_Layer_Height_mm": 0.25,
            "Other_Layer_Height_mm": 0.2,
        }
        status3, warpage_pred = await fetch_json(
            client, "POST", "/predict/warpage", json=warpage_body
        )
        assert status3 == 200, f"/predict/warpage status {status3}: {warpage_pred}"
        assert (
            isinstance(warpage_pred, dict) and "Warpage_mm" in warpage_pred
        ), "Missing 'Warpage_mm' in warpage prediction"


# Allow running directly
if __name__ == "__main__":
    asyncio.run(test_materials_and_kaggle())
