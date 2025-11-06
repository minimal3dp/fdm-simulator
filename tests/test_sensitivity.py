import os
import sys

from fastapi.testclient import TestClient

# Ensure project root is on sys.path so `import main` works in CI and local runs
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import main  # noqa: E402


def test_sensitivity_analysis_kaggle():
    """Test sensitivity analysis endpoint with Kaggle model inputs."""
    # Inject minimal materials DB and a mock model for testing
    main.materials_database = {"PLA": {"common": {}, "fea": {}, "density_g_cm3": 1.24}}

    # Create a simple mock model that returns predictable outputs
    class MockModel:
        def predict(self, df):
            # Return fixed predictions for testing - must return numpy array shape
            import numpy as np

            return np.array([[50.0, 10.0, 5.0]])  # tensile_strength, roughness, elongation

    # Ensure MODEL_REGISTRY exists and inject mock model
    if not hasattr(main, "MODEL_REGISTRY") or not main.MODEL_REGISTRY:
        main.MODEL_REGISTRY = {}

    main.MODEL_REGISTRY["kaggle"] = {
        "model": MockModel(),
        "output_names": ["tensile_strength", "roughness", "elongation"],
        "cost_time_params": {},
        "output_model": None,
    }

    client = TestClient(main.app)

    request_body = {
        "model_name": "kaggle",
        "base_inputs": {
            "part_mass_g": 50.0,
            "filament_cost_kg": 25.0,
            "material_name": "PLA",
            "layer_height": 0.2,
            "wall_thickness": 2,
            "infill_density": 20,
            "infill_pattern": "grid",
            "nozzle_temperature": 210,
            "bed_temperature": 60,
            "print_speed": 60,
            "material": "PLA",
            "fan_speed": 100,
        },
        "perturbation_percent": 10.0,
    }

    resp = client.post("/analyze_sensitivity", json=request_body)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "model_name" in data
    assert data["model_name"] == "kaggle"
    assert "baseline_outputs" in data
    assert "parameter_sensitivities" in data
    assert "most_sensitive_params" in data
    assert "least_sensitive_params" in data

    # Check that we have sensitivities for numeric parameters
    assert len(data["parameter_sensitivities"]) > 0

    # Check structure of sensitivity objects
    first_sensitivity = data["parameter_sensitivities"][0]
    assert "parameter_name" in first_sensitivity
    assert "baseline_value" in first_sensitivity
    assert "impact_score" in first_sensitivity
    assert "output_changes" in first_sensitivity
    assert "perturbation_range" in first_sensitivity


def test_sensitivity_analysis_missing_model():
    """Test that sensitivity analysis returns 404 for non-existent model."""
    client = TestClient(main.app)

    request_body = {
        "model_name": "nonexistent_model",
        "base_inputs": {"some_param": 1.0},
        "perturbation_percent": 10.0,
    }

    resp = client.post("/analyze_sensitivity", json=request_body)
    assert resp.status_code == 404
