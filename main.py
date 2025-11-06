import io
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymoo.algorithms.moo.nsga2 import NSGA2

# Optional: External G-code parser not required for our analysis; using manual parsing
# --- v4: Optimization Imports ---
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.spatial.distance import directed_hausdorff

# --- Configuration (centralized) ---
from config import (  # noqa: E402
    FEA_TARGETS_PATH,
    MATERIALS_FILE,
    MODEL_ACCURACY_PATH,
    MODEL_C3_PATH,
    MODEL_COMPOSITE_PATH,
    MODEL_FATIGUE_PATH,
    MODEL_FEA_PATH,
    MODEL_HARDNESS_PATH,
    MODEL_KAGGLE_PATH,
    MODEL_MULTIMATERIAL_PATH,
    MODEL_WARPAGE_PATH,
    MODELS_DIR,
)

MODELS_DIR = str(MODELS_DIR)  # keep existing string usage without large refactor

# --- FastAPI App Initialization ---
app = FastAPI(title="FDM 3D Print Property Simulator API (v12 with Sensitivity Analysis)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Model Storage ---
materials_database = {}
fea_target_names = []
# v8: Refactored model storage
MODEL_REGISTRY: dict[str, dict[str, Any]] = {}

# --- Pydantic Models (Data Validation) ---


# v3: Changed part_volume_cm3 to part_mass_g
class GlobalInputs(BaseModel):
    part_mass_g: float = Field(..., gt=0, description="Mass of the part in grams.")
    filament_cost_kg: float = Field(
        ..., gt=0, description="Cost of the filament spool per kilogram."
    )
    material_name: str = Field(..., description="Name of the material (e.g., 'PLA', 'ABS')")


# v3: New global prediction outputs
class PredictionBase(BaseModel):
    estimated_cost_usd: float
    estimated_print_time_min: float


# --- Model 1: Kaggle ---
class KaggleInput(GlobalInputs):
    layer_height: float
    wall_thickness: int
    infill_density: int
    infill_pattern: str
    nozzle_temperature: int
    bed_temperature: int
    print_speed: int
    material: str
    fan_speed: int


class KaggleOutput(PredictionBase):
    tensile_strength: float
    roughness: float
    elongation: float


# --- Model 2: C3 ---
class C3Input(GlobalInputs):
    Temperature: float
    Speed: float
    Angle: float
    Height: float
    Fill: float


class C3Output(PredictionBase):
    Tensile_Strength_MPa: float
    Elongation_at_Break_percent: float


# --- Model 3: FEA ---
class FEAInput(GlobalInputs):
    Material_Bonding_Perfection: float
    Material_Youngs_Modulus_GPa: float
    Material_Tensile_Yield_Strenght_MPa: float
    Material_Poissons_Ratio: float
    User_Infill_Pattern: str
    User_Infill_Density: float  # This is 0.1 - 1.0
    User_Line_Thickenss_mm: float
    User_Layer_Height_mm: float


class FEAOutput(PredictionBase):
    properties: dict[str, float]


# --- Model 4: Fatigue ---
class FatigueInput(GlobalInputs):
    Nozzle_Diameter: float
    Print_Speed: float
    Nozzle_Temperature: float
    Stress_Level: float


class FatigueOutput(PredictionBase):
    Fatigue_Lifetime: float


# --- v5: Model 5: Dimensional Accuracy ---
class AccuracyInput(GlobalInputs):
    Layer_Thickness_mm: float
    Build_Orientation_deg: int
    Infill_Density_percent: int
    Number_of_Contours: int


class AccuracyOutput(PredictionBase):
    Var_Length_percent: float
    Var_Width_percent: float
    Var_Thickness_percent: float


# --- v6: Model 6: Warp Deformation ---
class WarpageInput(GlobalInputs):
    Layer_Temperature_C: int
    Infill_Density_percent: int
    First_Layer_Height_mm: float
    Other_Layer_Height_mm: float


class WarpageOutput(PredictionBase):
    Warpage_mm: float


# --- v7: Model 7: Hardness ---
class HardnessInput(GlobalInputs):
    Layer_Thickness_mm: float
    Shell_Thickness_mm: float
    Fill_Density_percent: int
    Fill_Pattern: str


class HardnessOutput(PredictionBase):
    Hardness_Shore_D: float


# --- v8: Model 8: Multi-Material Bond Strength ---
class MultiMaterialInput(GlobalInputs):
    Material_A: str
    Material_B: str
    Layer_Height_mm: float
    Extrusion_Temp_C: int
    Infill_Density_percent: int


class MultiMaterialOutput(PredictionBase):
    Tensile_Strength_MPa: float


# --- v9: Model 9: Composite Filaments ---
class CompositeInput(GlobalInputs):
    Reinforcement_percent: float
    Infill_Pattern: str
    Infill_Density_percent: int
    Layer_Thickness_mm: float


class CompositeOutput(PredictionBase):
    Tensile_Strength_MPa: float
    Elastic_Modulus_GPa: float


# --- v4: Optimization Pydantic Models ---
class OptimizationObjective(BaseModel):
    name: str  # e.g., "tensile_strength", "estimated_cost_usd"
    goal: str  # "minimize" or "maximize"


class OptimizationConstraint(BaseModel):
    name: str  # e.g., "print_speed"
    operator: str  # "lt" (less than) or "gt" (greater than)
    value: float


class OptimizationRequest(BaseModel):
    model_name: str
    objectives: list[OptimizationObjective]
    constraints: list[OptimizationConstraint]
    global_inputs: GlobalInputs  # part_mass_g, filament_cost_kg, material_name


# --- v8: Refactored Model Configuration ---
def get_model_registry() -> dict[str, dict[str, Any]]:
    return {
        "kaggle": {
            "path": str(MODEL_KAGGLE_PATH),
            "model": None,
            "input_model": KaggleInput,
            "output_model": KaggleOutput,
            "output_names": ["tensile_strength", "roughness", "elongation"],
            "cost_time_params": {
                'infill': 'infill_density',
                'layer_height': 'layer_height',
                'speed': 'print_speed',
            },
            "optimizer_config": {
                "numeric": [
                    ("layer_height", 0.02, 0.3),
                    ("wall_thickness", 1, 10),
                    ("infill_density", 10, 100),
                    ("nozzle_temperature", 200, 250),
                    ("bed_temperature", 50, 100),
                    ("print_speed", 40, 120),
                    ("fan_speed", 0, 100),
                ],
                "categorical": [
                    ("infill_pattern", ["grid", "honeycomb"]),
                    ("material", ["abs", "pla"]),
                ],
            },
        },
        "c3": {
            "path": str(MODEL_C3_PATH),
            "model": None,
            "input_model": C3Input,
            "output_model": C3Output,
            "output_names": ["Tensile_Strength_MPa", "Elongation_at_Break_percent"],
            "cost_time_params": {'infill': 'Fill', 'layer_height': 'Height', 'speed': 'Speed'},
            "optimizer_config": None,  # Not implemented for optimizer
        },
        "fea": {
            "path": str(MODEL_FEA_PATH),
            "model": None,
            "input_model": FEAInput,
            "output_model": FEAOutput,
            "output_names": [],  # Loaded dynamically from fea_target_names
            "cost_time_params": {
                'infill': 'User_Infill_Density_100',
                'layer_height': 'User_Layer_Height_mm',
                'speed': 60.0,
            },
            "optimizer_config": None,  # FEA is too complex for this optimizer
        },
        "fatigue": {
            "path": str(MODEL_FATIGUE_PATH),
            "model": None,
            "input_model": FatigueInput,
            "output_model": FatigueOutput,
            "output_names": ["Fatigue_Lifetime"],
            "cost_time_params": {'infill': 80.0, 'layer_height': 0.2, 'speed': 'Print_Speed'},
            "optimizer_config": None,  # Not implemented for optimizer
        },
        "accuracy": {
            "path": str(MODEL_ACCURACY_PATH),
            "model": None,
            "input_model": AccuracyInput,
            "output_model": AccuracyOutput,
            "output_names": ["Var_Length_percent", "Var_Width_percent", "Var_Thickness_percent"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent',
                'layer_height': 'Layer_Thickness_mm',
                'speed': 60.0,
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Thickness_mm", 0.1, 0.4),
                    ("Build_Orientation_deg", 0, 90),
                    ("Infill_Density_percent", 80, 100),
                    ("Number_of_Contours", 1, 3),
                ],
                "categorical": [],
            },
        },
        "warpage": {
            "path": str(MODEL_WARPAGE_PATH),
            "model": None,
            "input_model": WarpageInput,
            "output_model": WarpageOutput,
            "output_names": ["Warpage_mm"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent',
                'layer_height': 'Other_Layer_Height_mm',
                'speed': 60.0,
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Temperature_C", 190, 210),
                    ("Infill_Density_percent", 10, 30),
                    ("First_Layer_Height_mm", 0.2, 0.4),
                    ("Other_Layer_Height_mm", 0.3, 0.5),
                ],
                "categorical": [],
            },
        },
        "hardness": {
            "path": str(MODEL_HARDNESS_PATH),
            "model": None,
            "input_model": HardnessInput,
            "output_model": HardnessOutput,
            "output_names": ["Hardness_Shore_D"],
            "cost_time_params": {
                'infill': 'Fill_Density_percent',
                'layer_height': 'Layer_Thickness_mm',
                'speed': 60.0,
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Thickness_mm", 0.2, 0.5),
                    ("Shell_Thickness_mm", 1.2, 1.6),
                    ("Fill_Density_percent", 20, 80),
                ],
                "categorical": [("Fill_Pattern", ["Rectilinear", "Honey_Comb"])],
            },
        },
        "multimaterial": {
            "path": str(MODEL_MULTIMATERIAL_PATH),
            "model": None,
            "input_model": MultiMaterialInput,
            "output_model": MultiMaterialOutput,
            "output_names": ["Tensile_Strength_MPa"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent',
                'layer_height': 'Layer_Height_mm',
                'speed': 60.0,
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Height_mm", 0.2, 0.4),
                    ("Extrusion_Temp_C", 230, 250),
                    ("Infill_Density_percent", 60, 100),
                ],
                "categorical": [("Material_A", ["ABS"]), ("Material_B", ["PETG"])],
            },
        },
        "composite": {
            "path": str(MODEL_COMPOSITE_PATH),
            "model": None,
            "input_model": CompositeInput,
            "output_model": CompositeOutput,
            "output_names": ["Tensile_Strength_MPa", "Elastic_Modulus_GPa"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent',
                'layer_height': 'Layer_Thickness_mm',
                'speed': 60.0,
            },
            "optimizer_config": {
                "numeric": [
                    ("Reinforcement_percent", 0, 40),
                    ("Infill_Density_percent", 20, 100),
                    ("Layer_Thickness_mm", 0.1, 0.4),
                ],
                "categorical": [("Infill_Pattern", ["Grid", "Tri-Hexagon", "Gyroid"])],
            },
        },
    }


# --- v3: Cost & Time Calculation Logic ---


def get_material_density(material_name: str) -> float:
    """Gets density from the loaded materials database, with a fallback."""
    # Find the material in the database, case-insensitive
    for key, data in materials_database.items():
        if key.lower() == material_name.lower():
            if 'density_g_cm3' in data:
                return data['density_g_cm3']

    # Fallback density if not found
    if 'pla' in material_name.lower():
        return 1.24
    if 'abs' in material_name.lower():
        return 1.04
    if 'petg' in material_name.lower():
        return 1.27

    return 1.25  # Average density


def calculate_cost_time(inputs: GlobalInputs, model_params: dict[str, float]) -> dict[str, float]:
    """Calculates estimated cost and print time."""

    # Extract global inputs
    part_mass_g = inputs.part_mass_g
    filament_cost_kg = inputs.filament_cost_kg

    # --- Cost Calculation ---
    part_mass_kg = part_mass_g / 1000.0
    estimated_cost_usd = part_mass_kg * filament_cost_kg

    # --- Time Calculation ---
    material_density = get_material_density(inputs.material_name)
    if material_density == 0:
        material_density = 1.25  # Prevent division by zero

    part_volume_cm3 = part_mass_g / material_density

    # Extract model-specific parameters
    infill_percent = model_params.get('infill', 80.0)  # Infill as 0-100
    layer_height_mm = model_params.get('layer_height', 0.2)
    print_speed_mm_s = model_params.get('speed', 60.0)

    # Ensure infill is a percentage (0-100)
    if infill_percent <= 1.0 and infill_percent > 0:
        infill_percent *= 100  # Convert 0.5 to 50

    # This is a very rough estimate.
    extrusion_width_mm = 0.4
    volume_to_extrude_mm3 = part_volume_cm3 * (infill_percent / 100.0) * 1000.0  # cm3 to mm3

    # Prevent division by zero if speed or layer height is 0
    if print_speed_mm_s <= 0 or layer_height_mm <= 0:
        estimated_print_time_min = 0.0
    else:
        flow_rate_mm3_s = layer_height_mm * extrusion_width_mm * print_speed_mm_s
        estimated_print_time_sec = volume_to_extrude_mm3 / flow_rate_mm3_s
        estimated_print_time_min = estimated_print_time_sec / 60.0

    return {
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_print_time_min": estimated_print_time_min,
    }


# --- Server Startup ---
@app.on_event("startup")
async def startup_event():
    """Load all models and data on server startup."""
    global materials_database, fea_target_names, MODEL_REGISTRY

    # Load Materials JSON
    try:
        with open(MATERIALS_FILE) as f:
            materials_database = json.load(f)
        print(f"Successfully loaded '{MATERIALS_FILE}'")
    except Exception as e:
        print(f"Warning: '{MATERIALS_FILE}' not found or corrupt. Using empty database. Error: {e}")

    # Load FEA Target Names (special case)
    try:
        fea_target_names = joblib.load(FEA_TARGETS_PATH)
        print(f"Successfully loaded FEA target names ({len(fea_target_names)} properties).")
    except Exception:
        print(
            "Warning: FEA target names file not found at "
            f"{FEA_TARGETS_PATH}. "
            "FEA endpoint may fail."
        )

    # v8: Refactored model loading
    MODEL_REGISTRY = get_model_registry()
    for model_name, config in MODEL_REGISTRY.items():
        try:
            config["model"] = joblib.load(config["path"])
            print(f"Successfully loaded model '{model_name}' from {config['path']}")

            # Special case: dynamically set output names for FEA
            if model_name == "fea":
                config["output_names"] = fea_target_names

        except Exception as e:
            print(
                "Warning: Model file not found at "
                f"{config['path']}. "
                f"Endpoint for '{model_name}' will be disabled. "
                f"Error: {e}"
            )


# --- API Endpoints ---


@app.get("/")
def read_root():
    return {"message": "FDM Property Simulator API (v8) is running."}


@app.get("/materials", response_model=dict[str, dict])
async def get_materials():
    """Serves the loaded materials.json file."""
    if not materials_database:
        raise HTTPException(status_code=404, detail="Materials database is not loaded.")
    return materials_database


# --- v8: Refactored Generic Prediction Logic ---


async def _run_prediction(model_name: str, inputs: GlobalInputs):
    """Generic helper to run any prediction."""
    config = MODEL_REGISTRY.get(model_name)
    if not config or config.get("model") is None:
        raise HTTPException(
            status_code=503, detail=f"{model_name.capitalize()} model is not loaded."
        )

    try:
        # 1. Get cost/time info
        model_params = {}
        input_dict = inputs.dict()
        for key, val_key in config["cost_time_params"].items():
            if isinstance(val_key, str):
                model_params[key] = input_dict.get(val_key)
            else:
                model_params[key] = val_key  # Use default value

        # Special case for FEA infill (0.1-1.0) -> (10-100)
        if model_name == "fea":
            model_params['infill'] = input_dict.get('User_Infill_Density', 0.5) * 100.0

        cost_time_results = calculate_cost_time(inputs, model_params)

        # 2. Get model prediction
        features_dict = inputs.dict(exclude={'part_mass_g', 'filament_cost_kg', 'material_name'})

        # Special case for FEA feature name mapping
        if model_name == "fea":
            features_dict['Material_Youngs_Modulus_GPa'] = features_dict.pop(
                'Material_Youngs_Modulus_GPa'
            )
            features_dict['Material_Tensile_Yield_Strenght_MPa'] = features_dict.pop(
                'Material_Tensile_Yield_Strenght_MPa'
            )
            features_dict['Material_Poissons_Ratio'] = features_dict.pop('Material_Poissons_Ratio')

        features = pd.DataFrame([features_dict])
        prediction = config["model"].predict(features)[0]

        # 3. Combine and return
        OutputModel = config["output_model"]
        output_data = {}

        if model_name == "fea":
            output_data["properties"] = dict(zip(fea_target_names, prediction))
        elif isinstance(  # noqa: UP038
            prediction,
            (np.ndarray, list),  # Multi-output (Kaggle, Accuracy)
        ):
            for i, name in enumerate(config["output_names"]):
                output_data[name] = prediction[i]
        else:  # Single-output (C3, Fatigue, Warpage, Hardness, MultiMaterial)
            output_data[config["output_names"][0]] = prediction

        return OutputModel(**output_data, **cost_time_results)

    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: columns are missing: {e}")
    except Exception as e:
        print(f"Error during prediction for {model_name}: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# --- Dynamic Prediction Endpoints ---
@app.post("/predict/kaggle", response_model=KaggleOutput)
async def predict_kaggle(inputs: KaggleInput):
    return await _run_prediction("kaggle", inputs)


@app.post("/predict/c3", response_model=C3Output)
async def predict_c3(inputs: C3Input):
    return await _run_prediction("c3", inputs)


@app.post("/predict/fea", response_model=FEAOutput)
async def predict_fea(inputs: FEAInput):
    return await _run_prediction("fea", inputs)


@app.post("/predict/fatigue", response_model=FatigueOutput)
async def predict_fatigue(inputs: FatigueInput):
    return await _run_prediction("fatigue", inputs)


@app.post("/predict/accuracy", response_model=AccuracyOutput)
async def predict_accuracy(inputs: AccuracyInput):
    return await _run_prediction("accuracy", inputs)


@app.post("/predict/warpage", response_model=WarpageOutput)
async def predict_warpage(inputs: WarpageInput):
    return await _run_prediction("warpage", inputs)


@app.post("/predict/hardness", response_model=HardnessOutput)
async def predict_hardness(inputs: HardnessInput):
    return await _run_prediction("hardness", inputs)


@app.post("/predict/multimaterial", response_model=MultiMaterialOutput)
async def predict_multimaterial(inputs: MultiMaterialInput):
    return await _run_prediction("multimaterial", inputs)


@app.post("/predict/composite", response_model=CompositeOutput)
async def predict_composite(inputs: CompositeInput):
    return await _run_prediction("composite", inputs)


# --- v12: Sensitivity Analysis Endpoint ---


class SensitivityRequest(BaseModel):
    """Request for sensitivity analysis on a specific model."""

    model_name: str
    base_inputs: dict[str, Any]  # Baseline parameter values
    perturbation_percent: float = Field(
        default=10.0, gt=0, le=50, description="Percentage to perturb each parameter (1-50%)"
    )


class ParameterSensitivity(BaseModel):
    """Sensitivity metrics for a single parameter."""

    parameter_name: str
    baseline_value: float | str
    impact_score: float  # Normalized 0-100, higher = more sensitive
    output_changes: dict[str, float]  # For each output metric, % change when parameter is perturbed
    perturbation_range: dict[str, float]  # min/max values tested


class SensitivityResult(BaseModel):
    """Complete sensitivity analysis result."""

    model_name: str
    baseline_outputs: dict[str, float]
    parameter_sensitivities: list[ParameterSensitivity]
    most_sensitive_params: list[str]  # Top 3 most impactful parameters
    least_sensitive_params: list[str]  # Top 3 least impactful parameters


@app.post("/analyze_sensitivity", response_model=SensitivityResult)
async def analyze_sensitivity(request: SensitivityRequest):
    """
    Perform sensitivity analysis to quantify how each input parameter affects outputs.
    Uses one-at-a-time (OAT) perturbation method.
    """
    model_config = MODEL_REGISTRY.get(request.model_name)
    if not model_config or model_config.get("model") is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' not loaded or not found.",
        )

    try:
        model = model_config["model"]
        output_names = model_config["output_names"]

        # Get baseline prediction
        base_features = {
            k: v
            for k, v in request.base_inputs.items()
            if k not in ['part_mass_g', 'filament_cost_kg', 'material_name']
        }
        base_df = pd.DataFrame([base_features])
        base_prediction = model.predict(base_df)[0]

        # Format baseline outputs
        baseline_outputs = {}
        if request.model_name == "fea":
            baseline_outputs = dict(zip(fea_target_names, base_prediction))
        elif isinstance(base_prediction, (np.ndarray, list)):  # noqa: UP038
            for i, name in enumerate(output_names):
                baseline_outputs[name] = float(base_prediction[i])
        else:
            baseline_outputs[output_names[0]] = float(base_prediction)

        # Perform sensitivity analysis on each numeric parameter
        sensitivities = []
        perturbation_factor = request.perturbation_percent / 100.0

        for param_name, param_value in base_features.items():
            if not isinstance(param_value, (int, float)):  # noqa: UP038
                continue  # Skip categorical parameters for now

            # Perturb parameter up and down
            perturbed_predictions = []
            perturb_values = []

            for direction in [-1, 1]:
                perturbed_features = base_features.copy()
                delta = param_value * perturbation_factor * direction
                new_value = param_value + delta
                perturbed_features[param_name] = new_value
                perturb_values.append(new_value)

                perturbed_df = pd.DataFrame([perturbed_features])
                pred = model.predict(perturbed_df)[0]
                perturbed_predictions.append(pred)

            # Calculate output changes
            output_changes = {}
            total_impact = 0.0

            # Determine output names based on model type
            if request.model_name == "fea":
                active_output_names = fea_target_names
            elif isinstance(base_prediction, (np.ndarray, list)):  # noqa: UP038
                active_output_names = output_names
            else:
                active_output_names = [output_names[0]]

            for i, output_name in enumerate(active_output_names):
                if request.model_name == "fea":
                    base_val = baseline_outputs[output_name]
                    perturbed_vals = [pred[i] for pred in perturbed_predictions]
                elif isinstance(base_prediction, (np.ndarray, list)):  # noqa: UP038
                    base_val = baseline_outputs[output_name]
                    perturbed_vals = [pred[i] for pred in perturbed_predictions]
                else:
                    base_val = baseline_outputs[output_names[0]]
                    perturbed_vals = perturbed_predictions

                # Calculate average absolute percentage change
                if base_val != 0:
                    pct_changes = [
                        abs((pval - base_val) / base_val * 100) for pval in perturbed_vals
                    ]
                    avg_pct_change = np.mean(pct_changes)
                else:
                    avg_pct_change = 0.0

                output_changes[output_name] = round(avg_pct_change, 2)
                total_impact += avg_pct_change

            # Normalize impact score to 0-100
            impact_score = float(total_impact) / len(output_changes) if output_changes else 0.0

            sensitivities.append(
                ParameterSensitivity(
                    parameter_name=param_name,
                    baseline_value=param_value,
                    impact_score=float(round(impact_score, 2)),
                    output_changes=output_changes,
                    perturbation_range={
                        "min": round(min(perturb_values), 3),
                        "max": round(max(perturb_values), 3),
                    },
                )
            )

        # Sort by impact score
        sensitivities.sort(key=lambda x: x.impact_score, reverse=True)

        # Identify most and least sensitive parameters
        most_sensitive = [s.parameter_name for s in sensitivities[:3]]
        least_sensitive = [s.parameter_name for s in sensitivities[-3:]]

        return SensitivityResult(
            model_name=request.model_name,
            baseline_outputs=baseline_outputs,
            parameter_sensitivities=sensitivities,
            most_sensitive_params=most_sensitive,
            least_sensitive_params=least_sensitive,
        )

    except Exception as e:
        print(f"Sensitivity analysis error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis failed: {str(e)}")


# --- v4: Optimization Endpoint ---


class OptimizationProblem(Problem):
    """Defines the optimization problem for pymoo."""

    def __init__(self, model_name: str, request: OptimizationRequest, param_config: dict[str, Any]):
        self.model = MODEL_REGISTRY[model_name]["model"]
        self.model_name = model_name
        self.request = request
        self.param_config = param_config

        self.numeric_params = [p[0] for p in param_config["numeric"]]
        self.categorical_params = [p[0] for p in param_config["categorical"]]

        # Create mapping for categorical variables (e.g., "grid" -> 0)
        self.cat_mappings = {}
        for name, options in param_config["categorical"]:
            self.cat_mappings[name] = {option: i for i, option in enumerate(options)}
            self.cat_mappings[f"{name}_reverse"] = {i: option for i, option in enumerate(options)}

        # --- Define problem bounds (xl, xu) ---
        n_numeric = len(self.numeric_params)
        n_categorical = len(self.categorical_params)
        n_var = n_numeric + n_categorical

        xl = []
        xu = []
        # Numeric bounds
        for _, min_val, max_val in param_config["numeric"]:
            xl.append(min_val)
            xu.append(max_val)
        # Categorical bounds (integer indices)
        for name, options in param_config["categorical"]:
            xl.append(0)
            xu.append(len(options) - 1)

        # --- Define objectives and constraints ---
        n_obj_local = len(request.objectives)
        n_constr_local = len(request.constraints)

        # Determine if variables are integers
        # NOTE: Avoid using attribute name `vtype` because pymoo's Problem may define it internally.
        var_types_local = [float] * n_numeric
        for name, _ in param_config.get("categorical", []):
            var_types_local.append(int)

        # Handle integer numeric variables
        int_params = [
            "wall_thickness",
            "infill_density",
            "nozzle_temperature",
            "bed_temperature",
            "print_speed",
            "fan_speed",
            "Build_Orientation_deg",
            "Infill_Density_percent",
            "Number_of_Contours",
            "Layer_Temperature_C",
            "Infill_Density_percent",
            "Fill_Density_percent",
            "Extrusion_Temp_C",
            "Infill_Density_percent",
            # v9: Composite params (PascalCase)
            "Print_Speed",
            "Nozzle_Temperature",
            "Bed_Temperature",
        ]
        for i, (name, _, _) in enumerate(param_config["numeric"]):
            if name in int_params:
                var_types_local[i] = int

        # Pass local n_obj and n_constr to super()
        super().__init__(n_var=n_var, n_obj=n_obj_local, n_constr=n_constr_local, xl=xl, xu=xu)

        # Assign after super().__init__ to prevent any overwriting by base class initialization
        self.var_types = var_types_local

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a 2D array (n_solutions, n_vars)

        n_sols = x.shape[0]
        n_numeric = len(self.numeric_params)

        # 1. Decode the input array 'x' into a list of feature dictionaries
        solutions_df_list = []
        for sol_vector in x:
            sol_dict = {}
            # Decode numeric and round integers
            for i, name in enumerate(self.numeric_params):
                val = sol_vector[i]
                if self.var_types[i] is int:
                    val = int(round(val))
                sol_dict[name] = val
            # Decode categorical
            for i, name in enumerate(self.categorical_params):
                cat_index = int(round(sol_vector[n_numeric + i]))
                if name in self.cat_mappings:
                    sol_dict[name] = self.cat_mappings[f"{name}_reverse"][cat_index]
            solutions_df_list.append(sol_dict)

        features_df = pd.DataFrame(solutions_df_list)

        # 2. Get model predictions for all solutions in a batch
        predictions = self.model.predict(features_df)
        pred_map = {}

        # v8: Refactored Prediction Mapping
        model_config = MODEL_REGISTRY[self.model_name]
        output_names = model_config["output_names"]
        param_map_key = model_config["cost_time_params"]

        if isinstance(  # noqa: UP038
            predictions[0],
            (np.ndarray, list),  # Multi-output (Kaggle, Accuracy)
        ):
            for i, name in enumerate(output_names):
                pred_map[name] = predictions[:, i]
        else:  # Single-output
            pred_map[output_names[0]] = predictions

        # 3. Calculate cost/time for all solutions
        cost_time_list = []
        for i in range(n_sols):
            param_map = {}
            for key, val_key in param_map_key.items():
                if isinstance(val_key, str):
                    param_map[key] = features_df.loc[i, val_key]
                else:
                    param_map[key] = val_key  # Use default value

            # Special case for FEA infill
            if self.model_name == "fea":
                param_map['infill'] = features_df.loc[i, 'User_Infill_Density'] * 100.0

            cost_time = calculate_cost_time(self.request.global_inputs, param_map)
            cost_time_list.append(cost_time)

        cost_time_df = pd.DataFrame(cost_time_list)
        pred_map["estimated_cost_usd"] = cost_time_df["estimated_cost_usd"].values
        pred_map["estimated_print_time_min"] = cost_time_df["estimated_print_time_min"].values

        # 4. Calculate objectives (F)
        f_matrix = np.zeros((n_sols, self.n_obj))
        for i, obj in enumerate(self.request.objectives):
            if obj.name not in pred_map:
                f_matrix[:, i] = 0.0  # Objective not found
            else:
                values = pred_map[obj.name]
                if obj.goal == "maximize":
                    f_matrix[:, i] = -values  # pymoo minimizes by default
                else:
                    f_matrix[:, i] = values

        out["F"] = f_matrix

        # 5. Calculate constraints (G)
        if self.n_constr > 0:
            g_matrix = np.zeros((n_sols, self.n_constr))
            for i, constr in enumerate(self.request.constraints):
                current_values = features_df[constr.name].values
                if constr.operator == "lt":
                    # g(x) <= 0
                    g_matrix[:, i] = current_values - constr.value
                elif constr.operator == "gt":
                    # g(x) >= 0  ->  -g(x) <= 0
                    g_matrix[:, i] = constr.value - current_values
            out["G"] = g_matrix


@app.post("/optimize")
async def optimize(request: OptimizationRequest):
    # v8: Refactored config lookup
    model_config = MODEL_REGISTRY.get(request.model_name)

    if model_config is None or model_config.get("model") is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_name}' not loaded or not found."
        )

    param_config = model_config.get("optimizer_config")

    if param_config is None:
        raise HTTPException(
            status_code=404, detail=f"Optimization not supported for model '{request.model_name}'."
        )

    try:
        # 1. Initialize the problem
        problem = OptimizationProblem(request.model_name, request, param_config)

        # 2. Initialize the algorithm
        algorithm = NSGA2(
            pop_size=50,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        # 3. Define termination
        termination = get_termination("n_gen", 40)

        # 4. Run the optimization
        print(f"Starting optimization for {request.model_name}...")
        res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=False)
        print("Optimization complete.")

        # 5. Format and return the results
        results = []

        n_numeric = len(param_config["numeric"])

        # --- FIX for 'NoneType' error ---
        # Check if optimization returned valid results
        if res.F is None or res.X is None:
            print("Warning: Optimization returned no valid solutions (res.F or res.X is None).")
            return []

        objective_values = res.F

        for i, sol_vector in enumerate(res.X):
            solution = {}
            inputs = {}
            outputs = {}

            # Decode input parameters
            for j, (name, _, _) in enumerate(param_config["numeric"]):
                val = sol_vector[j]
                if problem.var_types[j] is int:  # Use type info from problem
                    val = int(round(val))
                inputs[name] = round(val, 2)
            for j, (name, _) in enumerate(param_config["categorical"]):
                cat_index = int(round(sol_vector[n_numeric + j]))
                inputs[name] = param_config["categorical"][j][1][cat_index]  # Get string value

            # Get output objectives
            for j, obj in enumerate(request.objectives):
                value = objective_values[i, j]
                if obj.goal == "maximize":
                    value = -value  # Flip back to positive
                outputs[obj.name] = round(value, 2)

            solution["inputs"] = inputs
            solution["outputs"] = outputs
            results.append(solution)

        # Sort results by the first objective
        if not results:
            return []

        first_obj_name = request.objectives[0].name
        first_obj_goal = request.objectives[0].goal
        reverse = True if first_obj_goal == "maximize" else False

        sorted_results = sorted(
            results, key=lambda r: r['outputs'][first_obj_name], reverse=reverse
        )

        return sorted_results[:5]  # Return top 5

    except Exception as e:
        print(f"Optimization error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


# --- v9: G-Code Analysis Endpoint ---


class GCodeAnalysisResult(BaseModel):
    """Response model for G-code analysis."""

    filename: str
    total_lines: int
    print_time_estimate_min: float
    material_used_g: float
    layer_count: int
    feature_breakdown: dict[str, float]  # Percentage of each feature type
    toolpath_segments: list[dict[str, Any]]  # Simplified toolpath data for visualization
    statistics: dict[str, Any]
    warnings: list[str]
    optimization_suggestions: list[str]


@app.post("/analyze_gcode", response_model=GCodeAnalysisResult)
async def analyze_gcode(
    file: UploadFile = File(...),
    material_name: str | None = Form(None),
    filament_diameter_mm: float | None = Form(None),
):
    """
    Analyzes a G-code file and returns detailed insights about the print.
    Based on research by Rivet et al., Pilch & Gibas, and Knirsch et al.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        # Enforce upload size limit (15 MB)
        if len(contents) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="G-code file too large (limit 15 MB)")
        gcode_text = contents.decode('utf-8')

        # Note: We perform manual G-code parsing; no external parser required

        # Split into lines for analysis
        lines = gcode_text.split('\n')
        total_lines = len(lines)

        # Initialize counters and data structures
        layer_count = 0
        current_layer = 0
        saw_layer_comment = False
        current_z = 0
        material_used_mm = 0  # Filament length in mm
        total_time_sec = 0
        # G-code state
        extrusion_mode_absolute = True  # M82 default
        units_scale = 1.0  # G21 default (mm). If G20, set to 25.4
        prev_e = 0.0
        current_feed_mm_min = 1800.0  # reasonable default

        feature_types = {
            'outer_wall': 0,
            'inner_wall': 0,
            'infill': 0,
            'top_surface': 0,
            'bottom_surface': 0,
            'support': 0,
            'unknown': 0,
        }

        toolpath_segments = []
        warnings = []
        optimization_suggestions = []

        current_feature = 'unknown'
        prev_x, prev_y, prev_z = 0, 0, 0
        current_speed = 60  # mm/s default

        # Parse G-code line by line
        for line_num, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines and handle comments
            if not line or line.startswith(';'):
                # Check for feature type comments (common in modern slicers)
                if ';TYPE:' in line.upper():
                    feature_tag = line.split(':')[1].strip().lower()
                    if 'outer' in feature_tag or 'external' in feature_tag:
                        current_feature = 'outer_wall'
                    elif 'inner' in feature_tag or 'perimeter' in feature_tag:
                        current_feature = 'inner_wall'
                    elif 'infill' in feature_tag or 'fill' in feature_tag:
                        current_feature = 'infill'
                    elif 'top' in feature_tag:
                        current_feature = 'top_surface'
                    elif 'bottom' in feature_tag:
                        current_feature = 'bottom_surface'
                    elif 'support' in feature_tag:
                        current_feature = 'support'

                # Check for layer changes
                if ';LAYER:' in line.upper() or 'LAYER_CHANGE' in line.upper():
                    saw_layer_comment = True
                    # Try to parse explicit layer index if available
                    try:
                        if ':' in line:
                            layer_idx_str = line.split(':', 1)[1].strip().split()[0]
                            current_layer = int(layer_idx_str)
                            layer_count = max(layer_count, current_layer)
                        else:
                            layer_count += 1
                            current_layer = layer_count
                    except Exception:
                        layer_count += 1
                        current_layer = layer_count
                continue

            # Modal and unit commands
            upper = line.upper()
            if upper.startswith('M82'):
                extrusion_mode_absolute = True
                continue
            if upper.startswith('M83'):
                extrusion_mode_absolute = False
                continue
            if upper.startswith('G21'):
                units_scale = 1.0
                continue
            if upper.startswith('G20'):
                units_scale = 25.4
                continue
            if upper.startswith('G92') and 'E' in upper:
                # Reset extruder position
                try:
                    parts = line.split()
                    for part in parts:
                        if part.upper().startswith('E'):
                            prev_e = float(part[1:]) * units_scale
                            break
                except Exception:
                    prev_e = 0.0
                continue

            # Parse movement commands
            if line.startswith('G1') or line.startswith('G0'):
                # Extract coordinates and extrusion
                parts = line.split()
                x, y, z, e, f = prev_x, prev_y, prev_z, None, None

                for part in parts:
                    if part.startswith('X'):
                        x = float(part[1:]) * units_scale
                    elif part.startswith('Y'):
                        y = float(part[1:]) * units_scale
                    elif part.startswith('Z'):
                        z = float(part[1:]) * units_scale
                        # If slicer doesn't emit ;LAYER comments, derive layer by Z increase
                        if not saw_layer_comment and z > prev_z:
                            layer_count += 1
                            current_layer = layer_count
                        if z > current_z:
                            current_z = z
                    elif part.startswith('E'):
                        e = float(part[1:]) * units_scale
                    elif part.startswith('F'):
                        f = float(part[1:])
                        current_feed_mm_min = f
                        current_speed = (
                            (current_feed_mm_min / 60.0) if current_feed_mm_min > 0 else 0
                        )

                # Calculate movement
                distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2)

                # Compute extrusion delta
                delta_e = None
                if e is not None:
                    if extrusion_mode_absolute:
                        delta_e = e - prev_e
                        prev_e = e
                    else:
                        delta_e = e

                if delta_e is not None and delta_e > 0:
                    # Extrusion move
                    material_used_mm += delta_e
                    feature_types[current_feature] += distance

                    # Add to toolpath (sample every 10th segment to reduce data size)
                    if line_num % 10 == 0:
                        toolpath_segments.append(
                            {
                                'x': round(x, 2),
                                'y': round(y, 2),
                                'z': round(z, 2),
                                'type': current_feature,
                                'layer': int(current_layer),
                            }
                        )

                    # Check for potential issues
                    if current_speed > 150:
                        if 'High print speed detected' not in warnings:
                            warnings.append(
                                "High print speed detected ("
                                f"{current_speed:.0f} mm/s) - "
                                "may affect quality"
                            )

                if distance > 0:
                    # Use latest feed rate; travel moves also contribute
                    current_speed = (current_feed_mm_min / 60.0) if current_feed_mm_min > 0 else 0
                    move_time = distance / current_speed if current_speed > 0 else 0
                    total_time_sec += move_time

                prev_x, prev_y, prev_z = x, y, z

        # Calculate feature breakdown percentages
        total_distance = sum(feature_types.values())
        feature_breakdown = {}
        if total_distance > 0:
            for feature, distance in feature_types.items():
                feature_breakdown[feature] = round((distance / total_distance) * 100, 2)

        # Estimate material weight using provided material/diameter if present
        filament_diameter = (
            filament_diameter_mm if filament_diameter_mm and filament_diameter_mm > 0 else 1.75
        )
        # Density lookup from materials DB if material_name provided
        density = 1.24
        if material_name and materials_database:
            mat = materials_database.get(material_name)
            if isinstance(mat, dict):
                dens = mat.get('density_g_cm3') or mat.get('common', {}).get('density_g_cm3')
                if isinstance(dens, (int, float)) and dens > 0:  # noqa: UP038
                    density = float(dens)
        filament_area = np.pi * (filament_diameter / 2) ** 2
        volume_cm3 = (material_used_mm * filament_area) / 1000
        material_used_g = volume_cm3 * density

        # Generate optimization suggestions based on analysis
        if feature_breakdown.get('infill', 0) > 60:
            optimization_suggestions.append(
                "High infill detected - consider reducing to 20-40% to save material and time"
            )

        if feature_breakdown.get('support', 0) > 10:
            optimization_suggestions.append(
                "Significant support material detected - consider reorienting part"
            )

        if layer_count > 500:
            optimization_suggestions.append(
                "High layer count - consider increasing layer height for faster printing"
            )

        # Calculate statistics
        statistics = {
            'avg_layer_time_sec': round(total_time_sec / layer_count, 2) if layer_count > 0 else 0,
            'estimated_height_mm': round(current_z, 2),
            'total_toolpath_length_mm': round(total_distance, 2),
            'material_per_layer_g': round(material_used_g / layer_count, 3)
            if layer_count > 0
            else 0,
        }

        return GCodeAnalysisResult(
            filename=file.filename,
            total_lines=total_lines,
            print_time_estimate_min=round(total_time_sec / 60, 2),
            material_used_g=round(material_used_g, 2),
            layer_count=layer_count,
            feature_breakdown=feature_breakdown,
            toolpath_segments=toolpath_segments[:1000],  # Limit to 1000 segments for performance
            statistics=statistics,
            warnings=warnings,
            optimization_suggestions=optimization_suggestions,
        )

    except Exception as e:
        print(f"G-code analysis error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to analyze G-code: {str(e)}")


# ===== v11: STL Mesh Quality Analysis =====


class STLMeshQuality(BaseModel):
    vertex_count: int
    face_count: int
    edge_count: int
    is_watertight: bool
    is_manifold: bool
    volume_cm3: float
    surface_area_cm2: float
    bounding_box_mm: dict[str, float]
    mesh_quality_score: float  # 0-100
    triangle_quality_mean: float
    triangle_quality_min: float
    edge_length_mean_mm: float
    edge_length_std_mm: float
    aspect_ratio_mean: float
    aspect_ratio_max: float
    warnings: list[str]
    recommendations: list[str]


class GCodeReconstructionResult(BaseModel):
    reconstructed_vertices: int
    hausdorff_distance_mm: float  # Max deviation between STL and G-code
    mean_deviation_mm: float
    error_regions: list[dict[str, Any]]  # Regions with high error
    overall_accuracy_percent: float


class STLAnalysisResult(BaseModel):
    filename: str
    mesh_quality: STLMeshQuality
    gcode_comparison: GCodeReconstructionResult | None = None


@app.post("/analyze_stl")
async def analyze_stl(file: UploadFile = File(...), compare_gcode: UploadFile | None = File(None)):
    """
    Analyze STL mesh quality and optionally compare with G-code reconstruction.
    Based on Montalti et al. (2024) - strategies to minimize CAD-to-G-code errors.
    """
    try:
        # Read per-request limits from environment (defaults chosen to be generous but safe)
        max_mb = int(os.getenv("MAX_STL_MB", "20"))
        max_bytes = max_mb * 1024 * 1024
        max_triangles = int(os.getenv("MAX_STL_TRIANGLES", "1000000"))

        # Read STL file (bytes) and enforce size limit prior to any parsing
        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": f"STL exceeds maximum allowed size of {max_mb} MB",
                    "max_mb": max_mb,
                },
            )

        # Load mesh from in-memory buffer
        # Robust mesh load: handle single geometry vs scene, ensure attributes exist
        loaded = trimesh.load(io.BytesIO(content), file_type='stl')
        if isinstance(loaded, trimesh.Scene):  # pick first geometry if scene
            if not loaded.geometry:
                raise HTTPException(status_code=422, detail="Empty STL scene: no geometries found")
            # merge into a single mesh for uniform handling
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            mesh = loaded

        # Enforce triangle complexity limit early to avoid heavy computations
        faces = getattr(mesh, "faces", np.empty((0, 3)))
        face_count = int(len(faces))
        if face_count > max_triangles:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "MESH_TOO_COMPLEX",
                    "message": (
                        f"STL has {face_count} triangles which exceeds the limit of {max_triangles}"
                    ),
                    "max_triangles": max_triangles,
                    "face_count": face_count,
                },
            )

        # Basic mesh metrics
        warnings = []
        recommendations = []

        # Check if watertight
        is_watertight = bool(getattr(mesh, "is_watertight", False))
        if not is_watertight:
            warnings.append("Mesh is not watertight - may cause slicing issues")
            recommendations.append("Repair mesh using Meshmixer or similar tool")

        # Check if manifold
        # Manifold heuristic guarded for missing attributes
        try:
            is_manifold = (not is_watertight) or (len(mesh.split()) == 1)
        except Exception:
            is_manifold = True  # assume manifold if split fails
        if not is_manifold:
            warnings.append("Mesh has non-manifold edges")
            recommendations.append("Fix non-manifold geometry before slicing")

        # Calculate triangle quality (aspect ratio)
        triangles = getattr(mesh, "triangles", np.empty((0, 3, 3)))
        triangle_qualities = []
        aspect_ratios = []

        for tri in triangles:
            # Calculate edge lengths
            edge1 = np.linalg.norm(tri[1] - tri[0])
            edge2 = np.linalg.norm(tri[2] - tri[1])
            edge3 = np.linalg.norm(tri[0] - tri[2])

            # Quality metric: ratio of shortest to longest edge
            longest = max(edge1, edge2, edge3)
            shortest = min(edge1, edge2, edge3)
            quality = shortest / longest if longest > 0 else 0
            triangle_qualities.append(quality)

            # Aspect ratio
            semi_perim = (edge1 + edge2 + edge3) / 2
            if semi_perim > 0:
                area = np.sqrt(
                    semi_perim * (semi_perim - edge1) * (semi_perim - edge2) * (semi_perim - edge3)
                )
                aspect = longest / (2 * np.sqrt(3) * area) if area > 0 else float('inf')
                aspect_ratios.append(aspect)

        triangle_quality_mean = np.mean(triangle_qualities) if triangle_qualities else 0
        triangle_quality_min = np.min(triangle_qualities) if triangle_qualities else 0
        aspect_ratio_mean = (
            np.mean([a for a in aspect_ratios if a != float('inf')]) if aspect_ratios else 0
        )
        aspect_ratio_max = max([a for a in aspect_ratios if a != float('inf')], default=0)

        if triangle_quality_mean < 0.3:
            warnings.append("Poor triangle quality detected")
            recommendations.append("Re-export STL with better tessellation settings")

        # Calculate edge statistics
        edges = getattr(mesh, "edges_unique", np.empty((0, 2)))
        edge_lengths = getattr(mesh, "edges_unique_length", np.array([]))
        edge_length_mean = float(np.mean(edge_lengths))
        edge_length_std = float(np.std(edge_lengths))

        if edge_length_std / edge_length_mean > 5:
            warnings.append("High edge length variance - mesh may have detail loss")
            recommendations.append("Use adaptive meshing or finer resolution")

        # Bounding box
        bounds = getattr(mesh, "bounds", np.array([[0, 0, 0], [0, 0, 0]]))
        bbox = {
            'x_min': float(bounds[0][0]),
            'y_min': float(bounds[0][1]),
            'z_min': float(bounds[0][2]),
            'x_max': float(bounds[1][0]),
            'y_max': float(bounds[1][1]),
            'z_max': float(bounds[1][2]),
            'width': float(bounds[1][0] - bounds[0][0]),
            'depth': float(bounds[1][1] - bounds[0][1]),
            'height': float(bounds[1][2] - bounds[0][2]),
        }

        # Overall quality score (0-100)
        score = 100.0
        if not is_watertight:
            score -= 30
        if not is_manifold:
            score -= 20
        if triangle_quality_mean < 0.5:
            score -= 20 * (1 - triangle_quality_mean / 0.5)
        if aspect_ratio_max > 10:
            score -= 10
        score = max(0, score)

        mesh_quality = STLMeshQuality(
            vertex_count=len(getattr(mesh, "vertices", [])),
            face_count=face_count,
            edge_count=len(edges),
            is_watertight=is_watertight,
            is_manifold=is_manifold,
            volume_cm3=float(getattr(mesh, "volume", 0.0) / 1000),  # mm³ to cm³
            surface_area_cm2=float(getattr(mesh, "area", 0.0) / 100),  # mm² to cm²
            bounding_box_mm=bbox,
            mesh_quality_score=round(score, 1),
            triangle_quality_mean=round(triangle_quality_mean, 3),
            triangle_quality_min=round(triangle_quality_min, 3),
            edge_length_mean_mm=round(edge_length_mean, 3),
            edge_length_std_mm=round(edge_length_std, 3),
            aspect_ratio_mean=round(aspect_ratio_mean, 3),
            aspect_ratio_max=round(aspect_ratio_max, 3),
            warnings=warnings,
            recommendations=recommendations,
        )

        # Optional: Compare with G-code reconstruction
        gcode_comparison = None
        if compare_gcode:
            gcode_content = await compare_gcode.read()
            gcode_text = gcode_content.decode('utf-8')

            # Reconstruct geometry from G-code
            reconstructed_points = []
            lines = gcode_text.split('\n')
            current_x, current_y, current_z = 0.0, 0.0, 0.0

            for line in lines:
                line = line.strip()
                if line.startswith('G1') or line.startswith('G0'):
                    parts = line.split()
                    for part in parts:
                        if part.startswith('X'):
                            current_x = float(part[1:])
                        elif part.startswith('Y'):
                            current_y = float(part[1:])
                        elif part.startswith('Z'):
                            current_z = float(part[1:])

                    if 'E' in line:  # Only extrusion moves
                        reconstructed_points.append([current_x, current_y, current_z])

            if reconstructed_points:
                reconstructed_array = np.array(reconstructed_points)

                # Calculate Hausdorff distance
                # sample guarded for meshes lacking sample()
                try:
                    stl_points = mesh.sample(min(len(reconstructed_array), 5000))
                except Exception:
                    stl_points = np.array(getattr(mesh, "vertices", []))[:5000]
                hausdorff_fwd = directed_hausdorff(stl_points, reconstructed_array)[0]
                hausdorff_back = directed_hausdorff(reconstructed_array, stl_points)[0]
                hausdorff_distance = max(hausdorff_fwd, hausdorff_back)

                # Mean deviation (approximate)
                mean_deviation = (hausdorff_fwd + hausdorff_back) / 2

                # Calculate accuracy percentage
                bbox_diagonal = np.linalg.norm([bbox['width'], bbox['depth'], bbox['height']])
                accuracy_percent = max(0, 100 * (1 - hausdorff_distance / bbox_diagonal))

                gcode_comparison = GCodeReconstructionResult(
                    reconstructed_vertices=len(reconstructed_points),
                    hausdorff_distance_mm=round(hausdorff_distance, 3),
                    mean_deviation_mm=round(mean_deviation, 3),
                    error_regions=[],  # TODO: Implement region-based error analysis
                    overall_accuracy_percent=round(accuracy_percent, 1),
                )

        return STLAnalysisResult(
            filename=file.filename, mesh_quality=mesh_quality, gcode_comparison=gcode_comparison
        )

    except HTTPException as e:
        # Propagate intentional HTTP errors (e.g., size/complexity limits)
        raise e
    except Exception as e:
        print(f"STL analysis error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to analyze STL: {str(e)}")
