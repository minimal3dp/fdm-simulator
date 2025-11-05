import os
import joblib
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Type
import io
# Optional: External G-code parser not required for our analysis; using manual parsing

# --- v4: Optimization Imports ---
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# --- Configuration ---
MODELS_DIR = "models"
MATERIALS_FILE = "materials.json"
FEA_TARGETS_PATH = os.path.join(MODELS_DIR, "fea_target_names.joblib")

# --- FastAPI App Initialization ---
app = FastAPI(title="FDM 3D Print Property Simulator API (v8 Refactored)")

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
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models (Data Validation) ---

# v3: Changed part_volume_cm3 to part_mass_g
class GlobalInputs(BaseModel):
    part_mass_g: float = Field(..., gt=0, description="Mass of the part in grams.")
    filament_cost_kg: float = Field(..., gt=0, description="Cost of the filament spool per kilogram.")
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
    User_Infill_Density: float # This is 0.1 - 1.0
    User_Line_Thickenss_mm: float
    User_Layer_Height_mm: float

class FEAOutput(PredictionBase):
    properties: Dict[str, float]

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
    name: str # e.g., "tensile_strength", "estimated_cost_usd"
    goal: str # "minimize" or "maximize"

class OptimizationConstraint(BaseModel):
    name: str # e.g., "print_speed"
    operator: str # "lt" (less than) or "gt" (greater than)
    value: float

class OptimizationRequest(BaseModel):
    model_name: str
    objectives: List[OptimizationObjective]
    constraints: List[OptimizationConstraint]
    global_inputs: GlobalInputs # part_mass_g, filament_cost_kg, material_name

# --- v8: Refactored Model Configuration ---
def get_model_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "kaggle": {
            "path": os.path.join(MODELS_DIR, "model_kaggle.joblib"),
            "model": None,
            "input_model": KaggleInput,
            "output_model": KaggleOutput,
            "output_names": ["tensile_strength", "roughness", "elongation"],
            "cost_time_params": {
                'infill': 'infill_density', 'layer_height': 'layer_height', 'speed': 'print_speed'
            },
            "optimizer_config": {
                "numeric": [
                    ("layer_height", 0.02, 0.3), ("wall_thickness", 1, 10), ("infill_density", 10, 100),
                    ("nozzle_temperature", 200, 250), ("bed_temperature", 50, 100), ("print_speed", 40, 120), ("fan_speed", 0, 100)
                ],
                "categorical": [("infill_pattern", ["grid", "honeycomb"]), ("material", ["abs", "pla"])]
            }
        },
        "c3": {
            "path": os.path.join(MODELS_DIR, "model_c3.joblib"),
            "model": None,
            "input_model": C3Input,
            "output_model": C3Output,
            "output_names": ["Tensile_Strength_MPa", "Elongation_at_Break_percent"],
            "cost_time_params": {'infill': 'Fill', 'layer_height': 'Height', 'speed': 'Speed'},
            "optimizer_config": None # Not implemented for optimizer
        },
        "fea": {
            "path": os.path.join(MODELS_DIR, "model_fea.joblib"),
            "model": None,
            "input_model": FEAInput,
            "output_model": FEAOutput,
            "output_names": [], # Loaded dynamically from fea_target_names
            "cost_time_params": {
                'infill': 'User_Infill_Density_100', 'layer_height': 'User_Layer_Height_mm', 'speed': 60.0
            },
            "optimizer_config": None # FEA is too complex for this optimizer
        },
        "fatigue": {
            "path": os.path.join(MODELS_DIR, "model_fatigue.joblib"),
            "model": None,
            "input_model": FatigueInput,
            "output_model": FatigueOutput,
            "output_names": ["Fatigue_Lifetime"],
            "cost_time_params": {'infill': 80.0, 'layer_height': 0.2, 'speed': 'Print_Speed'},
            "optimizer_config": None # Not implemented for optimizer
        },
        "accuracy": {
            "path": os.path.join(MODELS_DIR, "model_accuracy.joblib"),
            "model": None,
            "input_model": AccuracyInput,
            "output_model": AccuracyOutput,
            "output_names": ["Var_Length_percent", "Var_Width_percent", "Var_Thickness_percent"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent', 'layer_height': 'Layer_Thickness_mm', 'speed': 60.0
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Thickness_mm", 0.1, 0.4), ("Build_Orientation_deg", 0, 90),
                    ("Infill_Density_percent", 80, 100), ("Number_of_Contours", 1, 3)
                ],
                "categorical": []
            }
        },
        "warpage": {
            "path": os.path.join(MODELS_DIR, "model_warpage.joblib"),
            "model": None,
            "input_model": WarpageInput,
            "output_model": WarpageOutput,
            "output_names": ["Warpage_mm"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent', 'layer_height': 'Other_Layer_Height_mm', 'speed': 60.0
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Temperature_C", 190, 210), ("Infill_Density_percent", 10, 30),
                    ("First_Layer_Height_mm", 0.2, 0.4), ("Other_Layer_Height_mm", 0.3, 0.5)
                ],
                "categorical": []
            }
        },
        "hardness": {
            "path": os.path.join(MODELS_DIR, "model_hardness.joblib"),
            "model": None,
            "input_model": HardnessInput,
            "output_model": HardnessOutput,
            "output_names": ["Hardness_Shore_D"],
            "cost_time_params": {
                'infill': 'Fill_Density_percent', 'layer_height': 'Layer_Thickness_mm', 'speed': 60.0
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Thickness_mm", 0.2, 0.5), ("Shell_Thickness_mm", 1.2, 1.6),
                    ("Fill_Density_percent", 20, 80),
                ],
                "categorical": [("Fill_Pattern", ["Rectilinear", "Honey_Comb"])]
            }
        },
        "multimaterial": {
            "path": os.path.join(MODELS_DIR, "model_multimaterial.joblib"),
            "model": None,
            "input_model": MultiMaterialInput,
            "output_model": MultiMaterialOutput,
            "output_names": ["Tensile_Strength_MPa"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent', 'layer_height': 'Layer_Height_mm', 'speed': 60.0
            },
            "optimizer_config": {
                "numeric": [
                    ("Layer_Height_mm", 0.2, 0.4), ("Extrusion_Temp_C", 230, 250),
                    ("Infill_Density_percent", 60, 100)
                ],
                "categorical": [("Material_A", ["ABS"]), ("Material_B", ["PETG"])]
            }
        },
        "composite": {
            "path": os.path.join(MODELS_DIR, "model_composite.joblib"),
            "model": None,
            "input_model": CompositeInput,
            "output_model": CompositeOutput,
            "output_names": ["Tensile_Strength_MPa", "Elastic_Modulus_GPa"],
            "cost_time_params": {
                'infill': 'Infill_Density_percent', 'layer_height': 'Layer_Thickness_mm', 'speed': 60.0
            },
            "optimizer_config": {
                "numeric": [
                    ("Reinforcement_percent", 0, 40),
                    ("Infill_Density_percent", 20, 100),
                    ("Layer_Thickness_mm", 0.1, 0.4)
                ],
                "categorical": [("Infill_Pattern", ["Grid", "Tri-Hexagon", "Gyroid"])]
            }
        }
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
    
    return 1.25 # Average density

def calculate_cost_time(
    inputs: GlobalInputs,
    model_params: Dict[str, float]
) -> Dict[str, float]:
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
        material_density = 1.25 # Prevent division by zero
    
    part_volume_cm3 = part_mass_g / material_density

    # Extract model-specific parameters
    infill_percent = model_params.get('infill', 80.0) # Infill as 0-100
    layer_height_mm = model_params.get('layer_height', 0.2)
    print_speed_mm_s = model_params.get('speed', 60.0)
    
    # Ensure infill is a percentage (0-100)
    if infill_percent <= 1.0 and infill_percent > 0:
        infill_percent *= 100 # Convert 0.5 to 50
    
    # This is a very rough estimate.
    extrusion_width_mm = 0.4 
    volume_to_extrude_mm3 = part_volume_cm3 * (infill_percent / 100.0) * 1000.0 # cm3 to mm3
    
    # Prevent division by zero if speed or layer height is 0
    if print_speed_mm_s <= 0 or layer_height_mm <= 0:
        estimated_print_time_min = 0.0
    else:
        flow_rate_mm3_s = layer_height_mm * extrusion_width_mm * print_speed_mm_s
        estimated_print_time_sec = volume_to_extrude_mm3 / flow_rate_mm3_s
        estimated_print_time_min = estimated_print_time_sec / 60.0

    return {
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_print_time_min": estimated_print_time_min
    }

# --- Server Startup ---
@app.on_event("startup")
async def startup_event():
    """Load all models and data on server startup."""
    global materials_database, fea_target_names, MODEL_REGISTRY
    
    # Load Materials JSON
    try:
        with open(MATERIALS_FILE, 'r') as f:
            materials_database = json.load(f)
        print(f"Successfully loaded '{MATERIALS_FILE}'")
    except Exception as e:
        print(f"Warning: '{MATERIALS_FILE}' not found or corrupt. Using empty database. Error: {e}")

    # Load FEA Target Names (special case)
    try:
        fea_target_names = joblib.load(FEA_TARGETS_PATH)
        print(f"Successfully loaded FEA target names ({len(fea_target_names)} properties).")
    except Exception as e:
        print(f"Warning: FEA target names file not found at {FEA_TARGETS_PATH}. FEA endpoint may fail.")

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
            print(f"Warning: Model file not found at {config['path']}. Endpoint for '{model_name}' will be disabled. Error: {e}")


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "FDM Property Simulator API (v8) is running."}

@app.get("/materials", response_model=Dict[str, dict])
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
        raise HTTPException(status_code=503, detail=f"{model_name.capitalize()} model is not loaded.")
        
    try:
        # 1. Get cost/time info
        model_params = {}
        input_dict = inputs.dict()
        for key, val_key in config["cost_time_params"].items():
            if isinstance(val_key, str):
                model_params[key] = input_dict.get(val_key)
            else:
                model_params[key] = val_key # Use default value
        
        # Special case for FEA infill (0.1-1.0) -> (10-100)
        if model_name == "fea":
            model_params['infill'] = input_dict.get('User_Infill_Density', 0.5) * 100.0

        cost_time_results = calculate_cost_time(inputs, model_params)

        # 2. Get model prediction
        features_dict = inputs.dict(exclude={'part_mass_g', 'filament_cost_kg', 'material_name'})
        
        # Special case for FEA feature name mapping
        if model_name == "fea":
            features_dict['Material_Youngs_Modulus_GPa'] = features_dict.pop('Material_Youngs_Modulus_GPa')
            features_dict['Material_Tensile_Yield_Strenght_MPa'] = features_dict.pop('Material_Tensile_Yield_Strenght_MPa')
            features_dict['Material_Poissons_Ratio'] = features_dict.pop('Material_Poissons_Ratio')
        
        features = pd.DataFrame([features_dict])
        prediction = config["model"].predict(features)[0]
        
        # 3. Combine and return
        OutputModel = config["output_model"]
        output_data = {}
        
        if model_name == "fea":
            output_data["properties"] = dict(zip(fea_target_names, prediction))
        elif isinstance(prediction, (np.ndarray, list)): # Multi-output (Kaggle, Accuracy)
            for i, name in enumerate(config["output_names"]):
                output_data[name] = prediction[i]
        else: # Single-output (C3, Fatigue, Warpage, Hardness, MultiMaterial)
             output_data[config["output_names"][0]] = prediction
             
        return OutputModel(
            **output_data,
            **cost_time_results
        )
        
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


# --- v4: Optimization Endpoint ---

class OptimizationProblem(Problem):
    """Defines the optimization problem for pymoo."""
    def __init__(self, model_name: str, request: OptimizationRequest, param_config: Dict[str, Any]):
        
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
            "wall_thickness", "infill_density", "nozzle_temperature", 
            "bed_temperature", "print_speed", "fan_speed", 
            "Build_Orientation_deg", "Infill_Density_percent", "Number_of_Contours",
            "Layer_Temperature_C", "Infill_Density_percent", "Fill_Density_percent",
            "Extrusion_Temp_C", "Infill_Density_percent",
            # v9: Composite params (PascalCase)
            "Print_Speed", "Nozzle_Temperature", "Bed_Temperature"
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
                if self.var_types[i] == int:
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

        if isinstance(predictions[0], (np.ndarray, list)): # Multi-output (Kaggle, Accuracy)
             for i, name in enumerate(output_names):
                pred_map[name] = predictions[:, i]
        else: # Single-output
            pred_map[output_names[0]] = predictions

        # 3. Calculate cost/time for all solutions
        cost_time_list = []
        for i in range(n_sols):
            param_map = {}
            for key, val_key in param_map_key.items():
                if isinstance(val_key, str):
                    param_map[key] = features_df.loc[i, val_key]
                else:
                    param_map[key] = val_key # Use default value
            
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
                f_matrix[:, i] = 0.0 # Objective not found
            else:
                values = pred_map[obj.name]
                if obj.goal == "maximize":
                    f_matrix[:, i] = -values # pymoo minimizes by default
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
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not loaded or not found.")
        
    param_config = model_config.get("optimizer_config")
    
    if param_config is None:
        raise HTTPException(status_code=404, detail=f"Optimization not supported for model '{request.model_name}'.")
        
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
            eliminate_duplicates=True
        )

        # 3. Define termination
        termination = get_termination("n_gen", 40)

        # 4. Run the optimization
        print(f"Starting optimization for {request.model_name}...")
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=False)
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
                if problem.var_types[j] == int: # Use type info from problem
                    val = int(round(val))
                inputs[name] = round(val, 2)
            for j, (name, _) in enumerate(param_config["categorical"]):
                cat_index = int(round(sol_vector[n_numeric + j]))
                inputs[name] = param_config["categorical"][j][1][cat_index] # Get string value
            
            # Get output objectives
            for j, obj in enumerate(request.objectives):
                value = objective_values[i, j]
                if obj.goal == "maximize":
                    value = -value # Flip back to positive
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
        
        sorted_results = sorted(results, key=lambda r: r['outputs'][first_obj_name], reverse=reverse)
        
        return sorted_results[:5] # Return top 5

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
    feature_breakdown: Dict[str, float]  # Percentage of each feature type
    toolpath_segments: List[Dict[str, Any]]  # Simplified toolpath data for visualization
    statistics: Dict[str, Any]
    warnings: List[str]
    optimization_suggestions: List[str]

@app.post("/analyze_gcode", response_model=GCodeAnalysisResult)
async def analyze_gcode(file: UploadFile = File(...)):
    """
    Analyzes a G-code file and returns detailed insights about the print.
    Based on research by Rivet et al., Pilch & Gibas, and Knirsch et al.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        gcode_text = contents.decode('utf-8')
        
    # Note: We perform manual G-code parsing; no external parser required
        
        # Split into lines for analysis
        lines = gcode_text.split('\n')
        total_lines = len(lines)
        
        # Initialize counters and data structures
        layer_count = 0
        current_z = 0
        material_used_mm = 0  # Filament length in mm
        total_time_sec = 0
        
        feature_types = {
            'outer_wall': 0,
            'inner_wall': 0,
            'infill': 0,
            'top_surface': 0,
            'bottom_surface': 0,
            'support': 0,
            'unknown': 0
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
            
            # Skip empty lines and comments
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
                    layer_count += 1
                continue
            
            # Parse G-code commands
            if line.startswith('G1') or line.startswith('G0'):
                # Extract coordinates and extrusion
                parts = line.split()
                x, y, z, e, f = prev_x, prev_y, prev_z, None, None
                
                for part in parts:
                    if part.startswith('X'):
                        x = float(part[1:])
                    elif part.startswith('Y'):
                        y = float(part[1:])
                    elif part.startswith('Z'):
                        z = float(part[1:])
                        if z > current_z:
                            current_z = z
                    elif part.startswith('E'):
                        e = float(part[1:])
                    elif part.startswith('F'):
                        f = float(part[1:])
                        current_speed = f / 60  # Convert mm/min to mm/s
                
                # Calculate movement
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)
                
                if e is not None and e > 0:
                    # Extrusion move
                    material_used_mm += e
                    feature_types[current_feature] += distance
                    
                    # Add to toolpath (sample every 10th segment to reduce data size)
                    if line_num % 10 == 0:
                        toolpath_segments.append({
                            'x': round(x, 2),
                            'y': round(y, 2),
                            'z': round(z, 2),
                            'type': current_feature,
                            'layer': layer_count
                        })
                    
                    # Check for potential issues
                    if current_speed > 150:
                        if 'High print speed detected' not in warnings:
                            warnings.append(f"High print speed detected ({current_speed:.0f} mm/s) - may affect quality")
                
                if distance > 0:
                    move_time = distance / current_speed if current_speed > 0 else 0
                    total_time_sec += move_time
                
                prev_x, prev_y, prev_z = x, y, z
        
        # Calculate feature breakdown percentages
        total_distance = sum(feature_types.values())
        feature_breakdown = {}
        if total_distance > 0:
            for feature, distance in feature_types.items():
                feature_breakdown[feature] = round((distance / total_distance) * 100, 2)
        
        # Estimate material weight (assuming 1.75mm PLA at 1.24 g/cm³)
        filament_diameter = 1.75  # mm
        filament_area = np.pi * (filament_diameter / 2) ** 2
        volume_cm3 = (material_used_mm * filament_area) / 1000
        material_used_g = volume_cm3 * 1.24  # PLA density
        
        # Generate optimization suggestions based on analysis
        if feature_breakdown.get('infill', 0) > 60:
            optimization_suggestions.append("High infill detected - consider reducing to 20-40% to save material and time")
        
        if feature_breakdown.get('support', 0) > 10:
            optimization_suggestions.append("Significant support material detected - consider reorienting part")
        
        if layer_count > 500:
            optimization_suggestions.append("High layer count - consider increasing layer height for faster printing")
        
        # Calculate statistics
        statistics = {
            'avg_layer_time_sec': round(total_time_sec / layer_count, 2) if layer_count > 0 else 0,
            'estimated_height_mm': round(current_z, 2),
            'total_toolpath_length_mm': round(total_distance, 2),
            'material_per_layer_g': round(material_used_g / layer_count, 3) if layer_count > 0 else 0
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
            optimization_suggestions=optimization_suggestions
        )
        
    except Exception as e:
        print(f"G-code analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to analyze G-code: {str(e)}")
