import os
import joblib
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

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

# Model paths
MODEL_KAGGLE_PATH = os.path.join(MODELS_DIR, "model_kaggle.joblib")
MODEL_C3_PATH = os.path.join(MODELS_DIR, "model_c3.joblib")
MODEL_FEA_PATH = os.path.join(MODELS_DIR, "model_fea.joblib")
MODEL_FATIGUE_PATH = os.path.join(MODELS_DIR, "model_fatigue.joblib")
FEA_TARGETS_PATH = os.path.join(MODELS_DIR, "fea_target_names.joblib")

# --- FastAPI App Initialization ---
app = FastAPI(title="FDM 3D Print Property Simulator API (v4)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Model Storage ---
model_kaggle = None
model_c3 = None
model_fea = None
model_fatigue = None
fea_target_names = []
materials_database = {}

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
    
    # --- Cost Calculation (CHANGED) ---
    # This is now much simpler
    part_mass_kg = part_mass_g / 1000.0
    estimated_cost_usd = part_mass_kg * filament_cost_kg

    # --- Time Calculation (CHANGED) ---
    # Need to get density to convert mass back to volume for time estimation
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
    # (Volume to extrude) / (Flow rate)
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
    global model_kaggle, model_c3, model_fea, model_fatigue, fea_target_names, materials_database
    
    # Load Materials JSON
    try:
        with open(MATERIALS_FILE, 'r') as f:
            materials_database = json.load(f)
        print(f"Successfully loaded '{MATERIALS_FILE}'")
    except Exception as e:
        print(f"Warning: '{MATERIALS_FILE}' not found or corrupt. Using empty database. Error: {e}")

    # Load Model 1: Kaggle
    try:
        model_kaggle = joblib.load(MODEL_KAGGLE_PATH)
        print(f"Successfully loaded model 'kaggle' from {MODEL_KAGGLE_PATH}")
    except Exception as e:
        print(f"Warning: Model file not found at {MODEL_KAGGLE_PATH}. Endpoint for 'kaggle' will be disabled. Error: {e}")

    # Load Model 2: C3
    try:
        model_c3 = joblib.load(MODEL_C3_PATH)
        print(f"Successfully loaded model 'c3' from {MODEL_C3_PATH}")
    except Exception as e:
        print(f"Warning: Model file not found at {MODEL_C3_PATH}. Endpoint for 'c3' will be disabled. Error: {e}")

    # Load Model 3: FEA
    try:
        model_fea = joblib.load(MODEL_FEA_PATH)
        print(f"Successfully loaded model 'fea' from {MODEL_FEA_PATH}")
        # Load target names
        fea_target_names = joblib.load(FEA_TARGETS_PATH)
        print(f"Successfully loaded FEA target names ({len(fea_target_names)} properties).")
    except Exception as e:
        print(f"Warning: Model file or target names not found for 'fea'. Endpoint will be disabled. Error: {e}")
        model_fea = None # Ensure it's disabled

    # Load Model 4: Fatigue
    try:
        model_fatigue = joblib.load(MODEL_FATIGUE_PATH)
        print(f"Successfully loaded model 'fatigue' from {MODEL_FATIGUE_PATH}")
    except Exception as e:
        print(f"Warning: Model file not found at {MODEL_FATIGUE_PATH}. Endpoint for 'fatigue' will be disabled. Error: {e}")


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "FDM Property Simulator API (v4) is running."}

@app.get("/materials", response_model=Dict[str, dict])
async def get_materials():
    """Serves the loaded materials.json file."""
    if not materials_database:
        raise HTTPException(status_code=404, detail="Materials database is not loaded.")
    return materials_database

# --- Endpoint 1: Kaggle (Prediction) ---
@app.post("/predict/kaggle", response_model=KaggleOutput)
async def predict_kaggle(inputs: KaggleInput):
    if model_kaggle is None:
        raise HTTPException(status_code=503, detail="Kaggle model is not loaded.")
    
    try:
        # 1. Get cost/time info
        param_map = {
            'infill': inputs.infill_density,
            'layer_height': inputs.layer_height,
            'speed': inputs.print_speed
        }
        cost_time_results = calculate_cost_time(inputs, param_map)

        # 2. Get model prediction
        features = pd.DataFrame([inputs.dict(exclude={'part_mass_g', 'filament_cost_kg', 'material_name'})])
        prediction = model_kaggle.predict(features)[0]
        
        # 3. Combine and return
        return KaggleOutput(
            tensile_strength=prediction[0],
            roughness=prediction[1],
            elongation=prediction[2],
            **cost_time_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Endpoint 2: C3 (Prediction) ---
@app.post("/predict/c3", response_model=C3Output)
async def predict_c3(inputs: C3Input):
    if model_c3 is None:
        raise HTTPException(status_code=503, detail="C3 model is not loaded.")
        
    try:
        # 1. Get cost/time info
        param_map = {
            'infill': inputs.Fill,
            'layer_height': inputs.Height,
            'speed': inputs.Speed
        }
        cost_time_results = calculate_cost_time(inputs, param_map)

        # 2. Get model prediction
        features = pd.DataFrame([inputs.dict(exclude={'part_mass_g', 'filament_cost_kg', 'material_name'})])
        prediction = model_c3.predict(features)[0]
        
        # 3. Combine and return
        return C3Output(
            Tensile_Strength_MPa=prediction[0],
            Elongation_at_Break_percent=prediction[1],
            **cost_time_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Endpoint 3: FEA (Prediction) ---
@app.post("/predict/fea", response_model=FEAOutput)
async def predict_fea(inputs: FEAInput):
    if model_fea is None or not fea_target_names:
        raise HTTPException(status_code=503, detail="FEA model or target names are not loaded.")
        
    try:
        # 1. Get cost/time info
        param_map = {
            'infill': inputs.User_Infill_Density * 100.0, # Convert 0.1-1.0 to 10-100
            'layer_height': inputs.User_Layer_Height_mm,
            'speed': 60.0 # FEA model doesn't use print speed, so use a default for time
        }
        cost_time_results = calculate_cost_time(inputs, param_map)

        # 2. Get model prediction
        features_dict = inputs.dict(exclude={'part_mass_g', 'filament_cost_kg', 'material_name'})
        
        # Ensure correct feature names match training
        features_dict['Material_Youngs_Modulus_GPa'] = features_dict.pop('Material_Youngs_Modulus_GPa')
        features_dict['Material_Tensile_Yield_Strenght_MPa'] = features_dict.pop('Material_Tensile_Yield_Strenght_MPa')
        features_dict['Material_Poissons_Ratio'] = features_dict.pop('Material_Poissons_Ratio')
        
        features = pd.DataFrame([features_dict])
        prediction = model_fea.predict(features)[0]
        
        # 3. Combine and return
        properties_dict = dict(zip(fea_target_names, prediction))
        
        return FEAOutput(
            properties=properties_dict,
            **cost_time_results
        )
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: columns are missing: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Endpoint 4: Fatigue (Prediction) ---
@app.post("/predict/fatigue", response_model=FatigueOutput)
async def predict_fatigue(inputs: FatigueInput):
    if model_fatigue is None:
        raise HTTPException(status_code=503, detail="Fatigue model is not loaded.")
        
    try:
        # 1. Get cost/time info
        param_map = {
            # Fatigue model doesn't use infill or layer height, use defaults
            'infill': 80.0, 
            'layer_height': 0.2,
            'speed': inputs.Print_Speed
        }
        cost_time_results = calculate_cost_time(inputs, param_map)

        # 2. Get model prediction
        features = pd.DataFrame([inputs.dict(exclude={'part_mass_g', 'filament_cost_kg', 'material_name'})])
        prediction = model_fatigue.predict(features)[0]
        
        # 3. Combine and return
        return FatigueOutput(
            Fatigue_Lifetime=prediction,
            **cost_time_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- v4: Optimization Endpoint ---

# Define problem parameters for each model
# (param_name, type, [min, max] or [options])
KAGGLE_OPT_PARAMS = {
    "numeric": [
        ("layer_height", 0.02, 0.3),
        ("wall_thickness", 1, 10),
        ("infill_density", 10, 100),
        ("nozzle_temperature", 200, 250),
        ("bed_temperature", 50, 100),
        ("print_speed", 40, 120),
        ("fan_speed", 0, 100)
    ],
    "categorical": [
        ("infill_pattern", ["grid", "honeycomb"]),
        ("material", ["abs", "pla"])
    ]
}
# Add other models as needed...

class OptimizationProblem(Problem):
    """Defines the optimization problem for pymoo."""
    def __init__(self, model, model_name, request: OptimizationRequest, param_config):
        self.model = model
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
        self.n_obj = len(request.objectives)
        self.n_constr = len(request.constraints)
        
        super().__init__(n_var=n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=xl, xu=xu, vtype=float) # Treat all as float for simplicity, then round categorical

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a 2D array (n_solutions, n_vars)
        
        n_sols = x.shape[0]
        n_numeric = len(self.numeric_params)
        
        # 1. Decode the input array 'x' into a list of feature dictionaries
        solutions_df_list = []
        for sol_vector in x:
            sol_dict = {}
            # Decode numeric
            for i, name in enumerate(self.numeric_params):
                sol_dict[name] = sol_vector[i]
            # Decode categorical
            for i, name in enumerate(self.categorical_params):
                cat_index = int(round(sol_vector[n_numeric + i]))
                sol_dict[name] = self.cat_mappings[f"{name}_reverse"][cat_index]
            solutions_df_list.append(sol_dict)
            
        features_df = pd.DataFrame(solutions_df_list)
        
        # 2. Get model predictions for all solutions in a batch
        # This assumes the model's `predict` method matches the outputs
        if self.model_name == "kaggle":
            predictions = self.model.predict(features_df) # (n_sols, 3)
            # Map predictions to names
            pred_map = {
                "tensile_strength": predictions[:, 0],
                "roughness": predictions[:, 1],
                "elongation": predictions[:, 2]
            }
            # Also calculate cost/time for all solutions
            cost_time_list = []
            for i in range(n_sols):
                param_map = {
                    'infill': features_df.loc[i, 'infill_density'],
                    'layer_height': features_df.loc[i, 'layer_height'],
                    'speed': features_df.loc[i, 'print_speed']
                }
                cost_time = calculate_cost_time(self.request.global_inputs, param_map)
                cost_time_list.append(cost_time)
            
            cost_time_df = pd.DataFrame(cost_time_list)
            pred_map["estimated_cost_usd"] = cost_time_df["estimated_cost_usd"].values
            pred_map["estimated_print_time_min"] = cost_time_df["estimated_print_time_min"].values
        else:
            # Placeholder for other models
            # You would need to implement this logic for C3, FEA, etc.
            # For now, just return zeros if not kaggle
            pred_map = {}
            for obj in self.request.objectives:
                pred_map[obj.name] = np.zeros(n_sols)

        # 3. Calculate objectives (F)
        f_matrix = np.zeros((n_sols, self.n_obj))
        for i, obj in enumerate(self.request.objectives):
            if obj.name not in pred_map:
                # Handle error if objective name is invalid
                f_matrix[:, i] = 0.0 # Or raise error
            else:
                values = pred_map[obj.name]
                if obj.goal == "maximize":
                    f_matrix[:, i] = -values # pymoo minimizes by default, so flip sign
                else:
                    f_matrix[:, i] = values
        
        out["F"] = f_matrix

        # 4. Calculate constraints (G)
        if self.n_constr > 0:
            g_matrix = np.zeros((n_sols, self.n_constr))
            for i, constr in enumerate(self.request.constraints):
                current_values = features_df[constr.name].values
                if constr.operator == "lt":
                    # Constraint is g(x) <= 0, so we want current - value <= 0
                    g_matrix[:, i] = current_values - constr.value
                elif constr.operator == "gt":
                    # Constraint is g(x) >= 0, so we want value - current <= 0
                    g_matrix[:, i] = constr.value - current_values
            out["G"] = g_matrix

@app.post("/optimize")
async def optimize(request: OptimizationRequest):
    
    model = None
    param_config = None
    
    # Load the correct model and param config
    if request.model_name == "kaggle":
        model = model_kaggle
        param_config = KAGGLE_OPT_PARAMS
    # Add 'elif request.model_name == "c3":' etc. here
        
    if model is None:
        raise HTTPException(status_code=404, detail=f"Optimization not supported or model '{request.model_name}' not loaded.")
        
    try:
        # 1. Initialize the problem
        problem = OptimizationProblem(model, request.model_name, request, param_config)

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
        print("Starting optimization...")
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=False)
        print("Optimization complete.")

        # 5. Format and return the results
        results = []
        
        # res.X = inputs (n_solutions, n_vars)
        # res.F = outputs (n_solutions, n_objectives)
        
        n_numeric = len(param_config["numeric"])
        
        # Get the objective values from the result
        objective_values = res.F
        
        for i, sol_vector in enumerate(res.X):
            solution = {}
            inputs = {}
            outputs = {}
            
            # Decode input parameters
            for j, (name, _, _) in enumerate(param_config["numeric"]):
                # Round numeric values to reasonable precision
                inputs[name] = round(sol_vector[j], 2)
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
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

