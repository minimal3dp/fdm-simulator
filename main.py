import os
import joblib
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

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
app = FastAPI(title="FDM 3D Print Property Simulator API")

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

# v3: New global inputs
class GlobalInputs(BaseModel):
    part_volume_cm3: float = Field(..., gt=0, description="Volume of the part in cubic centimeters.")
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

# --- v3: Cost & Time Calculation Logic ---

def get_material_density(material_name: str) -> float:
    """Gets density from the loaded materials database, with a fallback."""
    material_data = materials_database.get(material_name)
    if material_data and 'density_g_cm3' in material_data:
        return material_data['density_g_cm3']
    # Fallback density if not found
    return 1.25 # Average density for PLA/ABS

def calculate_cost_time(
    inputs: GlobalInputs,
    material_density: float,
    model_params: Dict[str, float]
) -> Dict[str, float]:
    """Calculates estimated cost and print time."""
    
    # Extract global inputs
    part_volume_cm3 = inputs.part_volume_cm3
    filament_cost_kg = inputs.filament_cost_kg
    
    # Extract model-specific parameters
    # Use defaults if a model doesn't provide the param (e.g., FEA has no print speed)
    infill_percent = model_params.get('infill', 80.0) # Infill as 0-100
    layer_height_mm = model_params.get('layer_height', 0.2)
    print_speed_mm_s = model_params.get('speed', 60.0)
    
    # Ensure infill is a percentage (0-100)
    if infill_percent <= 1.0 and infill_percent > 0:
        infill_percent *= 100 # Convert 0.5 to 50
    
    # --- Cost Calculation ---
    # 1. Calculate part mass in grams
    # (Volume * density) * (infill / 100) -> assumes infill is the main volume
    part_mass_g = (part_volume_cm3 * material_density) * (infill_percent / 100.0)
    # 2. Convert mass to kg
    part_mass_kg = part_mass_g / 1000.0
    # 3. Calculate cost
    estimated_cost_usd = part_mass_kg * filament_cost_kg

    # --- Time Calculation (Simple Model) ---
    # This is a very rough estimate.
    # (Volume to extrude) / (Flow rate)
    # Volume = part_volume * (infill / 100)
    # Flow rate (mm^3/s) = layer_height * extrusion_width * print_speed
    # We'll assume extrusion_width = nozzle_diameter (~0.4mm)
    extrusion_width_mm = 0.4 
    volume_to_extrude_mm3 = part_volume_cm3 * (infill_percent / 100.0) * 1000.0 # cm3 to mm3
    
    # Prevent division by zero if speed or layer height is 0
    if print_speed_mm_s == 0 or layer_height_mm == 0:
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
    return {"message": "FDM Property Simulator API is running."}

@app.get("/materials", response_model=Dict[str, dict])
async def get_materials():
    """Serves the loaded materials.json file."""
    if not materials_database:
        raise HTTPException(status_code=404, detail="Materials database is not loaded.")
    return materials_database

# --- Endpoint 1: Kaggle ---
@app.post("/predict/kaggle", response_model=KaggleOutput)
async def predict_kaggle(inputs: KaggleInput):
    if model_kaggle is None:
        raise HTTPException(status_code=503, detail="Kaggle model is not loaded.")
    
    try:
        # 1. Get cost/time info
        material_density = get_material_density(inputs.material_name)
        param_map = {
            'infill': inputs.infill_density,
            'layer_height': inputs.layer_height,
            'speed': inputs.print_speed
        }
        cost_time_results = calculate_cost_time(inputs, material_density, param_map)

        # 2. Get model prediction
        features = pd.DataFrame([inputs.dict(exclude={'part_volume_cm3', 'filament_cost_kg', 'material_name'})])
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

# --- Endpoint 2: C3 ---
@app.post("/predict/c3", response_model=C3Output)
async def predict_c3(inputs: C3Input):
    if model_c3 is None:
        raise HTTPException(status_code=503, detail="C3 model is not loaded.")
        
    try:
        # 1. Get cost/time info
        material_density = get_material_density(inputs.material_name)
        param_map = {
            'infill': inputs.Fill,
            'layer_height': inputs.Height,
            'speed': inputs.Speed
        }
        cost_time_results = calculate_cost_time(inputs, material_density, param_map)

        # 2. Get model prediction
        features = pd.DataFrame([inputs.dict(exclude={'part_volume_cm3', 'filament_cost_kg', 'material_name'})])
        prediction = model_c3.predict(features)[0]
        
        # 3. Combine and return
        return C3Output(
            Tensile_Strength_MPa=prediction[0],
            Elongation_at_Break_percent=prediction[1],
            **cost_time_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Endpoint 3: FEA ---
@app.post("/predict/fea", response_model=FEAOutput)
async def predict_fea(inputs: FEAInput):
    if model_fea is None or not fea_target_names:
        raise HTTPException(status_code=503, detail="FEA model or target names are not loaded.")
        
    try:
        # 1. Get cost/time info
        material_density = get_material_density(inputs.material_name)
        param_map = {
            'infill': inputs.User_Infill_Density * 100.0, # Convert 0.1-1.0 to 10-100
            'layer_height': inputs.User_Layer_Height_mm,
            'speed': 60.0 # FEA model doesn't use print speed, so use a default for time
        }
        cost_time_results = calculate_cost_time(inputs, material_density, param_map)

        # 2. Get model prediction
        features_dict = inputs.dict(exclude={'part_volume_cm3', 'filament_cost_kg', 'material_name'})
        
        # Ensure correct feature names match training
        features_dict['Material_Youngs_Modulus_GPa'] = features_dict.pop('Material_Young_s_Modulus_GPa')
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

# --- Endpoint 4: Fatigue ---
@app.post("/predict/fatigue", response_model=FatigueOutput)
async def predict_fatigue(inputs: FatigueInput):
    if model_fatigue is None:
        raise HTTPException(status_code=503, detail="Fatigue model is not loaded.")
        
    try:
        # 1. Get cost/time info
        material_density = get_material_density(inputs.material_name)
        param_map = {
            # Fatigue model doesn't use infill or layer height, use defaults
            'infill': 80.0, 
            'layer_height': 0.2,
            'speed': inputs.Print_Speed
        }
        cost_time_results = calculate_cost_time(inputs, material_density, param_map)

        # 2. Get model prediction
        features = pd.DataFrame([inputs.dict(exclude={'part_volume_cm3', 'filament_cost_kg', 'material_name'})])
        prediction = model_fatigue.predict(features)[0]
        
        # 3. Combine and return
        return FatigueOutput(
            Fatigue_Lifetime=prediction,
            **cost_time_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

