import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List

# --- Configuration ---
MODELS_DIR = "models"
MODEL_KAGGLE_PATH = os.path.join(MODELS_DIR, "model_kaggle.joblib")
MODEL_C3_PATH = os.path.join(MODELS_DIR, "model_c3.joblib")
MODEL_FEA_PATH = os.path.join(MODELS_DIR, "model_fea.joblib")
MODEL_FATIGUE_PATH = os.path.join(MODELS_DIR, "model_fatigue.joblib")
MATERIALS_JSON_PATH = "materials.json"

# --- NEW: Path for FEA target names ---
FEA_TARGETS_PATH = os.path.join(MODELS_DIR, "fea_target_names.joblib")


# --- Globals ---
app = FastAPI(title="FDM Property Simulator API")
model_kaggle = None
model_c3 = None
model_fea = None
model_fatigue = None
fea_target_names: List[str] = [] # To store the FEA model's target column names

# --- Pydantic Models (Data Validation) ---
class KaggleParams(BaseModel):
    layer_height: float
    wall_thickness: int
    infill_density: int
    infill_pattern: str
    nozzle_temperature: int
    bed_temperature: int
    print_speed: int
    material: str
    fan_speed: int

class C3Params(BaseModel):
    Temperature: float
    Speed: float
    Angle: float
    Height: float
    Fill: float

class FEAParams(BaseModel):
    Material_Bonding_Perfection: float
    Material_Youngs_Modulus_GPa: float  # Corrected: Young's -> Youngs
    Material_Tensile_Yield_Strenght_MPa: float
    Material_Poissons_Ratio: float    # Corrected: Poisson's -> Poissons
    User_Infill_Pattern: str
    User_Infill_Density: float
    User_Line_Thickenss_mm: float
    User_Layer_Height_mm: float

class FatigueParams(BaseModel):
    Nozzle_Diameter: float
    Print_Speed: float
    Nozzle_Temperature: float
    Stress_Level: float

# --- Server Lifecycle ---
@app.on_event("startup")
def load_models():
    """Load all ML models into memory on server startup."""
    global model_kaggle, model_c3, model_fea, model_fatigue, fea_target_names

    # Load Kaggle Model
    try:
        model_kaggle = joblib.load(MODEL_KAGGLE_PATH)
        print(f"Successfully loaded model 'kaggle' from {MODEL_KAGGLE_PATH}")
    except FileNotFoundError:
        print(f"Warning: Model file not found at {MODEL_KAGGLE_PATH}. Endpoint for 'kaggle' will be disabled.")
    
    # Load C3 Model
    try:
        model_c3 = joblib.load(MODEL_C3_PATH)
        print(f"Successfully loaded model 'c3' from {MODEL_C3_PATH}")
    except FileNotFoundError:
        print(f"Warning: Model file not found at {MODEL_C3_PATH}. Endpoint for 'c3' will be disabled.")

    # Load FEA Model
    try:
        model_fea = joblib.load(MODEL_FEA_PATH)
        print(f"Successfully loaded model 'fea' from {MODEL_FEA_PATH}")
        
        # --- NEW: Load target names from the dedicated file ---
        try:
            fea_target_names = joblib.load(FEA_TARGETS_PATH)
            print(f"Successfully loaded FEA target names ({len(fea_target_names)} properties).")
        except FileNotFoundError:
            print(f"Warning: FEA target names file not found at {FEA_TARGETS_PATH}. FEA endpoint may fail.")
            fea_target_names = [] # Ensure it's an empty list
        except Exception as e:
            print(f"Error loading FEA target names: {e}")
            fea_target_names = []

    except FileNotFoundError:
        print(f"Warning: Model file not found at {MODEL_FEA_PATH}. Endpoint for 'fea' will be disabled.")

    # Load Fatigue Model
    try:
        model_fatigue = joblib.load(MODEL_FATIGUE_PATH)
        print(f"Successfully loaded model 'fatigue' from {MODEL_FATIGUE_PATH}")
    except FileNotFoundError:
        print(f"Warning: Model file not found at {MODEL_FATIGUE_PATH}. Endpoint for 'fatigue' will be disabled.")


# --- Middleware (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (for local file:// access)
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "FDM Property Simulator API is running."}

@app.get("/materials")
def get_materials():
    """Serves the materials.json file."""
    try:
        # Use a standard file read for JSON
        with open(MATERIALS_JSON_PATH, 'r') as f:
            import json
            materials_data = json.load(f)
        return materials_data
    except FileNotFoundError:
        print(f"Error: {MATERIALS_JSON_PATH} not found.")
        raise HTTPException(status_code=404, detail="Materials database not found.")
    except Exception as e:
        print(f"Error reading materials JSON: {e}")
        raise HTTPException(status_code=500, detail="Could not process materials database.")

# --- Prediction Endpoints ---

@app.post("/predict/kaggle")
def predict_kaggle(data: KaggleParams):
    if model_kaggle is None:
        raise HTTPException(status_code=503, detail="Kaggle model is not loaded.")
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model_kaggle.predict(df)
        return {
            "tensile_strength": prediction[0, 0],
            "roughness": prediction[0, 1],
            "elongation": prediction[0, 2]
        }
    except Exception as e:
        print(f"Kaggle prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict/c3")
def predict_c3(data: C3Params):
    if model_c3 is None:
        raise HTTPException(status_code=503, detail="C3 model is not loaded.")
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model_c3.predict(df)
        return {
            "Tensile_Strength_MPa": prediction[0, 0],
            "Elongation_at_Break_percent": prediction[0, 1]
        }
    except Exception as e:
        print(f"C3 prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict/fea")
def predict_fea(data: FEAParams):
    if model_fea is None:
        raise HTTPException(status_code=503, detail="FEA model is not loaded.")
    try:
        # Create DataFrame. The Pydantic model now has the *correct* names.
        df = pd.DataFrame([data.dict()])
        
        # Run prediction
        prediction = model_fea.predict(df)
        
        # --- NEW: Use the loaded target names ---
        if not fea_target_names:
             print("Error: /predict/fea called but fea_target_names is empty.")
             raise HTTPException(status_code=500, detail="FEA target names not loaded on server.")
             
        results = {name: value for name, value in zip(fea_target_names, prediction[0])}
        return results
        
    except ValueError as e:
        # This is where the "columns are missing" error would be caught
        print(f"FEA prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    except Exception as e:
        print(f"FEA prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict/fatigue")
def predict_fatigue(data: FatigueParams):
    if model_fatigue is None:
        raise HTTPException(status_code=503, detail="Fatigue model is not loaded.")
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model_fatigue.predict(df)
        return {"Fatigue_Lifetime": prediction[0]}
    except Exception as e:
        print(f"Fatigue prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

