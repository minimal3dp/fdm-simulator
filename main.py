import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path

# --- 1. Setup and Model Loading ---

# Define paths consistent with run_all_training.py
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_RAW_DIR = BASE_DIR / "data" / "raw" # Needed for FEA column names

KAGGLE_MODEL_PATH = MODELS_DIR / "model_kaggle.joblib"
C3_MODEL_PATH = MODELS_DIR / "model_c3.joblib"
FEA_MODEL_PATH = MODELS_DIR / "model_fea.joblib"
FATIGUE_MODEL_PATH = MODELS_DIR / "model_fatigue.joblib" # New model path
FEA_DATA_PATH = DATA_RAW_DIR / "3D_Printing_Data.xlsx - Sheet1.csv" # For column names

# Helper function to clean column names (must match the one in training)
def clean_fea_columns(df):
    """Applies standardized cleaning to FEA column names."""
    df.columns = df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')
    return df

# Helper function to get FEA target column names
def get_fea_target_names():
    """Reads the FEA data file to get the list of numeric target columns."""
    try:
        df = pd.read_csv(FEA_DATA_PATH)
        df = clean_fea_columns(df)
        
        # --- FIX: Dynamically identify numeric target columns ---
        all_specimen_cols = [col for col in df.columns if col.startswith('Specimen_')]
        non_numeric_specimen_cols = df[all_specimen_cols].select_dtypes(exclude=np.number).columns
        target_cols = [col for col in all_specimen_cols if col not in non_numeric_specimen_cols]
        # --- END FIX ---
        
        print(f"Found {len(target_cols)} numeric target names for FEA model.")
        return target_cols
    except FileNotFoundError:
        print(f"Warning: {FEA_DATA_PATH} not found. Cannot determine FEA target names.")
        return []
    except Exception as e:
        print(f"Error loading FEA target names: {e}")
        return []

fea_target_names = get_fea_target_names()

# Load models at startup
try:
    model_kaggle = joblib.load(KAGGLE_MODEL_PATH)
    print("Kaggle model loaded successfully.")
except FileNotFoundError:
    print(f"Warning: {KAGGLE_MODEL_PATH} not found. Kaggle API will not work.")
    model_kaggle = None

try:
    model_c3 = joblib.load(C3_MODEL_PATH)
    print("C3 model loaded successfully.")
except FileNotFoundError:
    print(f"Warning: {C3_MODEL_PATH} not found. C3 API will not work.")
    model_c3 = None

try:
    model_fea = joblib.load(FEA_MODEL_PATH)
    print("FEA model loaded successfully.")
except FileNotFoundError:
    print(f"Warning: {FEA_MODEL_PATH} not found. FEA API will not work.")
    model_fea = None

# New model loading
try:
    model_fatigue = joblib.load(FATIGUE_MODEL_PATH)
    print("Fatigue model loaded successfully.")
except FileNotFoundError:
    print(f"Warning: {FATIGUE_MODEL_PATH} not found. Fatigue API will not work.")
    model_fatigue = None


# --- 2. FastAPI App Setup ---
app = FastAPI(
    title="FDM 3D Print Property Simulator API",
    description="Serves ML models for predicting 3D print properties."
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for simple local development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Pydantic Input Models ---
# These must match the features used in run_all_training.py

class KaggleInput(BaseModel):
    layer_height: float
    wall_thickness: float
    infill_density: float
    infill_pattern: str
    nozzle_temperature: float
    bed_temperature: float
    print_speed: float
    material: str
    fan_speed: float

class C3Input(BaseModel):
    Temperature: float
    Speed: float
    Angle: float
    Height: float
    Fill: float

class FEAInput(BaseModel):
    # These names must match the cleaned column names
    Material_Bonding_Perfection: float
    Material_Young_s_Modulus_GPa: float
    Material_Tensile_Yield_Strenght_MPa: float
    Material_Poisson_s_Ratio: float
    User_Infill_Pattern: str
    User_Infill_Density: float
    User_Line_Thickenss_mm: float
    User_Layer_Height_mm: float

# New Pydantic model for Fatigue
class FatigueInput(BaseModel):
    # These must match the renamed columns from training
    Nozzle_Diameter: float
    Print_Speed: float
    Nozzle_Temperature: float
    Stress_Level: float


# --- 4. API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "FDM Property Simulator API is running. Access the frontend via fdm_simulator.html"}

@app.post("/predict/kaggle")
def predict_kaggle(data: KaggleInput) -> Dict[str, float]:
    if model_kaggle is None:
        raise HTTPException(status_code=503, detail="Kaggle model is not loaded.")
    try:
        # Convert Pydantic model to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # Make prediction
        prediction = model_kaggle.predict(input_df)
        
        # Return formatted results
        results = {
            "roughness": prediction[0, 0],
            "tensile_strength": prediction[0, 1],
            "elongation": prediction[0, 2]
        }
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/c3")
def predict_c3(data: C3Input) -> Dict[str, float]:
    if model_c3 is None:
        raise HTTPException(status_code=503, detail="C3 model is not loaded.")
    try:
        # Convert Pydantic model to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # Make prediction
        prediction = model_c3.predict(input_df)
        
        # Return formatted results
        results = {
            "Tensile_Strength_MPa": prediction[0, 0],
            "Elongation_at_Break_percent": prediction[0, 1]
        }
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/fea")
def predict_fea(data: FEAInput) -> Dict[str, float]:
    if model_fea is None:
        raise HTTPException(status_code=503, detail="FEA model is not loaded.")
    if not fea_target_names:
        raise HTTPException(status_code=500, detail="FEA target names not loaded on server.")
        
    try:
        # Convert Pydantic model to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # Make prediction
        prediction = model_fea.predict(input_df)
        
        # Dynamically zip the target names with the prediction values
        results = dict(zip(fea_target_names, prediction[0]))
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# New endpoint for Fatigue model
@app.post("/predict/fatigue")
def predict_fatigue(data: FatigueInput) -> Dict[str, float]:
    if model_fatigue is None:
        raise HTTPException(status_code=503, detail="Fatigue model is not loaded.")
    try:
        # Convert Pydantic model to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # Make prediction
        prediction = model_fatigue.predict(input_df)
        
        # Return formatted results
        results = {
            "Fatigue_Lifetime": prediction[0] # Only one value is predicted
        }
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

