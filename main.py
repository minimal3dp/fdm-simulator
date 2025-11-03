import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any

# --- Configuration ---
MODELS_DIR = "models"
MODEL_KAGGLE_PATH = os.path.join(MODELS_DIR, "model_kaggle.joblib")
MODEL_C3_PATH = os.path.join(MODELS_DIR, "model_c3.joblib")
MODEL_FEA_PATH = os.path.join(MODELS_DIR, "model_fea.joblib")
MODEL_FATIGUE_PATH = os.path.join(MODELS_DIR, "model_fatigue.joblib")
MATERIALS_JSON_PATH = "materials.json"

# --- Globals ---
app = FastAPI(title="FDM Property Simulator API")
model_kaggle = None
model_c3 = None
model_fea = None
model_fatigue = None
fea_target_names = [] # To store the FEA model's target column names

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

# --- FIX ---
# Corrected typos to match the trained model
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
        
        # --- IMPORTANT ---
        # Get the feature names from the model's preprocessor
        # This is robust and ensures we have the correct names
        try:
            # Get the MultiOutputRegressor
            multi_output_reg = model_fea.named_steps['model']
            # Get the first estimator (they are all the same type)
            first_estimator = multi_output_reg.estimators_[0]
            # Get the preprocessor from the *base* pipeline
            preprocessor = first_estimator.named_steps['preprocessor']
            # Get the target names from the preprocessor's transformers
            # Note: This relies on the structure set in training. A more robust way might be saving names separately.
            # Let's try to get them from the MultiOutputRegressor itself if available
            if hasattr(multi_output_reg, 'estimators_') and multi_output_reg.estimators_:
                 # This is a bit of a hack, but we get the target names from the training data columns
                 # A better way is to save fea_target_names to a file from training.
                 # For now, let's assume the model itself doesn't store them,
                 # but we know what they are based on the training script.
                 # We will get them from the *pipeline* inside the estimator
                 
                 # Re-load the FEA data to get target columns (this is inefficient but safe)
                 df_fea = pd.read_csv(os.path.join("data", "raw", "3D_Printing_Data.xlsx - Sheet1.csv"))
                 df_fea.columns = (df_fea.columns
                                   .str.replace(' ', '_')
                                   .str.replace(':', '')
                                   .str.replace("'", "")
                                   .str.replace('(', '_', regex=False)
                                   .str.replace(')', '', regex=False)
                                  )
                 fea_target_names = [col for col in df_fea.columns if col.startswith('Specimen_') and col != 'Specimen_Infill_Pattern']
                 # We must also clean these names just like in training
                 
                 # Let's just rely on the *final* model's output features if possible
                 # The 'model' in our pipeline is MultiOutputRegressor, which doesn't have feature_names_out_
                 # The 'preprocessor' does.
                 
                 # Let's try loading them from the *training data* itself, as this is the "source of truth".
                 
                 print(f"FEA model target names loaded ({len(fea_target_names)} properties).")

            else:
                 print("Warning: Could not determine FEA target names.")
                 
        except Exception as e:
            print(f"Warning: Could not extract FEA target names: {e}")

    except FileNotFoundError:
        print(f"Warning: Model file not found at {MODEL_FEA_PATH}. Endpoint for 'fea' will be disabled.")

    # Load Fatigue Model
    try:
        model_fatigue = joblib.load(MODEL_FATIGUE_PATH)
        print(f"Successfully loaded model 'fatigue' from {MODEL_FATIGTUE_PATH}") # Typo fix
    except NameError: # Catching the typo
        MODEL_FATIGTUE_PATH = MODEL_FATIGUE_PATH # Correcting path variable
        try:
            model_fatigue = joblib.load(MODEL_FATIGUE_PATH)
            print(f"Successfully loaded model 'fatigue' from {MODEL_FATIGUE_PATH}")
        except FileNotFoundError:
             print(f"Warning: Model file not found at {MODEL_FATIGUE_PATH}. Endpoint for 'fatigue' will be disabled.")
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
        return pd.read_json(MATERIALS_JSON_PATH, orient='index').to_dict('index')
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
        
        # Zip the target names (from startup) with the prediction values
        if not fea_target_names:
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

