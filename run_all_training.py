import os
import subprocess
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- Configuration ---
# Define directory paths
DATA_RAW_DIR = os.path.join("data", "raw")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

# Input data paths
PATH_KAGGLE_RAW = os.path.join(DATA_RAW_DIR, "data.csv")
PATH_C3_RAW = os.path.join(DATA_RAW_DIR, "C3-RAW DATA.csv")
PATH_FEA_RAW = os.path.join(DATA_RAW_DIR, "3D_Printing_Data.xlsx - Sheet1.csv")
PATH_FATIGUE_RAW = os.path.join(DATA_RAW_DIR, "1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv")

# Processed data paths
PATH_C3_PROCESSED = os.path.join(DATA_PROCESSED_DIR, "c3_processed_data.csv")

# Model output paths
MODEL_KAGGLE_PATH = os.path.join(MODELS_DIR, "model_kaggle.joblib")
MODEL_C3_PATH = os.path.join(MODELS_DIR, "model_c3.joblib")
MODEL_FEA_PATH = os.path.join(MODELS_DIR, "model_fea.joblib")
MODEL_FATIGUE_PATH = os.path.join(MODELS_DIR, "model_fatigue.joblib")


# --- Model 1: Kaggle ---
def train_kaggle_model():
    """Trains the Kaggle model (Roughness, Tensile Strength, Elongation)."""
    print("--- Training Kaggle Model ---")
    try:
        df = pd.read_csv(PATH_KAGGLE_RAW)

        # Define features (X) and targets (y)
        feature_cols = [
            'layer_height', 'wall_thickness', 'infill_density', 'infill_pattern',
            'nozzle_temperature', 'bed_temperature', 'print_speed', 'material', 'fan_speed'
        ]
        target_cols = ['roughness', 'tension_strenght', 'elongation']

        X = df[feature_cols]
        y = df[target_cols]

        # Define preprocessing steps
        numeric_features = ['layer_height', 'wall_thickness', 'infill_density',
                            'nozzle_temperature', 'bed_temperature', 'print_speed', 'fan_speed']
        categorical_features = ['infill_pattern', 'material']

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the full pipeline with the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

        # Train the model
        print("Training Kaggle model...")
        pipeline.fit(X, y)

        # Save the pipeline
        joblib.dump(pipeline, MODEL_KAGGLE_PATH)
        print(f"Kaggle model saved to {MODEL_KAGGLE_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_KAGGLE_RAW}")
    except Exception as e:
        print(f"Error training Kaggle model: {e}")


# --- Model 2: C3 ---
def process_c3_data():
    """Processes the raw C3 stress-strain data into a summary table."""
    print("--- Processing C3 Data ---")
    try:
        df = pd.read_csv(PATH_C3_RAW)
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Group by the unique print parameters
        group_cols = ['Temperature', 'Speed', 'Angle', 'Height', 'Fill']
        
        # Calculate max stress (Tensile Strength) and max strain (Elongation) for each group
        processed = df.groupby(group_cols).agg(
            Tensile_Strength_MPa=('Stress', 'max'),
            Elongation_at_Break_percent=('Strain', 'max')
        ).reset_index()
        
        # Convert Elongation from strain (mm/mm) to percentage
        processed['Elongation_at_Break_percent'] *= 100

        # Save the processed data
        processed.to_csv(PATH_C3_PROCESSED, index=False)
        print(f"Processed C3 data saved to {PATH_C3_PROCESSED}")
        return True

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_C3_RAW}")
        return False
    except Exception as e:
        print(f"Error processing C3 data: {e}")
        return False

def train_c3_model():
    """Trains the C3 model (Tensile Strength, Elongation) using processed data."""
    print("--- Training C3 Model ---")
    try:
        # Ensure processed data exists
        if not os.path.exists(PATH_C3_PROCESSED):
            if not process_c3_data():
                print("C3 data processing failed. Aborting C3 model training.")
                return

        df = pd.read_csv(PATH_C3_PROCESSED)

        feature_cols = ['Temperature', 'Speed', 'Angle', 'Height', 'Fill']
        target_cols = ['Tensile_Strength_MPa', 'Elongation_at_Break_percent']

        X = df[feature_cols]
        y = df[target_cols]

        # Simple StandardScaler as all features are numeric
        preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

        print("Training C3 model...")
        pipeline.fit(X, y)
        
        joblib.dump(pipeline, MODEL_C3_PATH)
        print(f"C3 model saved to {MODEL_C3_PATH}")

    except Exception as e:
        print(f"Error training C3 model: {e}")


# --- Model 3: FEA ---
def train_fea_model():
    """Trains the FEA model (Multi-output anisotropic material card)."""
    print("--- Training FEA Model ---")
    try:
        df = pd.read_csv(PATH_FEA_RAW)
        
        # --- FIX V3 ---
        # Correctly clean column names
        df.columns = (df.columns
                      .str.replace(' ', '_')
                      .str.replace(':', '')
                      .str.replace("'", "") # Remove apostrophes
                      .str.replace('(', '_', regex=False)
                      .str.replace(')', '', regex=False)
                     )

        # --- FIX V3 ---
        # Correct typos in feature names (Young's -> Youngs, Poisson's -> Poissons)
        feature_cols = [
            "Material_Bonding_Perfection", "Material_Youngs_Modulus_GPa", 
            "Material_Tensile_Yield_Strenght_MPa", "Material_Poissons_Ratio",
            "User_Infill_Pattern", "User_Infill_Density", 
            "User_Line_Thickenss_mm", "User_Layer_Height_mm"
        ]
        
        # Define numeric and categorical features
        numeric_features = [
            "Material_Bonding_Perfection", "Material_Youngs_Modulus_GPa",
            "Material_Tensile_Yield_Strenght_MPa", "Material_Poissons_Ratio",
            "User_Infill_Density", "User_Line_Thickenss_mm", "User_Layer_Height_mm"
        ]
        categorical_features = ["User_Infill_Pattern"]

        # Define targets
        target_cols = [col for col in df.columns if col.startswith('Specimen_') and col != 'Specimen_Infill_Pattern']
        
        if not target_cols:
            print("Error: No 'Specimen_' target columns found in FEA data.")
            return

        # --- ROBUST FIX V2 ---
        # 1. Define all columns that MUST be numeric
        numeric_cols_to_clean = target_cols + numeric_features
        
        # 2. Force-convert all target AND numeric feature columns to numeric
        df[numeric_cols_to_clean] = df[numeric_cols_to_clean].apply(pd.to_numeric, errors='coerce')

        # 3. Drop any rows that now contain NaN values in these critical columns
        original_rows = len(df)
        df = df.dropna(subset=numeric_cols_to_clean) # Clean based on features AND targets
        print(f"Dropped {original_rows - len(df)} rows with bad data from FEA dataset.")
            
        print(f"FEA model will predict {len(target_cols)} properties.")

        X = df[feature_cols] # Now X is clean
        y = df[target_cols] # y is clean

        # Define preprocessing
        # We can now be confident the numeric_transformer will only see numbers
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Use MultiOutputRegressor wrapping a RandomForestRegressor
        base_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', MultiOutputRegressor(base_regressor))])

        print("Training FEA model (this may take a minute)...")
        pipeline.fit(X, y)

        joblib.dump(pipeline, MODEL_FEA_PATH)
        print(f"FEA model saved to {MODEL_FEA_PATH}")
        
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_FEA_RAW}")
    except Exception as e:
        print(f"Error training FEA model: {e}")


# --- Model 4: Fatigue ---
def train_fatigue_model():
    """Trains the Fatigue model (Fatigue Lifetime)."""
    print("--- Training Fatigue Model ---")
    try:
        # --- FIX ---
        # Correctly read the CSV, setting header to row 0 and skipping row 1 (units)
        df = pd.read_csv(PATH_FATIGUE_RAW, header=0, skiprows=[1])

        # Drop rows with NaN values which can occur from malformed data
        df = df.dropna(subset=['Nozzle Diameter', 'Print Speed', 'Nozzle Temperature', 'Stress Level', 'Fatigue Lifetime'])
        
        # Clean column names
        df.columns = df.columns.str.replace(' ', '_').str.replace(r'\[.*\]', '', regex=True)

        feature_cols = ['Nozzle_Diameter', 'Print_Speed', 'Nozzle_Temperature', 'Stress_Level']
        target_col = 'Fatigue_Lifetime'

        X = df[feature_cols]
        y = df[target_col]

        # Simple StandardScaler as all features are numeric
        preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

        print("Training Fatigue model...")
        pipeline.fit(X, y)
        
        joblib.dump(pipeline, MODEL_FATIGUE_PATH)
        print(f"Fatigue model saved to {MODEL_FATIGUE_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_FATIGUE_RAW}")
    except KeyError as e:
        print(f"KeyError: {e}. One of the expected columns is missing or misnamed in {PATH_FATIGUE_RAW}.")
    except Exception as e:
        print(f"Error training Fatigue model: {e}")


# --- Main execution ---
def main():
    """Main function to create directories, train all models, and start the server."""
    
    # Create necessary directories if they don't exist
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # --- "Smart" Training: Only train if model file is missing ---
    
    if not os.path.exists(MODEL_KAGGLE_PATH):
        train_kaggle_model()
    else:
        print("--- Kaggle Model already trained. Skipping. ---")
        
    if not os.path.exists(MODEL_C3_PATH):
        train_c3_model()
    else:
        print("--- C3 Model already trained. Skipping. ---")
        
    if not os.path.exists(MODEL_FEA_PATH):
        train_fea_model()
    else:
        print("--- FEA Model already trained. Skipping. ---")
        
    if not os.path.exists(MODEL_FATIGUE_PATH):
        train_fatigue_model()
    else:
        print("--- Fatigue Model already trained. Skipping. ---")
    
    print("\nAll models checked/trained.")
    print("Starting FastAPI server...")
    
    # Start the Uvicorn server as a subprocess
    try:
        subprocess.run(["uvicorn", "main:app", "--reload"], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Failed to start server: {e}")

if __name__ == "__main__":
    main()

