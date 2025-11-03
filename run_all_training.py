import pandas as pd
import numpy as np
import joblib
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- 1. DEFINE FILE PATHS ---
# Define base directory
BASE_DIR = Path(__file__).resolve().parent

# Define data directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Define model directory
MODELS_DIR = BASE_DIR / "models"

# Define file paths for datasets
KAGGLE_DATA_PATH = DATA_RAW_DIR / "data.csv"
C3_DATA_PATH = DATA_RAW_DIR / "C3-RAW DATA.csv"
FEA_DATA_PATH = DATA_RAW_DIR / "3D_Printing_Data.xlsx - Sheet1.csv"
FATIGUE_DATA_PATH = DATA_RAW_DIR / "1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv" # New dataset

# Define paths for processed data
C3_PROCESSED_PATH = DATA_PROCESSED_DIR / "c3_processed_data.csv"

# Define paths for model outputs
KAGGLE_MODEL_PATH = MODELS_DIR / "model_kaggle.joblib"
C3_MODEL_PATH = MODELS_DIR / "model_c3.joblib"
FEA_MODEL_PATH = MODELS_DIR / "model_fea.joblib"
FATIGUE_MODEL_PATH = MODELS_DIR / "model_fatigue.joblib" # New model

def clean_fea_columns(df):
    """Applies standardized cleaning to FEA column names."""
    # Replace special characters with underscore, strip leading/trailing underscores
    df.columns = df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')
    return df

def train_kaggle_model():
    """Trains the model on the Kaggle dataset."""
    print(f"--- Training Kaggle Model ---")
    try:
        df = pd.read_csv(KAGGLE_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: {KAGGLE_DATA_PATH} not found.")
        print("Please place 'data.csv' in data/raw/")
        return

    # Define features and targets
    feature_cols = [
        'layer_height', 'wall_thickness', 'infill_density', 'infill_pattern',
        'nozzle_temperature', 'bed_temperature', 'print_speed', 'material', 'fan_speed'
    ]
    target_cols = ['roughness', 'tension_strenght', 'elongation']
    
    X = df[feature_cols]
    y = df[target_cols]

    # Define categorical and numerical features
    categorical_features = ['infill_pattern', 'material']
    numerical_features = [col for col in feature_cols if col not in categorical_features]

    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline with a MultiOutputRegressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
    ])

    # Train the model
    print("Training Kaggle model...")
    pipeline.fit(X, y)

    # Save the model
    joblib.dump(pipeline, KAGGLE_MODEL_PATH)
    print(f"Kaggle model saved to {KAGGLE_MODEL_PATH}")


def train_c3_model():
    """Processes the C3 data and trains the C3 model."""
    print(f"--- Training C3 Model ---")
    try:
        df_raw = pd.read_csv(C3_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: {C3_DATA_PATH} not found.")
        print("Please place 'C3-RAW DATA.csv' in data/raw/")
        return

    # --- C3 Data Processing ---
    print("Processing C3 raw data...")
    # Clean column names
    df_raw.columns = df_raw.columns.str.strip()
    
    # Group by print parameters to identify individual tests
    params = ['Temperature', 'Speed', 'Angle', 'Height', 'Fill']
    grouped = df_raw.groupby(params)

    processed_data = []
    for (temp, speed, angle, height, fill), group in grouped:
        # Find max stress (Tensile Strength)
        max_stress = group['Stress'].max()
        
        # Find strain at max stress (Elongation at Break)
        elongation = group.loc[group['Stress'].idxmax(), 'Strain'] * 100 # As percentage
        
        processed_data.append({
            'Temperature': temp,
            'Speed': speed,
            'Angle': angle,
            'Height': height,
            'Fill': fill,
            'Tensile_Strength_MPa': max_stress,
            'Elongation_at_Break_percent': elongation
        })

    df_processed = pd.DataFrame(processed_data)
    df_processed.to_csv(C3_PROCESSED_PATH, index=False)
    print(f"Processed C3 data saved to {C3_PROCESSED_PATH}")

    # --- C3 Model Training ---
    feature_cols = ['Temperature', 'Speed', 'Angle', 'Height', 'Fill']
    target_cols = ['Tensile_Strength_MPa', 'Elongation_at_Break_percent']
    
    X = df_processed[feature_cols]
    y = df_processed[target_cols]

    # Preprocessing (all features are numeric)
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), feature_cols)],
        remainder='passthrough'
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
    ])

    # Train the model
    print("Training C3 model...")
    pipeline.fit(X, y)

    # Save the model
    joblib.dump(pipeline, C3_MODEL_PATH)
    print(f"C3 model saved to {C3_MODEL_PATH}")


def train_fea_model():
    """Trains the model on the FEA dataset."""
    print(f"--- Training FEA Model ---")
    try:
        df = pd.read_csv(FEA_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: {FEA_DATA_PATH} not found.")
        print("Please place '3D_Printing_Data.xlsx - Sheet1.csv' in data/raw/")
        return

    # Clean column names
    df = clean_fea_columns(df)

    # Define feature (input) columns
    feature_cols = [
        'Material_Bonding_Perfection',
        'Material_Young_s_Modulus_GPa',
        'Material_Tensile_Yield_Strenght_MPa',
        'Material_Poisson_s_Ratio',
        'User_Infill_Pattern',
        'User_Infill_Density',
        'User_Line_Thickenss_mm',
        'User_Layer_Height_mm'
    ]

    # --- FIX: Dynamically identify numeric target columns ---
    # Select all Specimen columns
    all_specimen_cols = [col for col in df.columns if col.startswith('Specimen_')]
    
    # Identify which of them are non-numeric (like 'Specimen_Infill_Pattern')
    non_numeric_specimen_cols = df[all_specimen_cols].select_dtypes(exclude=np.number).columns
    
    # The target columns are all Specimen columns MINUS the non-numeric ones.
    target_cols = [col for col in all_specimen_cols if col not in non_numeric_specimen_cols]
    # --- END FIX ---
    
    print(f"Found {len(feature_cols)} features and {len(target_cols)} numeric targets for FEA model.")

    X = df[feature_cols]
    y = df[target_cols]

    # Define categorical and numerical features
    categorical_features = ['User_Infill_Pattern']
    numerical_features = [col for col in feature_cols if col not in categorical_features]

    # Define preprocessing
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
    ])

    # Train the model
    print("Training FEA model (this may take a minute)...")
    pipeline.fit(X, y)

    # Save the model
    joblib.dump(pipeline, FEA_MODEL_PATH)
    print(f"FEA model saved to {FEA_MODEL_PATH}")

def train_fatigue_model():
    """Trains the model on the Fatigue dataset."""
    print(f"--- Training Fatigue Model ---")
    try:
        # --- FIX: Read header from row 0, then skip row 1 (the unit row) ---
        df = pd.read_csv(FATIGUE_DATA_PATH, header=0, skiprows=[1])
    except FileNotFoundError:
        print(f"ERROR: {FATIGUE_DATA_PATH} not found.")
        print("Please place '1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv' in data/raw/")
        return
        
    # Clean and rename columns for consistency
    df = df.rename(columns={
        'Nozzle Diameter': 'Nozzle_Diameter',
        'Print Speed': 'Print_Speed',
        'Nozzle Temperature': 'Nozzle_Temperature',
        'Stress Level': 'Stress_Level',
        'Fatigue Lifetime': 'Fatigue_Lifetime'
    })
    
    # Drop unnecessary columns and rows with missing data
    df = df.drop(columns=['No.', 'Material Code'], errors='ignore')
    df = df.dropna()

    # Define features and target
    feature_cols = ['Nozzle_Diameter', 'Print_Speed', 'Nozzle_Temperature', 'Stress_Level']
    target_col = 'Fatigue_Lifetime'
    
    X = df[feature_cols]
    y = df[target_col]

    # Define preprocessing (all features are numeric)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Using a single regressor as there is only one target
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Train the model
    print("Training Fatigue model...")
    pipeline.fit(X, y)

    # Save the model
    joblib.dump(pipeline, FATIGUE_MODEL_PATH)
    print(f"Fatigue model saved to {FATIGUE_MODEL_PATH}")


def main():
    """Main function to run all steps."""
    
    # Create necessary directories if they don't exist
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting model training process...")
    
    print("-" * 30)
    train_kaggle_model()
    print("-" * 30)
    train_c3_model()
    print("-" * 30)
    train_fea_model()
    print("-" * 30)
    train_fatigue_model() # Add new training step
    print("-" * 30)
    
    print("\nAll models have been trained and saved successfully.")
    
    # Launch the FastAPI server
    print("\nLaunching FastAPI server with uvicorn...")
    print("Access the frontend by opening fdm_simulator.html in your browser.")
    print("Press Ctrl+C to stop the server.")
    
    # Use subprocess to run uvicorn
    # This will keep running until you stop it (Ctrl+C)
    try:
        subprocess.run(["uvicorn", "main:app", "--reload"], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except FileNotFoundError:
        print("\nERROR: 'uvicorn' command not found.")
        print("Please ensure your virtual environment is active and dependencies are installed (`uv pip install .`)")

if __name__ == "__main__":
    main()


