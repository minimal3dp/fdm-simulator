import os
import re  # Import regex library
import subprocess

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Configuration ---
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File Paths
PATH_KAGGLE_RAW = os.path.join(RAW_DIR, "data.csv")
PATH_C3_RAW = os.path.join(RAW_DIR, "C3-RAW DATA.csv")
PATH_FEA_RAW = os.path.join(RAW_DIR, "3D_Printing_Data.xlsx - Sheet1.csv")
PATH_FATIGUE_RAW = os.path.join(RAW_DIR, "1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv")
PATH_ACCURACY_RAW = os.path.join(RAW_DIR, "dimensional_accuracy_deswal.csv")
PATH_WARPAGE_RAW = os.path.join(RAW_DIR, "warpage_data_nazan.csv")
PATH_HARDNESS_RAW = os.path.join(RAW_DIR, "hardness_data_kadam.csv")
PATH_MULTIMATERIAL_RAW = os.path.join(RAW_DIR, "multimaterial_bond_yadav.csv")  # v8: New Path
PATH_COMPOSITE_RAW = os.path.join(
    RAW_DIR, "composite_data_alarifi.csv"
)  # v9: New Composite dataset path

PATH_C3_PROCESSED = os.path.join(PROCESSED_DIR, "c3_processed_data.csv")

MODEL_KAGGLE_PATH = os.path.join(MODELS_DIR, "model_kaggle.joblib")
MODEL_C3_PATH = os.path.join(MODELS_DIR, "model_c3.joblib")
MODEL_FEA_PATH = os.path.join(MODELS_DIR, "model_fea.joblib")
MODEL_FATIGUE_PATH = os.path.join(MODELS_DIR, "model_fatigue.joblib")
MODEL_ACCURACY_PATH = os.path.join(MODELS_DIR, "model_accuracy.joblib")
MODEL_WARPAGE_PATH = os.path.join(MODELS_DIR, "model_warpage.joblib")
MODEL_HARDNESS_PATH = os.path.join(MODELS_DIR, "model_hardness.joblib")
MODEL_MULTIMATERIAL_PATH = os.path.join(MODELS_DIR, "model_multimaterial.joblib")  # v8: New Model
MODEL_COMPOSITE_PATH = os.path.join(MODELS_DIR, "model_composite.joblib")  # v9: New Model

# Path for FEA target names
FEA_TARGETS_PATH = os.path.join(MODELS_DIR, "fea_target_names.joblib")


# --- Model 1: Kaggle ---
def train_kaggle_model():
    """Trains the Kaggle model (Tensile, Roughness, Elongation)."""
    print("--- Training Kaggle Model ---")
    try:
        df = pd.read_csv(PATH_KAGGLE_RAW)

        # Define features and targets
        feature_cols = [
            "layer_height",
            "wall_thickness",
            "infill_density",
            "infill_pattern",
            "nozzle_temperature",
            "bed_temperature",
            "print_speed",
            "material",
            "fan_speed",
        ]
        target_cols = ["tension_strenght", "roughness", "elongation"]

        # Define numeric and categorical features
        numeric_features = [
            "layer_height",
            "wall_thickness",
            "infill_density",
            "nozzle_temperature",
            "bed_temperature",
            "print_speed",
            "fan_speed",
        ]
        categorical_features = ["infill_pattern", "material"]

        X = df[feature_cols]
        y = df[target_cols]

        # Define preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        # Create the full pipeline
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        # Train the model
        print("Training Kaggle model...")
        pipeline.fit(X, y)
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

        # Group by all parameter columns
        group_cols = ['Temperature', 'Speed', 'Angle', 'Height', 'Fill']

        # Calculate max stress (Tensile Strength) and corresponding strain (Elongation)
        results = []
        for params, group in df.groupby(group_cols):
            max_stress_idx = group['Stress'].idxmax()
            max_stress = group.loc[max_stress_idx, 'Stress']
            elongation_at_break = group.loc[max_stress_idx, 'Strain']

            row = list(params) + [max_stress, elongation_at_break * 100]  # Strain as %
            results.append(row)

        # Create a new DataFrame
        processed_df = pd.DataFrame(
            results, columns=group_cols + ['Tensile_Strength_MPa', 'Elongation_at_Break_percent']
        )
        processed_df.to_csv(PATH_C3_PROCESSED, index=False)
        print(f"C3 processed data saved to {PATH_C3_PROCESSED}")
        return True

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_C3_RAW}")
        return False
    except Exception as e:
        print(f"Error processing C3 data: {e}")
        return False


def train_c3_model():
    """Trains the C3 model (Tensile, Elongation) on processed data."""
    print("--- Training C3 Model ---")
    try:
        df = pd.read_csv(PATH_C3_PROCESSED)

        feature_cols = ['Temperature', 'Speed', 'Angle', 'Height', 'Fill']
        target_cols = ['Tensile_Strength_MPa', 'Elongation_at_Break_percent']

        X = df[feature_cols]
        y = df[target_cols]

        # Simple pipeline (all features are numeric)
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        print("Training C3 model...")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_C3_PATH)
        print(f"C3 model saved to {MODEL_C3_PATH}")

    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PATH_C3_PROCESSED}. Run processing first.")
    except Exception as e:
        print(f"Error training C3 model: {e}")


# --- Model 3: FEA ---
def train_fea_model():
    """Trains the FEA model (Multi-output anisotropic material card)."""
    print("--- Training FEA Model ---")
    try:
        df = pd.read_csv(PATH_FEA_RAW)

        # This cleaning logic correctly handles parentheses and apostrophes
        def clean_col_name(col_name):
            new_col = col_name.strip()
            new_col = new_col.replace(' ', '_').replace(':', '').replace("'", "")
            # Replace ( and ) with _ and remove duplicates
            new_col = re.sub(r'\(', '_', new_col)
            new_col = re.sub(r'\)', '', new_col)
            new_col = new_col.replace('__', '_')
            # Remove trailing underscores
            if new_col.endswith('_'):
                new_col = new_col[:-1]
            return new_col

        df.columns = [clean_col_name(col) for col in df.columns]

        # Corrected feature names (no apostrophes)
        feature_cols = [
            "Material_Bonding_Perfection",
            "Material_Youngs_Modulus_GPa",
            "Material_Tensile_Yield_Strenght_MPa",
            "Material_Poissons_Ratio",
            "User_Infill_Pattern",
            "User_Infill_Density",
            "User_Line_Thickenss_mm",
            "User_Layer_Height_mm",
        ]

        # Define numeric and categorical features
        numeric_features = [
            "Material_Bonding_Perfection",
            "Material_Youngs_Modulus_GPa",
            "Material_Tensile_Yield_Strenght_MPa",
            "Material_Poissons_Ratio",
            "User_Infill_Density",
            "User_Line_Thickenss_mm",
            "User_Layer_Height_mm",
        ]
        categorical_features = ["User_Infill_Pattern"]

        # Define targets
        target_cols = [
            col
            for col in df.columns
            if col.startswith('Specimen_') and col != 'Specimen_Infill_Pattern'
        ]

        if not target_cols:
            print("Error: No 'Specimen_' target columns found in FEA data.")
            return

        # 1. Define all columns that MUST be numeric
        numeric_cols_to_clean = target_cols + numeric_features

        # 2. Force-convert all target AND numeric feature columns to numeric
        df[numeric_cols_to_clean] = df[numeric_cols_to_clean].apply(pd.to_numeric, errors='coerce')

        # 3. Drop any rows that now contain NaN values in these critical columns
        original_rows = len(df)
        # Also drop NaNs in categorical features
        df = df.dropna(subset=numeric_cols_to_clean + categorical_features)
        print(f"Dropped {original_rows - len(df)} rows with bad data from FEA dataset.")

        print(f"FEA model will predict {len(target_cols)} properties.")

        X = df[feature_cols]  # Now X is clean
        y = df[target_cols]  # y is clean

        # Define preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        # Use MultiOutputRegressor wrapping a RandomForestRegressor
        base_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        pipeline = Pipeline(
            steps=[('preprocessor', preprocessor), ('model', MultiOutputRegressor(base_regressor))]
        )

        print("Training FEA model (this may take a minute)...")
        pipeline.fit(X, y)

        # Save the model AND the target names
        joblib.dump(pipeline, MODEL_FEA_PATH)
        print(f"FEA model saved to {MODEL_FEA_PATH}")
        joblib.dump(target_cols, FEA_TARGETS_PATH)
        print(f"FEA target names saved to {FEA_TARGETS_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_FEA_RAW}")
    except Exception as e:
        print(f"Error training FEA model: {e}")


# --- Model 4: Fatigue ---
def train_fatigue_model():
    """Trains the Fatigue model (Cycles to Failure)."""
    print("--- Training Fatigue Model ---")
    try:
        # Corrected: Read header from row 0, skip row 1 (units)
        df = pd.read_csv(PATH_FATIGUE_RAW, header=0, skiprows=[1])

        # Clean column names
        df.columns = df.columns.str.replace(' ', '_', regex=False).str.replace('˚', '', regex=False)

        feature_cols = ['Nozzle_Diameter', 'Print_Speed', 'Nozzle_Temperature', 'Stress_Level']
        target_col = 'Fatigue_Lifetime'

        # Drop rows with NaN values in the relevant columns
        df = df.dropna(subset=feature_cols + [target_col])

        X = df[feature_cols]
        y = df[target_col]

        # Simple pipeline (all features are numeric)
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        print("Training Fatigue model...")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_FATIGUE_PATH)
        print(f"Fatigue model saved to {MODEL_FATIGUE_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_FATIGUE_RAW}")
    except Exception as e:
        print(f"Error training Fatigue model: {e}")


# --- Model 5: Dimensional Accuracy ---
def train_accuracy_model():
    """Trains the Dimensional Accuracy model (Deswal et al.)."""
    print("--- Training Accuracy Model ---")
    try:
        df = pd.read_csv(PATH_ACCURACY_RAW)

        # Define features and targets
        feature_cols = [
            "Layer_Thickness_mm",
            "Build_Orientation_deg",
            "Infill_Density_percent",
            "Number_of_Contours",
        ]
        target_cols = ["Var_Length_percent", "Var_Width_percent", "Var_Thickness_percent"]

        # Drop rows with NaN values in the relevant columns
        df = df.dropna(subset=feature_cols + target_cols)

        X = df[feature_cols]
        y = df[target_cols]

        # Simple pipeline (all features are numeric)
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        print("Training Accuracy model...")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_ACCURACY_PATH)
        print(f"Accuracy model saved to {MODEL_ACCURACY_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_ACCURACY_RAW}")
    except Exception as e:
        print(f"Error training Accuracy model: {e}")


# --- v6: Model 6: Warp Deformation ---
def train_warpage_model():
    """Trains the Warp Deformation model (Nazan et al.)."""
    print("--- Training Warpage Model ---")
    try:
        df = pd.read_csv(PATH_WARPAGE_RAW)

        # Define features and targets
        feature_cols = [
            "Layer_Temperature_C",
            "Infill_Density_percent",
            "First_Layer_Height_mm",
            "Other_Layer_Height_mm",
        ]
        target_col = "Warpage_mm"

        # Drop rows with NaN values in the relevant columns
        df = df.dropna(subset=feature_cols + [target_col])

        X = df[feature_cols]
        y = df[target_col]

        # Simple pipeline (all features are numeric)
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        print("Training Warpage model...")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_WARPAGE_PATH)
        print(f"Warpage model saved to {MODEL_WARPAGE_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_WARPAGE_RAW}")
    except Exception as e:
        print(f"Error training Warpage model: {e}")


# --- v7: Model 7: Hardness ---
def train_hardness_model():
    """Trains the Hardness model (Kadam et al.)."""
    print("--- Training Hardness Model ---")
    try:
        df = pd.read_csv(PATH_HARDNESS_RAW)

        # Define features and targets
        feature_cols = [
            "Layer_Thickness_mm",
            "Shell_Thickness_mm",
            "Fill_Density_percent",
            "Fill_Pattern",
        ]
        target_col = "Hardness_Shore_D"

        # Define numeric and categorical features
        numeric_features = ["Layer_Thickness_mm", "Shell_Thickness_mm", "Fill_Density_percent"]
        categorical_features = ["Fill_Pattern"]

        # Drop rows with NaN values
        df = df.dropna(subset=feature_cols + [target_col])

        X = df[feature_cols]
        y = df[target_col]

        # Define preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        # Create the full pipeline
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        # Train the model
        print("Training Hardness model...")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_HARDNESS_PATH)
        print(f"Hardness model saved to {MODEL_HARDNESS_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_HARDNESS_RAW}")
    except Exception as e:
        print(f"Error training Hardness model: {e}")


# --- v8: Model 8: Multi-Material Bond Strength ---
def train_multimaterial_model():
    """Trains the Multi-Material Bond Strength model (Yadav et al.)."""
    print("--- Training Multi-Material Model ---")
    try:
        df = pd.read_csv(PATH_MULTIMATERIAL_RAW)

        # Define features and targets
        feature_cols = [
            "Material_A",
            "Material_B",
            "Layer_Height_mm",
            "Extrusion_Temp_C",
            "Infill_Density_percent",
        ]
        target_col = "Tensile_Strength_MPa"

        # Define numeric and categorical features
        numeric_features = ["Layer_Height_mm", "Extrusion_Temp_C", "Infill_Density_percent"]
        categorical_features = ["Material_A", "Material_B"]

        # Drop rows with NaN values
        df = df.dropna(subset=feature_cols + [target_col])

        X = df[feature_cols]
        y = df[target_col]

        # Define preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        # Create the full pipeline
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        # Train the model
        print("Training Multi-Material model...")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_MULTIMATERIAL_PATH)
        print(f"Multi-Material model saved to {MODEL_MULTIMATERIAL_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_MULTIMATERIAL_RAW}")
    except Exception as e:
        print(f"Error training Multi-Material model: {e}")


# --- v9: Model 9: Composite Filaments (e.g., CF PLA/PETG) ---
def train_composite_model():
    """Trains the Composite Filament model (e.g., PETG/Carbon Fiber)."""
    print("--- Training Composite Model ---")
    try:
        df = pd.read_csv(PATH_COMPOSITE_RAW)

        # Define features and targets (based on Alarifi paper columns)
        feature_cols = [
            "Reinforcement_percent",
            "Infill_Pattern",
            "Infill_Density_percent",
            "Layer_Thickness_mm",
        ]
        target_cols = ["Tensile_Strength_MPa", "Elastic_Modulus_GPa"]

        # Define numeric and categorical features
        numeric_features = ["Reinforcement_percent", "Infill_Density_percent", "Layer_Thickness_mm"]
        categorical_features = ["Infill_Pattern"]

        # Drop rows with NaN values
        df = df.dropna(subset=feature_cols + target_cols)

        X = df[feature_cols]
        y = df[target_cols]

        # Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        # Use MultiOutputRegressor for two targets
        base_regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        pipeline = Pipeline(
            steps=[('preprocessor', preprocessor), ('model', MultiOutputRegressor(base_regressor))]
        )

        print("Training Composite model...")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_COMPOSITE_PATH)
        print(f"Composite model saved to {MODEL_COMPOSITE_PATH}")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {PATH_COMPOSITE_RAW}")
    except Exception as e:
        print(f"Error training Composite model: {e}")


# --- Main execution ---
def main():
    """Main function to run all processing and training."""

    # Kaggle Model
    if not os.path.exists(MODEL_KAGGLE_PATH):
        train_kaggle_model()
    else:
        print("--- Kaggle Model already trained. Skipping. ---")

    # C3 Model
    if not os.path.exists(MODEL_C3_PATH):
        if not os.path.exists(PATH_C3_PROCESSED):
            if not process_c3_data():
                print("C3 data processing failed. Skipping C3 model training.")
                return
        train_c3_model()
    else:
        print("--- C3 Model already trained. Skipping. ---")

    # FEA Model
    # Check for targets file as well
    if not os.path.exists(MODEL_FEA_PATH) or not os.path.exists(FEA_TARGETS_PATH):
        train_fea_model()
    else:
        print("--- FEA Model already trained. Skipping. ---")

    # Fatigue Model
    if not os.path.exists(MODEL_FATIGUE_PATH):
        train_fatigue_model()
    else:
        print("--- Fatigue Model already trained. Skipping. ---")

    # Accuracy Model
    if not os.path.exists(MODEL_ACCURACY_PATH):
        train_accuracy_model()
    else:
        print("--- Accuracy Model already trained. Skipping. ---")

    # Warpage Model
    if not os.path.exists(MODEL_WARPAGE_PATH):
        train_warpage_model()
    else:
        print("--- Warpage Model already trained. Skipping. ---")

    # Hardness Model
    if not os.path.exists(MODEL_HARDNESS_PATH):
        train_hardness_model()
    else:
        print("--- Hardness Model already trained. Skipping. ---")

    # v8: Multi-Material Model
    if not os.path.exists(MODEL_MULTIMATERIAL_PATH):
        train_multimaterial_model()
    else:
        print("--- Multi-Material Model already trained. Skipping. ---")

    # v9: Composite Model
    if not os.path.exists(MODEL_COMPOSITE_PATH):
        train_composite_model()
    else:
        print("--- Composite Model already trained. Skipping. ---")

    print("\nAll models checked/trained.")
    print("Starting FastAPI server...")

    # Start the Uvicorn server as a subprocess
    subprocess.run(["uvicorn", "main:app", "--reload"])


if __name__ == "__main__":
    main()
