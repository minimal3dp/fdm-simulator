"""Central configuration and path management for the FDM Simulator.

All code should import paths from here instead of hardcoding strings.
This improves portability (e.g., running from different working directories)
and makes future refactors (adding environment overrides or CLI flags) easier.
"""

from __future__ import annotations

from pathlib import Path

# Root of the project (directory containing this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Models directory
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Primary data/config files
MATERIALS_FILE: Path = PROJECT_ROOT / "materials.json"
FEA_TARGETS_PATH: Path = MODELS_DIR / "fea_target_names.joblib"

# Raw dataset paths (mirrors previous constants in run_all_training.py)
KAGGLE_RAW_PATH: Path = RAW_DIR / "data.csv"
C3_RAW_PATH: Path = RAW_DIR / "C3-RAW DATA.csv"
FEA_RAW_PATH: Path = RAW_DIR / "3D_Printing_Data.xlsx - Sheet1.csv"
FATIGUE_RAW_PATH: Path = RAW_DIR / "1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv"
ACCURACY_RAW_PATH: Path = RAW_DIR / "dimensional_accuracy_deswal.csv"
WARPAGE_RAW_PATH: Path = RAW_DIR / "warpage_data_nazan.csv"
HARDNESS_RAW_PATH: Path = RAW_DIR / "hardness_data_kadam.csv"
MULTIMATERIAL_RAW_PATH: Path = RAW_DIR / "multimaterial_bond_yadav.csv"
COMPOSITE_RAW_PATH: Path = RAW_DIR / "composite_data_alarifi.csv"

# Processed dataset paths
C3_PROCESSED_PATH: Path = PROCESSED_DIR / "c3_processed_data.csv"

# Model artifact paths
MODEL_KAGGLE_PATH: Path = MODELS_DIR / "model_kaggle.joblib"
MODEL_C3_PATH: Path = MODELS_DIR / "model_c3.joblib"
MODEL_FEA_PATH: Path = MODELS_DIR / "model_fea.joblib"
MODEL_FATIGUE_PATH: Path = MODELS_DIR / "model_fatigue.joblib"
MODEL_ACCURACY_PATH: Path = MODELS_DIR / "model_accuracy.joblib"
MODEL_WARPAGE_PATH: Path = MODELS_DIR / "model_warpage.joblib"
MODEL_HARDNESS_PATH: Path = MODELS_DIR / "model_hardness.joblib"
MODEL_MULTIMATERIAL_PATH: Path = MODELS_DIR / "model_multimaterial.joblib"
MODEL_COMPOSITE_PATH: Path = MODELS_DIR / "model_composite.joblib"


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    # Root
    "PROJECT_ROOT",
    # Dirs
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    # Files
    "MATERIALS_FILE",
    "FEA_TARGETS_PATH",
    # Raw paths
    "KAGGLE_RAW_PATH",
    "C3_RAW_PATH",
    "FEA_RAW_PATH",
    "FATIGUE_RAW_PATH",
    "ACCURACY_RAW_PATH",
    "WARPAGE_RAW_PATH",
    "HARDNESS_RAW_PATH",
    "MULTIMATERIAL_RAW_PATH",
    "COMPOSITE_RAW_PATH",
    # Processed paths
    "C3_PROCESSED_PATH",
    # Model paths
    "MODEL_KAGGLE_PATH",
    "MODEL_C3_PATH",
    "MODEL_FEA_PATH",
    "MODEL_FATIGUE_PATH",
    "MODEL_ACCURACY_PATH",
    "MODEL_WARPAGE_PATH",
    "MODEL_HARDNESS_PATH",
    "MODEL_MULTIMATERIAL_PATH",
    "MODEL_COMPOSITE_PATH",
    # Functions
    "ensure_directories",
]
