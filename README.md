DM 3D Print Property Simulator (v5)

This application is a multi-model simulator and optimizer for Fused Deposition Modeling (FDM) 3D printing. It uses several machine learning models trained on published research data to predict the mechanical properties, dimensional accuracy, and cost/time of a print before you print it.

It also features a "v4" multi-objective genetic algorithm optimizer to help you find the best print settings for your specific goals.

Project Structure

Your project must be organized in this exact folder structure for the scripts to work.

fdm-simulator/
│
├── .venv/                   # Your UV virtual environment
│
├── data/
│   ├── raw/                 # <-- PLACE ALL 5 RAW CSVs HERE
│   │   ├── data.csv         # (Kaggle dataset)
│   │   ├── C3-RAW DATA.csv  # (Wang et al. dataset)
│   │   ├── 3D_Printing_Data.xlsx - Sheet1.csv  # (Lee & Tucker FEA dataset)
│   │   ├── 1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv # (Azadi et al. Fatigue dataset)
│   │   └── dimensional_accuracy_deswal.csv # (v5: Deswal et al. Accuracy dataset)
│   │
│   └── processed/           # (This folder is created by the script)
│       └── c3_processed_data.csv
│
├── models/                  # (This folder is created by the script)
│   ├── model_kaggle.joblib
│   ├── model_c3.joblib
│   ├── model_fea.joblib
│   ├── fea_target_names.joblib
│   ├── model_fatigue.joblib
│   └── model_accuracy.joblib # (v5: New model)
│
├── pyproject.toml           # Python dependencies
├── run_all_training.py      # RUN THIS FIRST
├── main.py                  # The Python backend server
├── fdm_simulator.html       # The frontend (open this in your browser)
└── materials.json           # Filament datasheet properties


Quick Start: How to Run

Prerequisite: You must have uv installed (pip install uv).

Step 1: Set Up the Environment

Open your terminal in this project's root folder (fdm-simulator/).

Create a new virtual environment:

uv venv


Activate the environment:

macOS/Linux: source .venv/bin/activate

Windows: .venv\Scripts\activate

Install all Python dependencies from the pyproject.toml file:

uv pip install .


Step 2: Place Raw Data

Download all 5 raw datasets and place them inside the data/raw/ folder, ensuring their filenames match the project structure exactly.

Step 3: Train Models & Run Server

Run the unified training script. This script will automatically:

Check for existing models.

Train any models that are missing (this may take a few minutes the first time).

Once all models are trained, it will automatically start the backend API server.

python run_all_training.py


Keep this terminal running. It is now serving your models at http://127.0.0.1:8000.

Step 4: Use the Application

Go to the project folder and double-click the fdm_simulator.html file to open it in your web browser.

The application (running from a file://... address) will connect to your server (running at http://127.0.0.1:8000) and will be fully functional.

Application Features (v5)

1. Global Print Settings

These inputs are used by all models to calculate cost and time:

Filament Used (g): Get this value from your slicer's preview.

Filament Cost ($/kg): The price you paid for a 1kg spool.

2. "Simulator" Tab

This tab allows you to run "what-if" scenarios for five different engineering models.

Kaggle Model: A general-purpose model for PLA/ABS predicting Tensile Strength, Roughness, and Elongation.

C3 Model: A specialized model for PLA predicting Tensile Strength and Elongation based on the (Wang et al.) dataset.

FEA Material Card: An advanced model for engineers based on the (Lee & Tucker) dataset. It predicts the full anisotropic material card (100+ properties) needed for professional FEA software.

Fatigue Lifetime Model: A durability model for PLA based on the (Azadi et al.) dataset. It predicts how many Cycles to Failure a part can withstand at a given stress level.

Dimensional Accuracy Model (v5): A new model for PLA based on the (Deswal et al.) dataset. It predicts the % Variation in Length, Width, and Thickness, allowing you to optimize for dimensional precision.

3. "Optimizer" Tab

This is a powerful "v4" feature that runs a genetic algorithm (NSGA-II) on the backend to find the best possible settings for your goals.

Select Model: Choose either the "Kaggle" or "Dimensional Accuracy" model (others are not yet supported).

Define Objectives: Tell the app what you want to achieve (e.g., "Maximize: Tensile Strength" and "Minimize: Estimated Print Time").

Define Constraints: Set hard limits (e.g., "Infill Density must be <= 50%").

Run Optimization: The app runs thousands of simulations on the server and returns the Top 5 "best trade-off" solutions (the Pareto front) that balance your goals.