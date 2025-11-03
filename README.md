FDM 3D Print Property Simulator (v6)

This application is a multi-model simulator and optimizer for Fused Deposition Modeling (FDM) 3D printing. It uses a Python (FastAPI) backend to run several machine learning models trained on published research data. The frontend is a single, static HTML file that provides a UI for simulating properties and optimizing parameters.

This "v6" update adds a new model for Warp Deformation.

Project Structure

Your project must be organized in this exact structure for the scripts to work:

fdm-simulator/
│
├── .venv/                   # Your UV virtual environment
│
├── data/
│   ├── raw/                 # <-- PLACE ALL 6 RAW CSVs HERE
│   │   ├── data.csv         # (Kaggle dataset)
│   │   ├── C3-RAW DATA.csv  # (C3 dataset)
│   │   ├── 3D_Printing_Data.xlsx - Sheet1.csv  # (FEA dataset)
│   │   ├── 1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv # (Fatigue dataset)
│   │   ├── dimensional_accuracy_deswal.csv # (Accuracy dataset)
│   │   └── warpage_data_nazan.csv          # (v6: Warpage dataset)
│   │
│   └── processed/           # (This folder will be created)
│       └── c3_processed_data.csv
│
├── models/                  # (This folder will be created)
│   ├── model_kaggle.joblib
│   ├── model_c3.joblib
│   ├── model_fea.joblib
│   ├── fea_target_names.joblib
│   ├── model_fatigue.joblib
│   ├── model_accuracy.joblib
│   └── model_warpage.joblib   # (v6: New model)
│
├── run_all_training.py      # RUN THIS ONCE
├── main.py                  # The Python backend server
├── fdm_simulator.html       # The frontend (open in browser)
├── pyproject.toml           # Python dependencies
└── materials.json           # Filament datasheet


Setup & Execution (3 Steps)

Step 1: Create Environment & Install Dependencies

Open your terminal in the fdm-simulator/ root folder.

Create a new virtual environment using uv:

uv venv


Activate the environment:

macOS/Linux: source .venv/bin/activate

Windows (PowerShell): .venv\Scripts\Activate.ps1

Install all Python dependencies from the pyproject.toml file:

uv pip install -e .


(This command installs pymoo for optimization, fastapi for the server, and scikit-learn/pandas for the models).

Step 2: Place Raw Data

Place all 6 of your raw .csv data files into the data/raw/ folder, as shown in the project structure diagram.

Step 3: Run the Application

This application is now a single-command process.

While in your activated environment, run the main training script:

python run_all_training.py


What this script does:

It will "smartly" check for all 6 ML models in the models/ folder.

If any models are missing, it will automatically process the raw data and train them, saving the .joblib files.

Once all models are verified, it will automatically launch the FastAPI server.

You will see the output:

All models checked/trained.
Starting FastAPI server...
INFO:     Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000) (Press CTRL+C to quit)


Do not close this terminal. The server is now running.

To use the app: Open the fdm_simulator.html file directly in your web browser (e.g., by double-clicking it).

Application Features (v6)

1. Simulator Tab

A "what-if" tool to see how parameters affect outcomes. It now contains 6 models:

Kaggle Model: Predicts Tensile Strength, Roughness, and Elongation.

Warp Deformation (New): Predicts Warpage (mm) based on the Nazan et al. dataset.

Dimensional Accuracy: Predicts % Variation in Length, Width, and Thickness.

Fatigue Lifetime: Predicts how many Cycles a part can withstand.

C3 Model: Predicts Tensile Strength and Elongation.

FEA Material Card: Predicts the full anisotropic material card for professional FEA software.

2. Optimizer Tab

A "what's-best" tool to find the optimal settings for your goals.

Select a model (e.g., "Kaggle", "Dimensional Accuracy", or "Warp Deformation").

Select your objectives (e.g., "Minimize: Warpage (mm)" and "Minimize: Print Time").

The app runs a genetic algorithm (NSGA-II) to find the "Top 5" best trade-off solutions.

3. Global Settings

Filament Used (g): Enter the part weight from your slicer.

Filament Cost ($/kg): Enter your spool cost.

All simulation and optimization results will automatically include Estimated Cost and Estimated Print Time.