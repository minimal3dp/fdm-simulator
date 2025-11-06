FDM 3D Print Property Simulator (v11)

This application is a multi-model simulator and optimizer for Fused Deposition Modeling (FDM) 3D printing. It uses a Python (FastAPI) backend to run several machine learning models trained on published research data. The frontend is a single, static HTML file that provides a UI for simulating properties and optimizing parameters.

Recent updates:
- **v11: NEW - STL Mesh Quality Analysis & G-Code Deviation Detection** (Based on Montalti et al. 2024)
  - Analyze STL mesh quality before slicing
  - Detect non-manifold edges, holes, poor tessellation
  - Compare STL vs G-code geometry to quantify slicing errors
  - Calculate Hausdorff distance for dimensional accuracy
  - Get recommendations for mesh repair and export settings
- v10: Enhanced G-Code analysis with robust parsing (M82/M83, G20/G21, G92 support)
- v9: New Composite Filament model (e.g., Carbon/Glass Fiber reinforced PLA/PETG)
- v8: New Multi-Material Bond model (ABS+PETG)
- v7: New Hardness model
- v6: New Warp Deformation model

Project Structure

Your project must be organized in this exact structure for the scripts to work:

fdm-simulator/
│
├── .venv/                   # Your UV virtual environment
│
├── data/
│   ├── raw/                 # <-- PLACE ALL RAW CSVs HERE
│   │   ├── data.csv         # (Kaggle dataset)
│   │   ├── C3-RAW DATA.csv  # (C3 dataset)
│   │   ├── 3D_Printing_Data.xlsx - Sheet1.csv  # (FEA dataset)
│   │   ├── 1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv # (Fatigue dataset)
│   │   ├── dimensional_accuracy_deswal.csv # (Accuracy dataset)
│   │   ├── warpage_data_nazan.csv          # (v6: Warpage dataset)
│   │   ├── hardness_data_kadam.csv         # (v7: Hardness dataset)
│   │   ├── multimaterial_bond_yadav.csv    # (v8: Multi-Material dataset)
│   │   └── composite_data_alarifi.csv      # (v9: Composite filament dataset)
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
│   ├── model_warpage.joblib   # (v6)
│   ├── model_hardness.joblib  # (v7)
│   ├── model_multimaterial.joblib # (v8)
│   └── model_composite.joblib # (v9)
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

Place all raw .csv data files into the data/raw/ folder, as shown in the project structure diagram.

Step 3: Run the Application

This application is now a single-command process.

While in your activated environment, run the main training script:

python run_all_training.py


What this script does:

It will "smartly" check for all ML models in the models/ folder.

If any models are missing, it will automatically process the raw data and train them, saving the .joblib files.

Once all models are verified, it will automatically launch the FastAPI server.

You will see the output:

All models checked/trained.
Starting FastAPI server...
INFO:     Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000) (Press CTRL+C to quit)


Do not close this terminal. The server is now running.

To use the app: Open the fdm_simulator.html file directly in your web browser (e.g., by double-clicking it).

Application Features (v11)

1. Simulator Tab

A "what-if" tool to see how parameters affect outcomes. It now contains 9 models:

Kaggle Model: Predicts Tensile Strength, Roughness, and Elongation.

Warp Deformation: Predicts Warpage (mm) based on the Nazan et al. dataset.

Dimensional Accuracy: Predicts % Variation in Length, Width, and Thickness.

Fatigue Lifetime: Predicts how many Cycles a part can withstand.

C3 Model: Predicts Tensile Strength and Elongation.

FEA Material Card: Predicts the full anisotropic material card for professional FEA software.

Hardness: Predicts Shore D hardness.

Multi-Material Bond: Predicts ABS+PETG interface tensile strength.

Composite Filaments: Predicts tensile strength (MPa) and elastic modulus (GPa) for reinforced filaments.

2. Optimizer Tab

A "what's-best" tool to find the optimal settings for your goals.

Select a model (e.g., "Kaggle", "Dimensional Accuracy", or "Warp Deformation").

Select your objectives (e.g., "Minimize: Warpage (mm)" and "Minimize: Print Time").

The app runs a genetic algorithm (NSGA-II) to find the "Top 5" best trade-off solutions.

3. G-Code Analysis Tab (v10+)

Upload your sliced G-code file to get:
- Print time and material usage estimates
- Layer-by-layer visualization with feature coloring
- Feature breakdown (walls, infill, supports, etc.)
- Optimization suggestions based on toolpath analysis
- Interactive 3D viewer with per-layer scrubber

**NEW in v11: STL Mesh Quality Analysis**
- Upload STL files to analyze mesh quality before slicing
- Detect issues: non-watertight, non-manifold geometry, poor triangulation
- Calculate mesh statistics: vertices, faces, volume, surface area
- Triangle quality metrics (aspect ratio, edge length variance)
- Optional: Compare STL vs G-code to measure slicing deviation
  - Hausdorff distance calculation (max deviation)
  - Mean deviation and overall accuracy percentage
  - Identify error-prone regions
- Get actionable recommendations:
  - Re-export with better tessellation settings
  - Repair mesh using recommended tools
  - Adjust slicer resolution settings

**Research Foundation:**
Based on Montalti et al. (2024) "From CAD to G-code: Strategies to minimizing errors in 3D printing process" 
(CIRP Journal of Manufacturing Science and Technology)

4. Global Settings

Filament Used (g): Enter the part weight from your slicer.

Filament Cost ($/kg): Enter your spool cost.

All simulation and optimization results will automatically include Estimated Cost and Estimated Print Time.