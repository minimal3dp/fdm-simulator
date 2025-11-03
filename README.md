# FDM 3D Print Property Simulator (v4)

This is a multi-model FDM 3D print simulator powered by a Python (FastAPI) backend and a pure HTML/JS frontend. It uses machine learning models trained on published research data to predict mechanical properties, fatigue life, and more, based on your slicer settings.

Version 4 now includes a new "Optimizer" tab!

## New in v4: Optimizer

A new "Optimizer" tab is available. This feature, based on the NSGA-II genetic algorithm, allows you to define your goals (e.g., "Maximize Strength," "Minimize Cost") and will run thousands of simulations to find the Top 5 recommended settings that best balance your objectives.

Note: The Optimizer currently supports the "Kaggle Model."

## Project Structure

```bash
fdm-simulator/
│
├── .venv/                   # Your UV virtual environment
│
├── data/
│   ├── raw/                 # <-- PLACE ALL 4 RAW CSVs HERE
│   │   ├── data.csv
│   │   ├── C3-RAW DATA.csv
│   │   ├── 3D_Printing_Data.xlsx - Sheet1.csv
│   │   └── 1-s2.0-S2352340922000580-mmc1.xlsx - Data.csv
│   │
│   └── processed/           # (This folder will be created)
│       └── c3_processed_data.csv
│
├── models/                  # (This folder will be created)
│   ├── model_kaggle.joblib
│   ├── model_c3.joblib
│   ├── model_fea.joblib
│   ├── fea_target_names.joblib
│   └── model_fatigue.joblib
│
├── pyproject.toml           # Python dependencies for UV (Updated)
├── materials.json           # Material datasheet
├── run_all_training.py      # RUN THIS (trains models & starts server)
├── main.py                  # The Python backend server (Updated)
└── fdm_simulator.html       # The frontend (Updated)
```

##Setup & First Run

Requires Python 3.8+ and uv (or pip).

Place Your Data:

Download all 4 raw datasets.

Place them inside the data/raw/ folder.

Create/Update Environment (using UV):

 **Create a new virtual environment (if you haven't)**
```bash
uv venv
```

 **Activate the environment**
```bash
source .venv/bin/activate
```

**Install/Update Dependencies:**

(If this is your first time)

# This reads pyproject.toml and installs all packages
```bash
uv pip install .
```

(If you are updating from v3)

**This will read the updated pyproject.toml and install pymoo**
```bash
uv pip install .
```

### Train Models & Run Server:

This single command will process all data, train all 4 models (if they don't already exist), and then start the FastAPI server.

```bash
python run_all_training.py
```

The server will be running at http://127.0.0.1:8000.

Use the Application:

Keep the server terminal running.

Open the fdm_simulator.html file in your web browser (e.g., by double-clicking it). The app will connect to your local server automatically.

How to Use

Simulator Tab

Global Settings:

Filament Used (g): Enter the estimated filament mass from your slicer.

Filament Cost ($/kg): Enter the cost of your 1kg spool.

Select Model: Choose one of the four simulation models from the dropdown.

Adjust Parameters:

Use the G-Code parser to auto-fill settings (for Kaggle & C3 models).

Use the "Material Datasheet" dropdowns to auto-fill settings (for Kaggle, FEA, & Fatigue models).

Manually adjust the sliders to see how parameters affect the predictions.

Review Predictions:

The "Predicted Properties" card will update in real-time.

This now includes Est. Material Cost and Est. Print Time.

Optimizer Tab (New!)

Select Model: Choose the model you want to optimize for (currently supports "Kaggle Model").

Set Global Settings: Enter your "Filament Used (g)" and "Filament Cost ($/kg)". These are used if you optimize for cost or time.

Define Objectives:

Select a property (e.g., "Tensile Strength") and a goal ("Maximize" or "Minimize").

You can add multiple objectives (e.g., "Maximize Tensile Strength" and "Minimize Cost").

Define Constraints (Optional):

Set limits on your parameters (e.g., "Print Speed" must be "<= 100" mm/s).

Run Optimizer:

Click the "Run Optimizer" button. The backend server will run a genetic algorithm, which may take 20-30 seconds.

Review Results:

The "Optimization Results" table will show the Top 5 recommended parameter sets that provide the best trade-off for your defined goals.