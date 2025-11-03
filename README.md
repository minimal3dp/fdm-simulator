# FDM 3D Print Property Simulator

This is a multi-model application that uses machine learning to predict the mechanical properties of FDM 3D-printed parts. It consists of a Python (FastAPI) backend that serves the ML models and an HTML/JavaScript frontend that provides the user interface.

The application allows you to simulate properties based on four different research datasets:

Kaggle Dataset: Predicts Tensile Strength, Roughness, and Elongation.

C3 Dataset (Wang et al.): Predicts Tensile Strength and Elongation from raw stress-strain data.

FEA Dataset (Lee & Tucker): Predicts a full anisotropic material card (100+ properties) for use in FEA software.

Fatigue Dataset (Azadi & Dadashi): Predicts the Fatigue Lifetime (cycles to failure) of a part under load.

## Project Structure

Your project should be organized as follows:

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
│   └── processed/           # (This folder will be created by the script)
│       └── c3_processed_data.csv
│
├── models/                  # (This folder will be created by the script)
│   ├── model_kaggle.joblib
│   ├── model_c3.joblib
│   ├── model_fea.joblib
│   └── model_fatigue.joblib
│
├── pyproject.toml         # Python dependencies
├── README.md                # These are your project files
├── run_all_training.py      # RUN THIS
├── main.py                  # The Python backend server
└── fdm_simulator.html       # The frontend (open in browser)
│
└── pyproject.toml         # Python dependencies
```

## Quickstart Instructions

Follow these steps precisely to get the application running.

### Step 1: Environment Setup (with UV)

Create Environment:
Open your terminal in this project folder and create a new virtual environment:

```bash
uv venv
```

Activate Environment:

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Windows (CMD):

```cmd
.venv\Scripts\activate.bat
```

Your terminal prompt should now show (.venv).

Install Dependencies:
Install all required Python packages:

```bash
uv pip install -r requirements.txt
```

### Step 2: Download Datasets

Create the data/raw directory in your project folder.

Download data.csv from Kaggle and place it inside the data/raw/ folder.

Place your C3-RAW DATA.csv file inside the data/raw/ folder.

### Step 3: Process Data and Train Models

In your terminal (with the environment active), run the unified training script. This script will:

Create the data/processed/ and models/ directories.

Process the C3 raw data and save the result.

Train both the Kaggle and C3 models and save them.

Automatically start the server for you.

```bash
python run_all_training.py
```

Keep this terminal running. It will start the uvicorn server automatically once training is complete.

### Step 4: Open the Frontend

Open the fdm_simulator.html file directly in your web browser (e.g., by double-clicking it).

You can now use the application! It will automatically connect to your local server.