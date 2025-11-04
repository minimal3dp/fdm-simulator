# Copilot instructions for this repo (fdm-simulator)

This project is a local FastAPI backend that serves multiple ML models for FDM 3D printing, plus a single-file frontend (`fdm_simulator.html`) that calls the API directly. A training orchestrator (`run_all_training.py`) builds missing models from CSVs and then launches the server.

## Architecture in one glance
- Backend: `main.py` (FastAPI, CORS “*”) with endpoints: `/materials`, `/predict/{kaggle|c3|fea|fatigue|accuracy|warpage|hardness|multimaterial}`, and `/optimize`.
- Models: `models/*.joblib` loaded at startup; missing files cause 503 on that endpoint (do not crash the app).
- Data: `data/raw/*.csv` inputs; `data/processed/c3_processed_data.csv` generated; training uses scikit-learn pipelines + joblib.
- Frontend: `fdm_simulator.html` (no build step) fetches `http://127.0.0.1:8000` and renders results (Tailwind + a simple three.js STL viewer).
- Materials: `materials.json` powers dropdowns and cost/time via `get_material_density`.

## Developer workflows that matter
- One-shot bootstrap: run training + server
```bash
python run_all_training.py
```
  - Trains any missing models (Kaggle, C3, FEA, Fatigue, Accuracy, Warpage, Hardness, Multi-Material) then starts `uvicorn main:app --reload`.
- Direct server development:
```bash
uvicorn main:app --reload
```
- Frontend development: open `fdm_simulator.html` in a browser. API base is hardcoded as `API_BASE_URL = "http://127.0.0.1:8000"`.

## Project-specific patterns and conventions
- All prediction inputs extend `GlobalInputs` (part_mass_g, filament_cost_kg, material_name). Every response includes `estimated_cost_usd` and `estimated_print_time_min` computed by `calculate_cost_time`.
- Feature names must match training columns exactly. FEA training cleans headers and predicts targets with prefix `Specimen_`; names are saved in `fea_target_names.joblib` and zipped to results.
- Endpoints return 503 if model not loaded; `/materials` returns 404 if `materials.json` failed to load.
- Optimizer (`/optimize`) uses pymoo NSGA-II. Adding a model requires a param config (bounds + categoricals), mapping predicted keys to objective IDs, and ensuring integer variables are rounded.
- Frontend expects specific output keys per model (e.g., Kaggle: `tensile_strength`, `roughness`, `elongation`; Accuracy: `Var_*`; Warpage: `Warpage_mm`; Hardness: `Hardness_Shore_D`). Update both Pydantic outputs and UI update functions together.
- Materials DB schema:
  - Key = material name; `common` (nozzle_temperature, bed_temperature, print_speed, fan_speed), `fea` props, optional `fatigue`, `density_g_cm3`.

## Cross-component notes (how things connect)
- Frontend calls `/materials` on load to populate dropdowns; then debounced calls to `/predict/*` when sliders change; Optimizer tab posts to `/optimize` with `objectives[]`, `constraints[]`, and a `global_inputs` block.
- G-code parser (OrcaSlicer comments only) maps values into Kaggle/C3 sliders; it’s a best-effort regex parse.
- Cost/time depends on `material_name` density from `materials.json`; fallback densities exist for PLA/ABS/PETG.

## Minimal API example
- POST `/predict/kaggle`
  - Body keys: `GlobalInputs + { layer_height, wall_thickness, infill_density, infill_pattern, nozzle_temperature, bed_temperature, print_speed, material, fan_speed }`.
  - Returns: `{ tensile_strength, roughness, elongation, estimated_cost_usd, estimated_print_time_min }`.

## Adding a new model (short checklist)
1) Add train_* in `run_all_training.py` + path constants; save `models/<name>.joblib`.
2) Add Pydantic input/output and endpoint in `main.py`; load at startup; include cost/time mapping.
3) Wire frontend: add params panel, outputs, recs, update `simulationRunners` + UI updaters.
4) Optional: add to optimizer (bounds, categorical maps, objective IDs).
