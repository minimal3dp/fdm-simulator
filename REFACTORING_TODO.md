# FDM Simulator - Immediate Action Items & Refactoring Checklist

> **Created:** November 10, 2025
> **Purpose:** Track immediate refactoring tasks to streamline and condense the codebase
> **Related:** See `REFACTORING_ANALYSIS.md` for detailed analysis

---

## 🔥 THIS WEEK - Critical Deduplication

### ✅ Quick Wins (< 2 hours each)
- [ ] Extract magic numbers to `constants.py` (30 min)
- [ ] Replace `print()` with `logging` (1 hour)
- [ ] Add missing docstrings (2 hours)
- [ ] Move model registry to YAML (2 hours)

### 🎯 High-Impact (1-2 days each)

#### 1. Consolidate Training Functions ⭐⭐⭐
**File:** `run_all_training.py`
**Impact:** 540 lines → 150 lines (72% reduction)
**Status:** ⬜ Not Started

**Steps:**
- [ ] Create `MODEL_CONFIGS` dictionary
- [ ] Write generic `train_model(name, config)` function
- [ ] Replace `train_kaggle_model()`
- [ ] Replace `train_c3_model()`
- [ ] Replace `train_fea_model()`
- [ ] Replace `train_fatigue_model()`
- [ ] Replace `train_accuracy_model()`
- [ ] Replace `train_warpage_model()`
- [ ] Replace `train_hardness_model()`
- [ ] Replace `train_multimaterial_model()`
- [ ] Replace `train_composite_model()`
- [ ] Test all models train successfully
- [ ] Remove old functions

#### 2. Dynamic Pydantic Models ⭐⭐⭐
**File:** `main.py` lines 65-205
**Impact:** 140 lines → 50 lines (64% reduction)
**Status:** ⬜ Not Started

**Steps:**
- [ ] Create `MODEL_FIELDS` configuration dict
- [ ] Write `create_input_model()` factory function
- [ ] Write `create_output_model()` factory function
- [ ] Generate all Input models dynamically
- [ ] Generate all Output models dynamically
- [ ] Update MODEL_REGISTRY to use generated models
- [ ] Test all endpoints still work
- [ ] Remove old static model definitions

#### 3. Generic Frontend Updates ⭐⭐⭐
**File:** `fdm_simulator.html` lines 1836-2000
**Impact:** 350 lines → 100 lines (71% reduction)
**Status:** ⬜ Not Started

**Steps:**
- [ ] Create `UI_CONFIGS` object
- [ ] Write `updateModelUI(modelName, data)` function
- [ ] Write `updateModelRecs(modelName, inputs)` function
- [ ] Test with Kaggle model
- [ ] Test with C3 model
- [ ] Test with FEA model
- [ ] Test with Fatigue model
- [ ] Test with Accuracy model
- [ ] Test with Warpage model
- [ ] Test with Hardness model
- [ ] Test with MultiMaterial model
- [ ] Test with Composite model
- [ ] Remove old update functions

---

## 📦 NEXT WEEK - Module Extraction

### 4. Extract G-Code Parser ⭐⭐
**File:** `main.py` lines 1246-1543
**New Module:** `gcode_parser.py`
**Impact:** 297 lines out of main.py
**Status:** ⬜ Not Started

**Steps:**
- [ ] Create `gcode_parser.py`
- [ ] Create `GCodeParser` class
- [ ] Move parsing logic to class methods
- [ ] Create unit tests
- [ ] Update `analyze_gcode` endpoint
- [ ] Verify G-code analysis still works

### 5. Extract STL Analyzer ⭐⭐
**File:** `main.py` lines 1547-1777
**New Module:** `stl_analyzer.py`
**Impact:** 230 lines out of main.py
**Status:** ⬜ Not Started

**Steps:**
- [ ] Create `stl_analyzer.py`
- [ ] Create `STLAnalyzer` class
- [ ] Move analysis logic to class methods
- [ ] Create unit tests
- [ ] Update `analyze_stl` endpoint
- [ ] Verify STL analysis still works

### 6. Move Model Config to YAML ⭐
**File:** `main.py` lines 223-480
**New File:** `models_config.yaml`
**Impact:** 250 lines → 50 lines + config
**Status:** ⬜ Not Started

**Steps:**
- [ ] Create `models_config.yaml`
- [ ] Define schema for model config
- [ ] Convert MODEL_REGISTRY to YAML
- [ ] Add YAML loading function
- [ ] Add validation
- [ ] Test all models load correctly

---

## 🔧 ONGOING - Code Quality

### Error Handling ⭐
- [ ] Create `errors.py` with custom exceptions
- [ ] Add `FDMSimulatorError` base class
- [ ] Create `ModelNotLoadedError`
- [ ] Create `PredictionError`
- [ ] Add exception handlers to FastAPI
- [ ] Replace generic exceptions throughout

### Constants ⭐
- [ ] Create `constants.py`
- [ ] Extract `EXTRUSION_WIDTH_MM = 0.4`
- [ ] Extract `DEFAULT_PRINT_SPEED_MM_S = 60.0`
- [ ] Extract `MAX_FILE_SIZE_MB = 15`
- [ ] Extract `DEFAULT_DENSITY_G_CM3 = 1.25`
- [ ] Extract optimization parameters
- [ ] Update code to use constants

### Logging ⭐
- [ ] Import logging module
- [ ] Replace all `print()` statements
- [ ] Add structured logging
- [ ] Configure log levels
- [ ] Add file output

---

## 📊 METRICS & TRACKING

### Current State
- **main.py:** 1,777 lines
- **run_all_training.py:** 682 lines
- **fdm_simulator.html:** 3,416 lines
- **Total:** 5,875 lines

### After Phase 1 (Target)
- **main.py:** ~1,100 lines (-677)
- **run_all_training.py:** ~150 lines (-532)
- **fdm_simulator.html:** ~2,500 lines (-916)
- **New modules:** +800 lines
- **Net Total:** ~4,550 lines (-1,325 or 23%)

### Progress Tracker
```
[▱▱▱▱▱▱▱▱▱▱] 0/23 items complete (0%)

Critical Items: [▱▱▱▱▱▱▱] 0/7 (0%)
High Priority: [▱▱▱▱▱▱] 0/6 (0%)
Medium Priority: [▱▱▱▱▱▱] 0/6 (0%)
Low Priority: [▱▱▱▱] 0/4 (0%)
```

---

## ✅ COMPLETION CHECKLIST

### Before Merging Any Refactor
- [ ] All existing tests pass
- [ ] New tests added where applicable
- [ ] Documentation updated
- [ ] No breaking API changes (or documented)
- [ ] Code reviewed
- [ ] Linting passes
- [ ] Type checking passes

### Phase 1 Complete When
- [ ] Items 1-4 implemented
- [ ] ~1,078 lines reduced
- [ ] All 9 models still train
- [ ] All 9 predictions still work
- [ ] Frontend still functions correctly
- [ ] Performance not degraded

---

## 🚫 DO NOT CHANGE

### Protected During Refactoring
- ✋ API endpoint signatures
- ✋ Model training algorithms
- ✋ materials.json format
- ✋ CSV data structures
- ✋ UI layout and behavior
- ✋ Prediction outputs format

---

## 📝 NOTES

### Decisions Made
- **2025-11-10:** Keep existing endpoints for backward compatibility
- **2025-11-10:** Gradual migration, test each change independently
- **2025-11-10:** Extract to modules before making internal changes

### Open Questions
- Should we consolidate endpoints to single `/predict/{model}` or keep separate?
- YAML or JSON for model config?
- Keep tests in parallel or refactor together?

---

## 🔄 DAILY STANDUP TEMPLATE

### Today I Will
- [ ] Work on Item #___: _____________________
- [ ] Expected outcome: ______________________

### Blockers
- [ ] None / List any blockers

### Notes
- ___________________________________________

---

**Next Review:** End of week
**Owner:** Development Team
**Priority:** 🔴 High - Application complexity is impacting maintainability
