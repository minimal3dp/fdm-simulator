# FDM Simulator - Comprehensive Refactoring Analysis & Recommendations

**Date:** November 10, 2025
**Current State:** 1,777 lines (main.py) + 3,416 lines (fdm_simulator.html) + 682 lines (run_all_training.py)
**Total Core Codebase:** ~5,875 lines

---

## Executive Summary

The FDM Simulator has grown significantly in complexity with 9 ML models, multiple analysis features, and extensive UI. This analysis identifies **23 high-priority recommendations** to streamline, condense, and improve maintainability while preserving functionality.

**Critical Metrics:**
- **16 API endpoints** (3 GET, 13 POST)
- **18 Pydantic models** (9 Input + 9 Output classes)
- **9 ML model training functions** with ~95% code duplication
- **18 JavaScript update functions** in frontend
- **Estimated reduction potential:** 30-40% code reduction possible

---

## 🔴 CRITICAL ISSUES (Immediate Action Required)

### 1. **Massive Code Duplication in Training Functions**
**Location:** `run_all_training.py` lines 62-605
**Issue:** 9 nearly identical training functions (~60 lines each) with only CSV path and column names differing

**Current Pattern:**
```python
def train_kaggle_model():
    if MODEL_KAGGLE_PATH.exists():
        print("--- Kaggle Model already trained. Skipping. ---")
        return
    try:
        df = pd.read_csv(KAGGLE_DATA)
        # ... identical preprocessing ...
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        # ... identical training ...
        joblib.dump(pipeline, MODEL_KAGGLE_PATH)
```

**Recommendation:**
```python
# Create generic training function with config
MODEL_CONFIGS = {
    'kaggle': {
        'path': MODEL_KAGGLE_PATH,
        'data': KAGGLE_DATA,
        'features': ['layer_height', 'wall_thickness', ...],
        'targets': ['tensile_strength', 'roughness', 'elongation'],
        'multi_output': True
    },
    # ... 8 more configs
}

def train_model(model_name: str, config: dict):
    """Generic training function for all models."""
    if config['path'].exists():
        print(f"--- {model_name.title()} Model already trained. Skipping. ---")
        return

    df = pd.read_csv(config['data'])
    X = df[config['features']]
    y = df[config['targets']]

    if config.get('multi_output'):
        model = MultiOutputRegressor(RandomForestRegressor(...))
    else:
        model = RandomForestRegressor(...)

    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipeline.fit(X, y)
    joblib.dump(pipeline, config['path'])
```

**Impact:** Reduces 540 lines to ~150 lines (72% reduction)

---

### 2. **Redundant Pydantic Model Boilerplate**
**Location:** `main.py` lines 65-205
**Issue:** 18 separate Input/Output classes with minimal differences

**Current:**
```python
class KaggleInput(GlobalInputs):
    layer_height: float
    wall_thickness: int
    # ... 7 more fields

class C3Input(GlobalInputs):
    Temperature: float
    Speed: float
    # ... 4 more fields

# ... 7 more nearly identical classes
```

**Recommendation:**
```python
# Use dynamic model generation
def create_input_model(name: str, fields: dict):
    """Factory function to create Input models."""
    return create_model(
        f'{name}Input',
        __base__=GlobalInputs,
        **fields
    )

# Define in config
MODEL_FIELDS = {
    'kaggle': {
        'layer_height': (float, ...),
        'wall_thickness': (int, ...),
        # ...
    },
    'c3': {
        'Temperature': (float, ...),
        # ...
    }
}

# Generate models
KaggleInput = create_input_model('Kaggle', MODEL_FIELDS['kaggle'])
```

**Impact:** Reduces 140 lines to ~50 lines (64% reduction)

---

### 3. **8 Nearly Identical Prediction Endpoints**
**Location:** `main.py` lines 747-795
**Issue:** Duplicate endpoint definitions calling same internal function

**Current:**
```python
@app.post("/predict/kaggle", response_model=KaggleOutput)
async def predict_kaggle(inputs: KaggleInput):
    return await _run_prediction("kaggle", inputs)

@app.post("/predict/c3", response_model=C3Output)
async def predict_c3(inputs: C3Input):
    return await _run_prediction("c3", inputs)

# ... 6 more identical endpoints
```

**Recommendation:**
```python
# Single generic endpoint with model selection
@app.post("/predict/{model_name}")
async def predict(model_name: str, inputs: dict):
    """Universal prediction endpoint for all models."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    config = MODEL_REGISTRY[model_name]
    InputModel = config['input_model']
    validated_inputs = InputModel(**inputs)

    return await _run_prediction(model_name, validated_inputs)
```

**Impact:** Reduces 48 lines to 12 lines (75% reduction)
**Alternative:** Keep separate endpoints but use decorator factory to reduce boilerplate

---

## 🟠 HIGH PRIORITY (Significant Improvements)

### 4. **Frontend: Repetitive Update Functions**
**Location:** `fdm_simulator.html` lines 1836-2000
**Issue:** 18 separate update functions with 90% identical code

**Current Pattern:**
```javascript
function updateKaggleUI(data) {
    document.getElementById('kaggle_tensile').textContent = data.tensile_strength.toFixed(2);
    document.getElementById('kaggle_roughness').textContent = data.roughness.toFixed(2);
    // ...
}

function updateC3UI(data) {
    document.getElementById('c3_elongation').textContent = data.elongation.toFixed(2);
    // ...
}
// ... 16 more functions
```

**Recommendation:**
```javascript
// Generic update function with config
const UI_CONFIGS = {
    kaggle: {
        outputs: [
            { key: 'tensile_strength', id: 'kaggle_tensile', format: 2 },
            { key: 'roughness', id: 'kaggle_roughness', format: 2 },
            // ...
        ]
    },
    // ... other models
};

function updateModelUI(modelName, data) {
    const config = UI_CONFIGS[modelName];
    config.outputs.forEach(({key, id, format}) => {
        const value = data[key];
        if (value !== undefined) {
            document.getElementById(id).textContent =
                typeof value === 'number' ? value.toFixed(format) : value;
        }
    });
}

// Usage: updateModelUI('kaggle', data);
```

**Impact:** Reduces ~350 lines to ~100 lines (71% reduction)

---

### 5. **Model Registry Over-Engineering**
**Location:** `main.py` lines 223-480
**Issue:** 250+ lines of repetitive configuration that could be data-driven

**Current:** Massive nested dictionary with duplicate structure
**Recommendation:**
- Extract to external JSON/YAML config file
- Use dataclasses for type safety
- Auto-generate optimizer configs from model schemas

```python
# models_config.yaml
kaggle:
  path: models/model_kaggle.joblib
  features:
    - layer_height: {type: float, min: 0.02, max: 0.3}
    - wall_thickness: {type: int, min: 1, max: 10}
    # ...
  outputs: [tensile_strength, roughness, elongation]

# Load and validate
@dataclass
class ModelConfig:
    path: Path
    features: dict
    outputs: list

def load_model_configs() -> dict[str, ModelConfig]:
    with open('models_config.yaml') as f:
        configs = yaml.safe_load(f)
    return {name: ModelConfig(**cfg) for name, cfg in configs.items()}
```

**Impact:** Reduces 250 lines to 50 lines Python + external config (80% reduction in main.py)

---

### 6. **G-Code Parser Complexity**
**Location:** `main.py` lines 1246-1543
**Issue:** 297 lines of parsing logic embedded in API endpoint

**Recommendation:**
- Extract to separate `gcode_parser.py` module
- Create `GCodeParser` class with clean interface
- Add unit tests (currently missing)

```python
# gcode_parser.py
class GCodeParser:
    def __init__(self, gcode_text: str):
        self.lines = gcode_text.split('\n')
        self.metadata = {}
        self.toolpath = []

    def parse(self) -> GCodeAnalysis:
        self._parse_metadata()
        self._parse_toolpath()
        return self._build_result()

# main.py (simplified)
@app.post("/analyze_gcode")
async def analyze_gcode(file: UploadFile):
    contents = await file.read()
    parser = GCodeParser(contents.decode('utf-8'))
    return parser.parse()
```

**Impact:** Improves testability, readability, maintainability

---

### 7. **STL Analysis Complexity**
**Location:** `main.py` lines 1547-1777
**Issue:** 230 lines of mesh analysis embedded in endpoint

**Recommendation:**
- Extract to `stl_analyzer.py` module
- Separate concerns: mesh quality, geometry analysis, comparison
- Add caching for expensive operations

```python
# stl_analyzer.py
class STLAnalyzer:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self._cache = {}

    def analyze_quality(self) -> MeshQuality:
        # Extract quality analysis logic
        pass

    def analyze_geometry(self) -> GeometryStats:
        # Extract geometry logic
        pass
```

**Impact:** Reduces main.py by 230 lines

---

## 🟡 MEDIUM PRIORITY (Code Quality)

### 8. **Inconsistent Error Handling**
**Issue:** Mix of bare exceptions, generic catches, and inconsistent error messages

**Recommendation:**
```python
# errors.py
class FDMSimulatorError(Exception):
    """Base exception for FDM Simulator."""
    pass

class ModelNotLoadedError(FDMSimulatorError):
    """Raised when attempting to use unloaded model."""
    pass

class PredictionError(FDMSimulatorError):
    """Raised when prediction fails."""
    pass

# Consistent handling
@app.exception_handler(FDMSimulatorError)
async def fdm_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": exc.__class__.__name__, "detail": str(exc)}
    )
```

---

### 9. **Cost/Time Calculation Duplication**
**Location:** Multiple places calculate cost/time with slightly different logic

**Recommendation:**
- Create `CostTimeCalculator` class
- Centralize all cost/time logic
- Add configurable pricing models

---

### 10. **Materials Database Access Pattern**
**Issue:** Global `materials_database` dict accessed throughout code

**Recommendation:**
```python
class MaterialsDB:
    def __init__(self, filepath: Path):
        self.data = self._load(filepath)

    def get_material(self, name: str) -> Material:
        # Normalized access with error handling
        pass

    def get_density(self, name: str) -> float:
        # Centralized density lookup
        pass

# Dependency injection
@app.get("/materials")
async def get_materials(db: MaterialsDB = Depends(get_materials_db)):
    return db.data
```

---

### 11. **Frontend: Global State Management**
**Issue:** Global variables scattered throughout JavaScript

**Recommendation:**
```javascript
// Use state management pattern
const AppState = {
    materials: [],
    currentTab: 'kaggle',
    predictions: {},

    // Getters/setters
    setMaterials(materials) {
        this.materials = materials;
        this.notify('materials-updated');
    },

    // Observer pattern
    observers: {},
    subscribe(event, callback) { /* ... */ },
    notify(event) { /* ... */ }
};
```

---

### 12. **Missing Type Hints**
**Issue:** Some functions lack complete type hints

**Recommendation:**
- Add `from __future__ import annotations`
- Complete type hints for all functions
- Run `mypy --strict` and fix issues

---

### 13. **Magic Numbers and Constants**
**Issue:** Hard-coded values scattered throughout code

**Recommendation:**
```python
# constants.py
class PrintDefaults:
    EXTRUSION_WIDTH_MM = 0.4
    DEFAULT_PRINT_SPEED_MM_S = 60.0
    MAX_FILE_SIZE_MB = 15
    DEFAULT_DENSITY_G_CM3 = 1.25

class OptimizationDefaults:
    POPULATION_SIZE = 50
    NUM_GENERATIONS = 100
    CROSSOVER_PROB = 0.9
```

---

## 🟢 LOW PRIORITY (Nice to Have)

### 14. **Logging Infrastructure**
**Recommendation:** Replace `print()` statements with proper logging

```python
import logging
logger = logging.getLogger(__name__)

# Instead of: print(f"Loading model from {path}")
logger.info("Loading model", extra={"model": model_name, "path": str(path)})
```

---

### 15. **API Versioning**
**Recommendation:** Implement proper API versioning

```python
app = FastAPI(
    title="FDM Simulator API",
    version="2.0.0",
    openapi_prefix="/api/v2"
)

# Or path-based
@app.post("/api/v2/predict/{model_name}")
```

---

### 16. **Response Caching**
**Recommendation:** Cache expensive predictions with same inputs

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.post("/predict/{model_name}")
@cache(expire=3600)  # Cache for 1 hour
async def predict(...):
    pass
```

---

### 17. **Frontend: Component Architecture**
**Recommendation:** Split 3,416-line HTML into components

```
components/
  ├── ModelSelector.js
  ├── ParameterPanel.js
  ├── ResultsDisplay.js
  ├── OptimizationPanel.js
  └── STLViewer.js
```

---

### 18. **Testing Coverage**
**Current:** 5 test files, limited coverage
**Recommendation:**
- Add integration tests for all endpoints
- Add unit tests for extracted modules
- Set up CI/CD with coverage reporting
- Target: 80% coverage

---

### 19. **Documentation**
**Recommendation:**
- Add docstrings to all functions (50% currently missing)
- Generate API docs with examples
- Create architecture diagram
- Document model training procedures

---

### 20. **Database Migration**
**Issue:** JSON file for materials database doesn't scale

**Recommendation:**
- Migrate to SQLite for materials, predictions history
- Add caching layer (Redis)
- Enable analytics and query capabilities

---

### 21. **Async Optimization**
**Issue:** Synchronous code blocks event loop

**Recommendation:**
- Make model predictions truly async
- Use `asyncio.to_thread()` for CPU-bound tasks
- Consider Celery for long-running optimizations

---

### 22. **Security Hardening**
**Issues:**
- No rate limiting
- No input sanitization beyond Pydantic
- CORS allows all origins

**Recommendation:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict/{model_name}")
@limiter.limit("10/minute")
async def predict(...):
    pass

# Restrict CORS in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    # ...
)
```

---

### 23. **Performance Monitoring**
**Recommendation:**
- Add OpenTelemetry instrumentation
- Track prediction latencies
- Monitor model performance drift
- Set up alerts

---

## 📊 Prioritized Refactoring Roadmap

### Phase 1: Critical Deduplication (Week 1-2)
- [ ] Item 1: Consolidate training functions → **Expected: -540 lines**
- [ ] Item 2: Dynamic Pydantic models → **Expected: -140 lines**
- [ ] Item 3: Generic prediction endpoint → **Expected: -48 lines**
- [ ] Item 4: Frontend update functions → **Expected: -350 lines**

**Phase 1 Total: ~1,078 lines reduced (18% of codebase)**

### Phase 2: Architectural Improvements (Week 3-4)
- [ ] Item 5: Extract model config to YAML
- [ ] Item 6: Extract G-code parser module
- [ ] Item 7: Extract STL analyzer module
- [ ] Item 8: Implement custom exceptions

**Phase 2 Total: ~500 lines reduced, +3 new modules**

### Phase 3: Quality & Maintainability (Week 5-6)
- [ ] Items 9-13: Code quality improvements
- [ ] Items 14-16: Infrastructure upgrades
- [ ] Items 17-19: Testing and documentation

### Phase 4: Production Readiness (Week 7-8)
- [ ] Items 20-23: Database, security, monitoring

---

## 🎯 Quick Wins (Can Implement Today)

1. **Extract constants** → 30 min
2. **Add logging** → 1 hour
3. **Improve error messages** → 1 hour
4. **Add function docstrings** → 2 hours
5. **Move model registry to YAML** → 2 hours

---

## 📈 Expected Outcomes

### Code Metrics After Refactoring
- **main.py:** 1,777 → ~1,100 lines (38% reduction)
- **run_all_training.py:** 682 → ~150 lines (78% reduction)
- **fdm_simulator.html:** 3,416 → ~2,500 lines (27% reduction)
- **New modules:** +5 files (~800 lines total)
- **Net reduction:** ~1,325 lines (23% overall)

### Quality Improvements
- ✅ 90% reduction in code duplication
- ✅ Easier to add new models (15 min vs 2 hours)
- ✅ Better testability (unit tests possible)
- ✅ Clearer separation of concerns
- ✅ Maintainability significantly improved

---

## 🚫 What NOT to Change

1. **API contracts** - Keep existing endpoint signatures for backwards compatibility
2. **Model algorithms** - Don't change ML logic during refactoring
3. **UI layout** - Refactor code, not user experience
4. **Data formats** - Maintain CSV/JSON structures

---

## 📝 Implementation Notes

### Breaking Changes to Avoid
- Keep all existing `/predict/*` endpoints functional
- Maintain materials.json format compatibility
- Don't break existing frontend until refactored version tested

### Testing Strategy
1. Create integration tests for current behavior
2. Refactor code
3. Verify tests still pass
4. Add new unit tests for extracted modules

### Rollout Strategy
- Feature flag new implementations
- Run old and new code in parallel
- Gradual migration over 2-3 releases

---

## 🤝 Recommendations Summary

| Priority | Items | Estimated Effort | Impact |
|----------|-------|------------------|--------|
| 🔴 Critical | 1-7 | 40 hours | -1,578 lines, +testability |
| 🟠 High | 8-13 | 20 hours | +quality, +maintainability |
| 🟡 Medium | 14-19 | 30 hours | +production-ready |
| 🟢 Low | 20-23 | 40 hours | +scalability, +security |

**Total Estimated Effort:** 130 hours (~3-4 weeks full-time)

---

## Next Steps

1. **Review this analysis** with team/stakeholders
2. **Prioritize items** based on business needs
3. **Create GitHub issues** for each refactoring task
4. **Set up feature branch** for refactoring work
5. **Begin Phase 1** with critical deduplication items

**Questions? Need clarification on any recommendation? Ready to start implementation?**
