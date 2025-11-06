# FDM Simulator - Feature Roadmap & Research TODO

This document tracks planned features, proposed improvements, and research areas for the FDM 3D Print Property Simulator.

> **Last Updated:** November 6, 2025
> **Current Version:** v12 (Parameter Sensitivity Analysis)

---

## 🚀 Planned Features (v13+)

### High Priority

#### 1. Tool Path Optimization & Generation
**Status:** 📋 Planned
**Research Basis:** `Fok_ACO_based_Tool_path_Optimizer.pdf`, `131 Fundamental Path Optimization Strategies for E.pdf`
**Description:**
- Implement Ant Colony Optimization (ACO) for tool path planning
- Optimize infill patterns for minimum print time while maintaining strength targets
- Generate alternative raster strategies beyond slicer defaults
- Support for continuous fiber path optimization (for composite materials)

**Implementation Notes:**
- Add `/optimize/toolpath` endpoint
- Input: STL mesh + target properties (strength, time, cost)
- Output: Optimized G-code segments or slicer parameter recommendations
- Consider integration with existing optimizer (pymoo) for multi-objective path planning

**Research Gaps:**
- [ ] Review ACO vs. Genetic Algorithm vs. Particle Swarm for FDM path optimization
- [ ] Benchmark against OrcaSlicer/PrusaSlicer default strategies
- [ ] Investigate continuous vs. discrete path representations

---

#### 2. Real-Time Print Monitoring & Defect Detection
**Status:** 📋 Planned
**Research Basis:** `Machinelearningbasedmonitoringandoptimizationofprocessingparametersin3Dprinting.pdf`, `Cost-effective_sensor-2025.pdf`
**Description:**
- Add computer vision-based layer quality monitoring
- Detect under-extrusion, over-extrusion, warping during print
- Predictive alerts for potential failures (e.g., corner lifting at layer 15/200)
- Real-time parameter adjustment recommendations

**Implementation Notes:**
- Requires camera feed input (REST API for image upload or streaming)
- Train CNN on defect classification dataset
- Integrate with existing models to suggest corrective actions
- Optional: Hardware integration guide for webcam/Raspberry Pi setup

**Research Gaps:**
- [ ] Survey existing defect classification datasets (open-access)
- [ ] Benchmark lightweight models (MobileNet, EfficientNet) for edge deployment
- [ ] Investigate transfer learning from injection molding defect detection

---

#### 3. Advanced Multi-Objective Optimizer Enhancements
**Status:** 📋 Planned
**Research Basis:** `Shirmohammadi2021_Article_OptimizationOf3DPrintingProces.pdf`, `International Journal of Chemical Engineering - 2022 - Raja - Optimization of 3D Printing Process Parameters of Polylactic.pdf`
**Description:**
- Add constraint-based optimization (e.g., "strength ≥ 50 MPa, cost ≤ $2.00")
- Support for user-defined objective functions (custom formulas)
- Pareto front visualization with interactive trade-off exploration
- Template presets: "Maximum Strength", "Minimum Cost", "Fastest Print", "Best Surface Quality"

**Implementation Notes:**
- Extend existing NSGA-II implementation with constraint handling
- Add Pareto front plotting endpoint (returns JSON for frontend charting)
- Create optimizer templates in `materials.json` or separate config file
- UI: Interactive slider to explore Pareto optimal solutions

**Research Gaps:**
- [ ] Compare NSGA-II vs. NSGA-III for 4+ objectives
- [ ] Investigate adaptive constraint relaxation for infeasible spaces
- [ ] Benchmark optimizer runtime for 9-model simultaneous optimization

---

#### 4. Material Database Expansion & Recommendation System
**Status:** 📋 Planned
**Research Basis:** Existing `materials.json` structure
**Description:**
- Expand to 50+ materials with verified datasheets (currently ~10-15)
- Material recommendation based on application requirements
- Cost-performance analysis (e.g., "PETG 30% cheaper than ABS for this use case")
- Community-contributed material profiles with validation

**Implementation Notes:**
- Create `/materials/recommend` endpoint
- Input: Target properties (tensile ≥ X, temp resistance ≥ Y, cost ≤ Z)
- Output: Ranked list of suitable materials with justifications
- Add material similarity search (find alternatives to unavailable filaments)
- Schema validation for community contributions

**Research Gaps:**
- [ ] Identify authoritative filament datasheets (manufacturer specs vs. independent testing)
- [ ] Develop material clustering/similarity metrics
- [ ] Investigate material substitution rules (e.g., PLA+ ≈ PLA with higher temp resistance)

---

### Medium Priority

#### 5. G-Code Post-Processing & Correction
**Status:** 💡 Proposed
**Research Basis:** Existing G-code parser (v10-v11), tool path optimization research
**Description:**
- Automatically correct G-code based on predicted defects
- Add support structures only where warpage model predicts >1mm distortion
- Adjust extrusion multiplier per-layer based on FEA stress predictions
- Generate repair patches for failed prints (resume from specific layer)

**Implementation Notes:**
- Extend existing G-code parser to writer/modifier
- Add `/gcode/optimize` endpoint (input: G-code, target corrections)
- Integrate with warpage, accuracy, and FEA models for prediction-driven edits
- Safety checks to prevent invalid G-code generation

---

#### 6. Time-Series Print Simulation (Layer-by-Layer Prediction)
**Status:** 💡 Proposed
**Description:**
- Predict property evolution as print progresses (e.g., warpage at layer 10, 50, 100, 200)
- Identify critical layers where interventions are most effective
- Estimate cumulative thermal stress buildup
- Visualize property gradients across part height

**Implementation Notes:**
- Create `/simulate/timeseries` endpoint
- Input: G-code or sliced parameters + number of layers
- Output: Array of predictions per layer or layer range
- Frontend: Timeline visualization with property charts

---

#### 7. Failure Mode & Effects Analysis (FMEA) for FDM
**Status:** 💡 Proposed
**Description:**
- Systematic analysis of potential failure modes (delamination, warping, cracking, etc.)
- Risk scoring based on parameter combinations
- Preventive recommendations before printing
- Integration with existing models (fatigue, warpage, accuracy)

**Implementation Notes:**
- Create failure taxonomy (10-20 common FDM failure modes)
- Map parameter ranges to failure probabilities
- Add `/analyze/fmea` endpoint
- Output: Risk matrix with mitigation strategies

---

#### 8. Slicer Profile Auto-Generation
**Status:** 💡 Proposed
**Description:**
- Generate PrusaSlicer, OrcaSlicer, Cura config files from optimized parameters
- One-click export from optimizer results to slicer-ready profiles
- Template inheritance (base profile + optimized overrides)
- Version compatibility management (Cura 5.x vs. 4.x syntax)

**Implementation Notes:**
- Add `/export/slicer_profile` endpoint
- Support JSON (OrcaSlicer), INI (PrusaSlicer), and XML (Cura) formats
- Map simulator parameters to slicer-specific settings
- Include metadata (created by FDM Simulator, optimization objectives)

---

### Low Priority / Exploratory

#### 9. Blockchain-Based Print Certification
**Status:** 🔬 Exploratory
**Description:**
- Cryptographic verification of print parameters and predicted properties
- Immutable record for quality assurance and compliance
- Supply chain tracking for critical parts (aerospace, medical)

**Research Gaps:**
- [ ] Assess industry demand for blockchain certification
- [ ] Evaluate lightweight blockchain solutions (IPFS + hash anchoring)
- [ ] Legal/regulatory research on certification requirements

---

#### 10. Augmented Reality (AR) Print Preview
**Status:** 🔬 Exploratory
**Description:**
- AR visualization of predicted part properties overlaid on physical print bed
- Highlight stress concentration zones, warpage areas, weak interfaces
- Mobile app for print monitoring with AR guidance

**Research Gaps:**
- [ ] Survey AR frameworks for web/mobile (AR.js, WebXR, ARCore/ARKit)
- [ ] Evaluate computational feasibility (property visualization rendering)
- [ ] User study on AR utility vs. traditional 2D visualization

---

#### 11. Federated Learning for Model Improvement
**Status:** 🔬 Exploratory
**Description:**
- Allow users to contribute print outcome data without uploading sensitive designs
- Continuously improve models via federated learning
- Privacy-preserving collaborative dataset expansion

**Research Gaps:**
- [ ] Review federated learning frameworks (TensorFlow Federated, PySyft)
- [ ] Design privacy-preserving data contribution protocol
- [ ] Legal/ethical considerations for user data collection

---

## 🔬 Research Areas to Explore

### 1. Tool Path Optimization Algorithms
**Priority:** High
**Current Gap:** No path generation capabilities; rely on external slicers
**Papers to Review:**
- `Fok_ACO_based_Tool_path_Optimizer.pdf` - Ant Colony Optimization for FDM
- `131 Fundamental Path Optimization Strategies for E.pdf` - Survey of strategies
- Additional search: Genetic algorithms for 3D printing, shortest path with constraints

**Research Questions:**
- Which optimization algorithm provides best time-quality trade-off for FDM?
- Can we beat commercial slicers (Cura, OrcaSlicer) on specific objectives?
- How to incorporate anisotropy (raster angle) into path planning?
- Real-time path re-planning for mid-print adjustments?

**Action Items:**
- [ ] Implement baseline ACO for simple infill patterns
- [ ] Benchmark against PrusaSlicer's Hilbert curve, Archimedean chords
- [ ] Survey continuous fiber path planning (MarkForged patents)
- [ ] Investigate GPU-accelerated path optimization (CUDA/OpenCL)

---

### 2. In-Situ Monitoring & Computer Vision
**Priority:** High
**Current Gap:** Post-hoc prediction only; no real-time monitoring
**Papers to Review:**
- `Machinelearningbasedmonitoringandoptimizationofprocessingparametersin3Dprinting.pdf`
- `Cost-effective_sensor-2025.pdf` - Low-cost sensor strategies

**Research Questions:**
- What are minimum hardware requirements for effective defect detection?
- Can we use thermal imaging vs. RGB cameras for better warpage detection?
- How to correlate sensor data with existing predictive models?
- Edge deployment on Raspberry Pi vs. cloud processing trade-offs?

**Action Items:**
- [ ] Survey open datasets: layer images with defect labels
- [ ] Test transfer learning from injection molding defect detection
- [ ] Prototype with OctoPrint camera feeds (REST API integration)
- [ ] Investigate acoustic emission for layer adhesion quality

---

### 3. Multi-Material & Gradient Printing
**Priority:** Medium
**Current Gap:** Multi-material model exists but limited to binary interfaces
**Papers to Review:**
- Research on functionally graded materials (FGM) in FDM
- Continuous material transition studies
- Multi-extruder coordination and oozing prevention

**Research Questions:**
- How to model property gradients in 3+ material systems?
- Optimal transition zone thickness for material pairs?
- Toolpath strategies to minimize material mixing/contamination?
- Thermal management in multi-material prints (sequential vs. simultaneous)?

**Action Items:**
- [ ] Extend multi-material model to ternary and quaternary systems
- [ ] Collect FGM experimental data (literature or collaboration)
- [ ] Investigate lattice structures with material gradients
- [ ] Model purge tower optimization (minimize waste)

---

### 4. Topology Optimization Integration
**Priority:** Medium
**Current Gap:** FEA material card exists but no generative design loop
**Papers to Review:**
- Topology optimization for additive manufacturing
- SIMP, BESO, level-set methods for FDM
- Overhang constraints for self-supporting structures

**Research Questions:**
- Can we integrate topology optimization with anisotropic FEA properties?
- How to enforce FDM-specific manufacturing constraints (45° overhang, layer orientation)?
- Optimization for multi-load cases with different raster angles?
- Lattice infill vs. solid shell trade-offs in generative design?

**Action Items:**
- [ ] Survey open-source topology optimization tools (TopOpt, ToPy)
- [ ] Prototype with FEniCS or MOOSE for FEA backend
- [ ] Integrate with existing FEA material card predictions
- [ ] Generate STL from optimized density field (marching cubes)

---

### 5. Sustainability & Environmental Impact Modeling
**Priority:** Medium
**Current Gap:** Cost/time estimation exists but no carbon footprint or energy use
**Papers to Review:**
- Life cycle assessment (LCA) for FDM printing
- Energy consumption models for different materials
- Recycling and circular economy for filaments

**Research Questions:**
- How to model embodied energy of different filaments (virgin vs. recycled)?
- Electricity consumption prediction based on print parameters?
- Material waste quantification (supports, failed prints, purge towers)?
- Carbon offset strategies for FDM manufacturing?

**Action Items:**
- [ ] Add energy consumption to cost model (kWh per print)
- [ ] Integrate material recycling properties (e.g., rPETG vs. virgin PETG)
- [ ] Create sustainability optimizer objective (minimize carbon footprint)
- [ ] Partner with filament manufacturers for LCA data

---

### 6. Uncertainty Quantification & Confidence Intervals
**Priority:** Low (Technical Improvement)
**Current Gap:** Point predictions only; no confidence intervals or variance estimates
**Papers to Review:**
- Bayesian neural networks for AM
- Gaussian processes for uncertainty quantification
- Ensemble methods for prediction intervals

**Research Questions:**
- How to propagate input uncertainties (filament variability, printer calibration) to output predictions?
- Bayesian vs. frequentist approaches for FDM property prediction?
- Active learning to reduce uncertainty in under-explored parameter spaces?
- User-friendly visualization of prediction confidence?

**Action Items:**
- [ ] Implement prediction intervals (90%, 95% confidence)
- [ ] Add uncertainty bars to frontend visualizations
- [ ] Investigate quantile regression for asymmetric uncertainties
- [ ] Benchmarking: compare to experimental standard deviations

---

### 7. Adaptive Manufacturing & Closed-Loop Control
**Priority:** Low (Long-Term Vision)
**Current Gap:** Open-loop prediction; no feedback from actual prints
**Papers to Review:**
- Closed-loop control in additive manufacturing
- Model predictive control (MPC) for FDM
- Reinforcement learning for process parameter tuning

**Research Questions:**
- Can real-time sensor feedback improve model predictions for next print?
- How to implement MPC for layer-by-layer parameter adjustment?
- Reinforcement learning for exploring optimal parameter spaces?
- Integration with smart printers (Prusa XL, Bambu Lab X1)?

**Action Items:**
- [ ] Design closed-loop architecture (sensors → model update → parameter adjust)
- [ ] Prototype with simulated "digital twin" environment
- [ ] Investigate OctoPrint plugin for model-in-the-loop control
- [ ] Survey printer APIs (Prusa Connect, Bambu Cloud, Klipper)

---

## 📊 Data Collection & Model Improvements

### Needed Datasets
- [ ] **Tool path optimization benchmark suite** - Standard test cases with known optimal solutions
- [ ] **Defect image dataset** - Labeled layer images (good, under-extrusion, stringing, warping, etc.)
- [ ] **Multi-material gradient data** - Properties of FGM prints with 2-5 material transitions
- [ ] **Energy consumption data** - kWh measurements across materials, speeds, temperatures
- [ ] **Real-world failure data** - User-submitted failed prints with root cause analysis
- [ ] **Long-term aging data** - Mechanical property degradation over time (UV, moisture, stress)

### Model Architecture Experiments
- [ ] Replace scikit-learn with deep learning (PyTorch/TensorFlow) for complex interactions
- [ ] Investigate graph neural networks (GNNs) for STL mesh → property prediction
- [ ] Physics-informed neural networks (PINNs) to embed FDM equations
- [ ] Multi-task learning: single model predicting all 9 outputs simultaneously
- [ ] Gaussian process models for uncertainty-aware predictions

---

## 🛠️ Infrastructure & Tooling

### Testing & Validation
- [ ] Expand pytest coverage to 90%+ (currently ~8 tests)
- [ ] Add integration tests for full prediction pipeline
- [ ] Benchmark performance tests (response time, model inference speed)
- [ ] Create synthetic data generator for edge case testing
- [ ] CI/CD: Automated model retraining pipeline on new data

### Documentation
- [x] Create model documentation pages (v12 - completed)
- [ ] Add API tutorials (Python, JavaScript, curl examples)
- [ ] Video walkthrough of optimizer workflow
- [ ] Case studies: real-world applications (drone frame, prosthetic, enclosure)
- [ ] Developer guide: How to add a new model

### Deployment
- [ ] Docker containerization for easy deployment
- [ ] Kubernetes config for scalable cloud hosting
- [ ] Model versioning and A/B testing infrastructure
- [ ] Rate limiting and API key management
- [ ] Monitoring dashboard (Prometheus + Grafana)

---

## 🤝 Community & Collaboration

### Open-Source Contributions
- [ ] Publish anonymized training datasets (with user consent)
- [ ] Create contribution guide for new models
- [ ] Establish model validation protocol (benchmark against test set)
- [ ] Host community challenges (e.g., "Predict warpage competition")

### Industry Partnerships
- [ ] Collaborate with filament manufacturers for verified material data
- [ ] Partner with printer OEMs for in-situ monitoring integration
- [ ] Academic collaborations for research dataset sharing
- [ ] Engage with makerspaces for user feedback and testing

---

## 📅 Version Roadmap (Tentative)

### v13 - Tool Path Optimization (Q1 2026)
- ACO-based infill pattern generator
- Tool path export to G-code
- Benchmarking vs. commercial slicers

### v14 - Real-Time Monitoring (Q2 2026)
- Defect detection API
- Camera feed integration guide
- Alert system for print failures

### v15 - Advanced Optimization (Q3 2026)
- Constraint-based optimization
- Pareto front visualization
- Optimizer templates

### v16 - Material Intelligence (Q4 2026)
- Material recommendation system
- Expanded database (50+ materials)
- Community contribution platform

---

## 🔖 Priority Matrix

| Feature | Impact | Effort | Priority | Research Needed |
|---------|--------|--------|----------|-----------------|
| Tool Path Optimization | High | High | **P0** | Medium |
| Real-Time Monitoring | High | High | **P0** | High |
| Optimizer Enhancements | Medium | Low | **P1** | Low |
| Material Database | Medium | Medium | **P1** | Low |
| G-Code Post-Processing | Medium | Medium | **P2** | Medium |
| Time-Series Simulation | Medium | High | **P2** | Medium |
| FMEA Analysis | Low | Medium | **P3** | High |
| Slicer Profile Export | Low | Low | **P3** | Low |
| Topology Optimization | High | Very High | **P4** | Very High |
| Sustainability Modeling | Low | Medium | **P4** | Medium |

**Priority Levels:**
- **P0:** Critical for next release
- **P1:** High value, plan for near-term
- **P2:** Desirable, schedule as resources allow
- **P3:** Nice-to-have, opportunistic
- **P4:** Long-term vision, research phase

---

## 📝 Notes

- This document is living and should be updated as research progresses and priorities shift
- Research papers are cataloged in `docs/research/REFERENCES.md` with detailed notes in `docs/research/notes/`
- Feature requests from users should be added to the appropriate priority section
- Cross-reference with GitHub Issues for detailed technical discussions

**Maintainers:**
- Review this TODO quarterly
- Archive completed items to `CHANGELOG.md`
- Solicit community input on priority ranking

---

*For detailed technical implementation plans, see individual issues in the GitHub repository or create new issues referencing specific TODO items.*
