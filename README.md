# Understanding and Prediction of Electronic Band Gaps in Inorganic Crystals through Explainable AI and Ensemble Models

A supervised machine learning framework for predicting the electronic band gap of inorganic crystalline materials using ensemble models and SHAP-based explainability — a scalable, data-driven alternative to computationally expensive Density Functional Theory (DFT) simulations.

**NED University of Engineering & Technology — MS Artificial Intelligence | Spring 2026**  
**Author:** Aliza Jahanzaib | **Instructor:** Miss Madiha Aslam

---

## Overview

The electronic band gap determines whether a material behaves as an insulator, semiconductor, or conductor — a critical property for photovoltaics, optoelectronics, and semiconductor engineering. This project builds an end-to-end prediction pipeline over 17,087 inorganic crystalline materials sourced from the Materials Project database, engineers 19 physically meaningful features from raw chemical formulas, and interprets predictions using SHAP analysis.

---

## How It Works

The system is designed as an end-to-end band gap prediction tool for researchers and material scientists. Once the model is trained and saved, any user can query it with just a chemical formula and a Material ID.

```
User Input: Chemical Formula + Material ID
                    ↓
    Feature Extraction via pymatgen + Materials Project API
    (electronegativity, atomic mass, density, valence electrons, etc.)
                    ↓
         Pre-trained Model (best_model.pkl)
                    ↓
         Predicted Band Gap (eV)
                    ↓
    SHAP Waterfall Plot — explains which features
    drove the prediction and by how much
```

> The training pipeline (RF / GB / XGBoost comparison, evaluation, best model selection) runs once in the background. Everything the user interacts with is the prediction and explanation step.

---

## Models

| Model | Configuration |
|-------|--------------|
| Random Forest | 200 estimators, random_state=42 |
| Gradient Boosting | default, random_state=42 |
| XGBoost | 300 estimators, learning_rate=0.05 |

Best model is automatically selected by R² score and saved as `best_model.pkl`.

---

## Features (19 descriptors)

Extracted from chemical formula using `pymatgen` and the Materials Project API:

| Category | Features |
|----------|----------|
| Composition | `nelements`, `n_unique` |
| Electronegativity | `avg_electronegativity`, `max_en`, `min_en`, `range_en` |
| Atomic mass | `avg_atomic_mass` |
| Valence electrons | `avg_ve`, `range_ve`, `total_ve` |
| Periodic table | `avg_group`, `std_group`, `avg_period` |
| Atomic radius | `avg_rad`, `range_rad` |
| Structure (API) | `density`, `vol_per_site`, `density_ve_ratio` |

---

## Evaluation Metrics

- R² (Coefficient of Determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- 5-Fold Cross-Validation R²

Results are saved to `final_metrics.json` and `cv_results.json`.

---

## Output Files

| File | Description |
|------|-------------|
| `best_model.pkl` | Trained best model |
| `features.pkl` | Feature column order for inference |
| `final_metrics.json` | MAE, RMSE, R² on test set |
| `cv_results.json` | 5-fold CV mean and std R² |
| `predictions.csv` | Actual vs predicted band gaps on test set |

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train models
```bash
# Run in Google Colab or locally
python bandgap_prediction.py
```

### Predict for a new material
When prompted:
```
Enter formula (e.g. SiO2): TiO2
Enter Material ID (e.g. mp-149): mp-2657
```
The script outputs the predicted band gap in eV and a SHAP waterfall plot explaining the prediction.

---

## Key Findings

- Feature engineering had a greater impact on model performance than model complexity
- Random Forest achieved the best R² (~73%) after systematic feature engineering
- `max_en` (maximum electronegativity) was the top predictor per both feature importance and SHAP analysis
- Outlier removal negatively affected performance — outliers were retained
- Normalization did not improve tree-based model accuracy

---

## Tech Stack

Python · Scikit-learn · XGBoost · SHAP · pymatgen · Pandas · NumPy · Matplotlib · joblib · Materials Project API

---

## References

- Curtarolo et al., "High-throughput computational materials design," *Nature Materials*, 2013
- Ramakrishnan et al., "Electronic properties from machine learning," *Scientific Data*, 2014
- Ward et al., "Machine learning framework for inorganic materials," *npj Computational Materials*, 2016
- Lundberg & Lee, "SHAP: Interpretable machine learning," *NeurIPS*, 2017
