#            UNDERSTANDING AND PREDICTION OF ELECTRONIC BAND GAPS IN INORGANIC CRYSTAL THROUGH EXPLAINABLE AI & ENSEMBLE MODELS

# The dataset was obtained from the Materials Project database.
# I used three ensemble models: Random Forest, Gradient Boosting, and XGBoost. These models were chosen because they handle complex, non-linear relationships in material data effectively. I compared their performance to identify the best model for bandgap prediction.
# The objective was to improve predictive accuracy of bandgap values using ensemble learning techniques.

!pip install pymatgen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from pymatgen.core import Composition
from pymatgen.ext.matproj import MPRester


# LOAD DATA
df = pd.read_csv("/content/super_enhanced_materials_data.csv")

# The dataset used in this project is loaded from an csv file
# It is already cleaned!
# Missing values were replaced with the median
# No duplicates were found
# NO outlier is removed because of variation in bandgap could be possible
# not normalized data because of same results
# Dataset contains 17088 rows 19 columns of material properties

X = df.drop(columns=["band_gap"])
y = df["band_gap"]

# SAVE FEATURE NAMES
feature_cols = X.columns
joblib.dump(feature_cols, "features.pkl")

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODELS
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
}

results = {}
trained_models = {}

# TRAIN + EVALUATE
def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print("\n====================")
    print(name)
    print("R2   :", round(r2, 4))
    print("MAE  :", round(mae, 4))
    print("RMSE :", round(rmse, 4))

    plt.figure()
    plt.scatter(y_test, pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--')
    plt.xlabel("Actual Bandgap")
    plt.ylabel("Predicted Bandgap")
    plt.title(name + " Actual vs Predicted")
    plt.show()

    return model, r2

for name, model in models.items():
    m, score = evaluate(name, model)
    trained_models[name] = m
    results[name] = score

# BEST MODEL
best_name = max(results, key=results.get)
best_model = trained_models[best_name]

print("\nBEST MODEL:", best_name)

joblib.dump(best_model, "best_model.pkl")

# FEATURE IMPORTANCE
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    idx = np.argsort(imp)[-15:]

    plt.figure()
    plt.barh(range(len(idx)), imp[idx])
    plt.yticks(range(len(idx)), feature_cols[idx])
    plt.title("Feature Importance")
    plt.show()

# SHAP Explainability
explainer = shap.TreeExplainer(best_model)
X_sample = X_test.sample(100, random_state=42)
shap_values = explainer.shap_values(X_sample, check_additivity=False)
shap.summary_plot(shap_values, X_sample)

#Cross Validation:
scores = cross_val_score(best_model, X, y, cv=5, scoring="r2")

print("CV R2 Mean:", scores.mean())
print("CV R2 Std:", scores.std())

#Saving CV results

cv_results = {
    "cv_r2_mean": float(scores.mean()),
    "cv_r2_std": float(scores.std())
}

with open("cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=4)

#final Metrices save
y_pred = best_model.predict(X_test)

final_metrics = {
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "r2": float(r2_score(y_test, y_pred))
}

with open("final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=4)

#save Predictions:
pred_df = X_test.copy()
pred_df["actual_bandgap"] = y_test.values
pred_df["predicted_bandgap"] = best_model.predict(X_test)

pred_df.to_csv("predictions.csv", index=False)

# USER INPUT
formula = input("Enter formula (e.g. SiO2): ")
material_id = input("Enter Material ID (e.g. mp-149): ")

# FEATURE EXTRACTION (FIXED)
def extract_features(formula):
    comp = Composition(formula)
    elements = list(comp.elements)

    atomic_masses = np.array([el.atomic_mass for el in elements])
    electronegativities = np.array([el.X for el in elements if el.X is not None])
    groups = np.array([el.group for el in elements])
    periods = np.array([el.row for el in elements])
    valence = np.array([el.full_electronic_structure[-1][2] for el in elements])
    radii = np.array([el.atomic_radius if el.atomic_radius else np.nan for el in elements])

    feat = {}

    feat["nelements"] = len(elements)
    feat["avg_atomic_mass"] = atomic_masses.mean()

    feat["avg_electronegativity"] = electronegativities.mean()
    feat["max_en"] = electronegativities.max()
    feat["min_en"] = electronegativities.min()
    feat["range_en"] = feat["max_en"] - feat["min_en"]

    feat["avg_ve"] = valence.mean()
    feat["range_ve"] = valence.max() - valence.min()
    feat["total_ve"] = valence.sum()

    feat["avg_group"] = groups.mean()
    feat["std_group"] = groups.std()

    feat["avg_period"] = periods.mean()

    feat["avg_rad"] = np.nanmean(radii)
    feat["range_rad"] = np.nanmax(radii) - np.nanmin(radii)

    feat["n_unique"] = len(set(elements))

    return feat

# MATERIALS PROJECT FEATURES
def get_structure_features(material_id, total_ve):
    mpr = MPRester("KmE2G5s3nkK7j3hV29ahg0NagFmBM6Wz") #API KEY material Project

    try:
        structure = mpr.get_structure_by_material_id(material_id)

        density = float(structure.density)
        volume = float(structure.volume)
        nsites = len(structure.sites)

        vol_per_site = volume / nsites
        density_ve_ratio = density / total_ve if total_ve != 0 else 0

    except:
        density = 0
        vol_per_site = 0
        density_ve_ratio = 0

    return density, vol_per_site, density_ve_ratio

# BUILD INPUT
feat = extract_features(formula)

density, vol_per_site, density_ve_ratio = get_structure_features(
    material_id,
    feat["total_ve"]
)

feat["density"] = density
feat["vol_per_site"] = vol_per_site
feat["density_ve_ratio"] = density_ve_ratio

# LOAD TRAINED FEATURE ORDER
feature_cols = joblib.load("features.pkl")

X_new = pd.DataFrame([feat])
X_new = X_new.reindex(columns=feature_cols, fill_value=0)

# PREDICT
model = joblib.load("best_model.pkl")
pred = model.predict(X_new)[0]

print("\n====================")
print("PREDICTED BAND GAP:", pred)

# SHAP LOCAL EXPLANATION
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_new)

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_new.iloc[0]
    )
)
