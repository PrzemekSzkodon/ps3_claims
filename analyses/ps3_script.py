# %%

print("quick check")

# %%
# Debug cell: Check Python environment and package installation
import sys, importlib
print(sys.executable)  # Shows which Python interpreter is being used
print(importlib.util.find_spec("ps3"))  # Verifies that the ps3 package can be found

# %%
# Import all required libraries for data manipulation, modeling, and visualization
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For numerical operations and arrays
import pandas as pd  # For working with DataFrames (tabular data)
from dask_ml.preprocessing import Categorizer  # For converting categorical variables
from glum import GeneralizedLinearRegressor, TweedieDistribution  # GLM for insurance data
from lightgbm import LGBMRegressor  # Gradient boosting machine learning model
from sklearn.compose import ColumnTransformer  # Apply different transforms to different columns
from sklearn.metrics import auc  # Calculate area under curve for model evaluation
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.pipeline import Pipeline  # Chain multiple preprocessing and modeling steps
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler  # Data transformations

# Import custom functions from the ps3 package
from ps3.data import create_sample_split, load_transform
from ps3.evaluation import evaluate_predictions  # Step 3–8: evaluation metrics helper

# %%
# Load and transform the French motor insurance dataset
# This function loads data from local files or remote sources and applies preprocessing
df = load_transform()

# %%
# Explore the data: check dimensions and preview the first few rows
print(df.shape)  # Shows (number_of_rows, number_of_columns)
df.head()  # Displays the first 5 rows to understand the data structure

# %%
# Prepare data for modeling: Create target variable and sample weights
# Train benchmark tweedie model. This is entirely based on the glum tutorial.

# Extract exposure as weights: Exposure is the fraction of a year the policy was active
# E.g., 240 days → 0.658 years.
weight = df["Exposure"].values

# Calculate Pure Premium: This is the claim amount per unit time (per year)
# We divide by Exposure to normalize claim amounts to an annual basis
# Example: $1000 claim over 0.5 years → $2000 per year pure premium
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]  # This is our target variable (what we want to predict)

# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
# By dividing by exposure, we standardize the claim amounts to a per-unit-time basis (e.g., per year).
# This allows for a fair comparison of claim amounts across policies with different exposure periods.

# %%
# Split the data into training and test sets
# Fit a basic Tweedie GLM
# TODO: use your create_sample_split function here

# Create train/test split based on Region (ensures policies from same region stay together)
# This uses consistent hashing so the split is reproducible
df = create_sample_split(df, "IDpol")

# Get indices for training and test rows
train = np.where(df["sample"] == "train")  # Returns array of indices where sample == "train"
test = np.where(df["sample"] == "test")    # Returns array of indices where sample == "test"

# Create separate DataFrames for train and test (copy to avoid modifying original)
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

# Define which columns are categorical (non-numeric categories)
# These include vehicle characteristics, geographic info, and driver/vehicle age bins
categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

# Combine categorical and numeric predictors into one list
# BonusMalus: insurance bonus/malus score, Density: population density
predictors = categoricals + ["BonusMalus", "Density"]

# Create a categorizer that converts categorical columns to a format the model can use
glm_categorizer = Categorizer(columns=categoricals)

# Prepare training and test feature matrices (X) and target vectors (y)
# fit_transform learns the categories from training data and transforms it
X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
# transform only applies the learned categories to test data (no learning on test data!)
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])

# Split target variable (y) and weights (w) into train and test
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

# Create and fit a Tweedie GLM (Generalized Linear Model)
# Tweedie distribution with power=1.5 is ideal for insurance claims (handles zeros and positives)
TweedieDist = TweedieDistribution(1.5)

# Initialize the GLM with L1 regularization (Lasso) to reduce overfitting
# l1_ratio=1 means pure L1 penalty (some coefficients will be exactly zero)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)

# Fit the model using training data, weighted by exposure
# Longer exposures contribute more to the model fitting
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)

# Display model coefficients (intercept + all feature coefficients)
# This shows how much each feature contributes to the prediction
pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

# Generate predictions for both train and test sets
# pp = "pure premium" (predicted annual claim amount)
df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

# Evaluate model performance using deviance (lower is better)
# Deviance measures how well the model fits the data for the Tweedie distribution
# We divide by total weight to get average deviance per unit exposure
print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# Compare total predicted claims vs actual claims on test set
# Predicted: multiply predicted annual premium by exposure to get actual claim amount
print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)

# %%
# Display coefficients with approximate statistics
# Note: glum doesn't provide SEs directly, so we show coefficients and their relative magnitude

coef_df = pd.DataFrame(
    {
        "coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_)),
        "abs_coefficient": np.abs(np.concatenate(([t_glm1.intercept_], t_glm1.coef_)))
    },
    index=["intercept"] + t_glm1.feature_names_,
)

# Sort by absolute value to see most influential features
coef_df.sort_values("abs_coefficient", ascending=False).head(6)

# %%
# Compare train and test distributions side by side
pd.concat([
    df_train["pp_t_glm1"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]),
    df_test["pp_t_glm1"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99])
], axis=1, keys=['Train', 'Test'])
# %%# ...existing code...

# Build an improved model using splines for numeric variables
# Splines allow non-linear relationships (e.g., risk doesn't increase linearly with BonusMalus)
numeric_cols = ["BonusMalus", "Density"]

# Define a numeric pipeline: scale → spline
# Scaling stabilizes spline basis creation; splines capture non-linear effects.
numeric_pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("spline", SplineTransformer(knots="quantile", degree=3, extrapolation="continue"))
])

# ColumnTransformer applies different preprocessing to different column groups
# Use drop="first" so the categorical one-hot doesn’t add an extra intercept column.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
        ("num_spline", numeric_pipeline, numeric_cols),
    ],
    remainder="drop",
)
# Emit a pandas DataFrame for feature names downstream
preprocessor.set_output(transform="pandas")

# Chain preprocessing with the Tweedie GLM
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimate", GeneralizedLinearRegressor(
        family=TweedieDist,      # reuse TweedieDistribution(1.5) created earlier
        l1_ratio=1,
        fit_intercept=True       # ensure only GLM adds the intercept
    ))
])

# Optional: inspect pipeline
model_pipeline

# Fit on TRAIN features only to avoid leakage; pass sample weights to the estimator step
model_pipeline.fit(df_train[predictors], y_train_t, estimate__sample_weight=w_train_t)

# Show coefficients from the GLM step (after preprocessing)
pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

# Predict on train/test using the same learned preprocessing
df_test["pp_t_glm2"] = model_pipeline.predict(df_test[predictors])
df_train["pp_t_glm2"] = model_pipeline.predict(df_train[predictors])

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# =============================================================
# Monotonicity Exercise (BonusMalus) – LightGBM Tweedie Models
# Steps 1–5: Weighted curve, unconstrained vs monotonic constrained LGBM
# =============================================================

# Step 1: Exposure-weighted average pure premium per BonusMalus
# (PurePremium already = ClaimAmountCut / Exposure). We re-compute to be explicit.
bm_curve = (
    df_train.groupby("BonusMalus").apply(
        lambda g: np.sum(g["ClaimAmountCut"]) / np.sum(g["Exposure"])
    ).rename("weighted_pure_premium").reset_index()
)

plt.figure(figsize=(8,4))
plt.plot(bm_curve["BonusMalus"], bm_curve["weighted_pure_premium"], marker="o", linewidth=1)
plt.title("Exposure-weighted Pure Premium vs BonusMalus")
plt.xlabel("BonusMalus (higher = worse claim history)")
plt.ylabel("Weighted Pure Premium")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# If we do not include a monotonicity constraint, noise + interactions can cause
# local inversions (risk decreasing for worse scores), which is undesirable for pricing logic.

# Step 2: Build one-hot encoded design matrices for LightGBM
feature_cols = predictors  # same predictors list (categoricals + numeric)
X_train_lgb = pd.get_dummies(df_train[feature_cols], drop_first=True)
X_test_lgb = pd.get_dummies(df_test[feature_cols], drop_first=True)
X_test_lgb = X_test_lgb.reindex(columns=X_train_lgb.columns, fill_value=0)

# Step 3: Define baseline (unconstrained) LightGBM Tweedie model + CV
param_grid_unconstrained = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}

lgb_unconstrained = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    random_state=42,
)

gs_unconstrained = GridSearchCV(
    estimator=lgb_unconstrained,
    param_grid=param_grid_unconstrained,
    cv=3,
    n_jobs=-1,
    verbose=0,
)
gs_unconstrained.fit(X_train_lgb, y_train_t, sample_weight=w_train_t)

df_train["pp_t_lgbm"] = gs_unconstrained.best_estimator_.predict(X_train_lgb)
df_test["pp_t_lgbm"] = gs_unconstrained.best_estimator_.predict(X_test_lgb)

print(
    "training loss unconstrained LGBM:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)
print(
    "testing loss unconstrained LGBM:   {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# Step 4: Constrained model (monotone increasing in BonusMalus)
# Build monotone_constraints list matching feature order; 1 for BonusMalus, else 0.
monotone_constraints = [1 if col == "BonusMalus" else 0 for col in X_train_lgb.columns]

param_grid_constrained = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}

lgb_constrained = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    monotone_constraints=monotone_constraints,
    random_state=42,
)

gs_constrained = GridSearchCV(
    estimator=lgb_constrained,
    param_grid=param_grid_constrained,
    cv=3,
    n_jobs=-1,
    verbose=0,
)
gs_constrained.fit(X_train_lgb, y_train_t, sample_weight=w_train_t)

df_train["pp_t_lgbm_constrained"] = gs_constrained.best_estimator_.predict(X_train_lgb)
df_test["pp_t_lgbm_constrained"] = gs_constrained.best_estimator_.predict(X_test_lgb)

print(
    "training loss constrained LGBM:   {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)
print(
    "testing loss constrained LGBM:    {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# Step 5: Quick Gini comparison (optional inline; detailed curves below reuse pp_t_lgbm)
# Define lorenz_curve before we call gini_for to avoid NameError if execution stops earlier.
def lorenz_curve(y_true, y_pred, exposure):
    """Calculate Lorenz curve (cumulative claims vs exposure) ordered by predictions.

    Returns (cumulative_exposure_fraction, cumulative_claim_fraction).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cum_claims = np.cumsum(ranked_pure_premium * ranked_exposure)
    total_claims = cum_claims[-1]
    if total_claims == 0:
        cum_claims = np.zeros_like(cum_claims, dtype=float)
    else:
        cum_claims = cum_claims / total_claims
    cum_exposure = np.cumsum(ranked_exposure) / np.sum(ranked_exposure)
    return cum_exposure, cum_claims

def gini_for(pred_col):
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], df_test[pred_col], df_test["Exposure"]
    )
    return 1 - 2 * auc(ordered_samples, cum_claims)

print(
    "Gini (unconstrained LGBM):  {:.3f}".format(gini_for("pp_t_lgbm"))
)
print(
    "Gini (constrained LGBM):    {:.3f}".format(gini_for("pp_t_lgbm_constrained"))
)

# Interpretation placeholder
print("Monotonic constraint applied: BonusMalus forced to have non-decreasing effect on predicted pure premium.")

# ...existing code...
# %%
# Compare model performance using Lorenz curves and Gini index
# The Lorenz curve shows how well a model ranks policies by risk
# A perfect model would identify all high-claim policies first

# ...existing code...

def lorenz_curve(y_true, y_pred, exposure):
    """Calculate Lorenz curve for model evaluation.

    The Lorenz curve plots cumulative claim amount vs cumulative exposure fraction,
    ordered by predicted risk (ascending).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # Order policies from lowest to highest predicted risk
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]

    # Exposure-weighted cumulative claim amount (normalize to [0, 1])
    cum_claims = np.cumsum(ranked_pure_premium * ranked_exposure)
    total_claims = cum_claims[-1]
    if total_claims == 0:
        # If there are no claims, the Lorenz curve is flat; avoid division by zero
        cum_claims = np.zeros_like(cum_claims, dtype=float)
    else:
        cum_claims = cum_claims / total_claims

    # X-axis should be cumulative exposure fraction, not linspace
    cum_exposure = np.cumsum(ranked_exposure) / np.sum(ranked_exposure)

    return cum_exposure, cum_claims

# Create a figure to compare all three models
fig, ax = plt.subplots(figsize=(8, 8))

# Plot Lorenz curves for each model (skip if column missing)
for label, col in [
    ("LGBM", "pp_t_lgbm"),
    ("GLM Benchmark", "pp_t_glm1"),
    ("GLM Splines", "pp_t_glm2"),
]:
    if col in df_test.columns:
        ordered_samples, cum_claims = lorenz_curve(
            df_test["PurePremium"], df_test[col], df_test["Exposure"]
        )
        gini = 1 - 2 * auc(ordered_samples, cum_claims)
        ax.plot(ordered_samples, cum_claims, linestyle="-", label=f"{label} (Gini: {gini: .3f})")

# Oracle model: perfect predictions (y_pred == y_true)
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=f"Oracle (Gini: {gini: .3f})")

# Random baseline: diagonal
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Cumulative exposure fraction\n(ordered safest → riskiest)",
    ylabel="Cumulative fraction of total claim amount",
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
plt.show()
# ...existing code...
# %%
# =============================================================
# Glassbox Diagnostics: Understanding Monotonic Constraint Impact
# =============================================================
# We now visualize how the monotonicity constraint on BonusMalus affects
# predicted pure premiums compared to:
#   - Actual exposure-weighted pure premium (empirical)
#   - Unconstrained LGBM predictions
#   - Constrained LGBM predictions
# This helps confirm that the constraint removes accidental downward bumps.

def exposure_weighted_pred_curve(df_source, pred_col):
    """Return exposure-weighted average prediction per BonusMalus.

    For each BonusMalus bin:
        weighted_pred = sum(Exposure * prediction) / sum(Exposure)
    """
    return (
        df_source.groupby("BonusMalus").apply(
            lambda g: np.sum(g["Exposure"] * g[pred_col]) / np.sum(g["Exposure"])
        )
        .rename(pred_col)
        .reset_index()
    )

# Empirical curve (already computed earlier as bm_curve) renamed for clarity
empirical_curve = bm_curve.rename(columns={"weighted_pure_premium": "empirical"})

unconstrained_curve = exposure_weighted_pred_curve(df_test, "pp_t_lgbm") if "pp_t_lgbm" in df_test.columns else None
constrained_curve = exposure_weighted_pred_curve(df_test, "pp_t_lgbm_constrained") if "pp_t_lgbm_constrained" in df_test.columns else None

# Merge curves on BonusMalus for aligned plotting
curves_merged = empirical_curve.copy()
for extra in [unconstrained_curve, constrained_curve]:
    if extra is not None:
        curves_merged = curves_merged.merge(extra, on="BonusMalus", how="left")

plt.figure(figsize=(9,5))
plt.plot(curves_merged["BonusMalus"], curves_merged["empirical"], label="Empirical (Actual)", color="#2c7fb8")
if unconstrained_curve is not None:
    plt.plot(curves_merged["BonusMalus"], curves_merged["pp_t_lgbm"], label="Unconstrained LGBM", color="#fdae61")
if constrained_curve is not None:
    plt.plot(curves_merged["BonusMalus"], curves_merged["pp_t_lgbm_constrained"], label="Constrained LGBM", color="#d7191c")
plt.title("Exposure-weighted Predicted Pure Premium vs BonusMalus (Test Set)")
plt.xlabel("BonusMalus")
plt.ylabel("Exposure-weighted Pure Premium")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Difference plot to highlight constraint effect
if unconstrained_curve is not None and constrained_curve is not None:
    curves_merged["delta_constrained_vs_unconstrained"] = (
        curves_merged["pp_t_lgbm_constrained"] - curves_merged["pp_t_lgbm"]
    )
    plt.figure(figsize=(9,4))
    plt.plot(
        curves_merged["BonusMalus"],
        curves_merged["delta_constrained_vs_unconstrained"],
        marker="o",
        linestyle="-",
        color="#d95f02",
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Constrained minus Unconstrained Prediction (BonusMalus bins)")
    plt.xlabel("BonusMalus")
    plt.ylabel("Δ Pure Premium")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Scatter of individual policies: BonusMalus vs predicted pure premium (constrained)
if "pp_t_lgbm_constrained" in df_test.columns:
    plt.figure(figsize=(9,5))
    plt.scatter(
        df_test["BonusMalus"],
        df_test["pp_t_lgbm_constrained"],
        alpha=0.15,
        s=10,
        color="#d7191c",
        label="Constrained predictions",
    )
    # Overlay empirical curve for reference
    plt.plot(empirical_curve["BonusMalus"], empirical_curve["empirical"], color="#2c7fb8", linewidth=2, label="Empirical curve")
    plt.title("Constrained Predictions Scatter vs Empirical Curve")
    plt.xlabel("BonusMalus")
    plt.ylabel("Predicted Pure Premium")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# LightGBM feature importances (constrained)
if "pp_t_lgbm_constrained" in df_test.columns:
    feat_imp = pd.Series(
        gs_constrained.best_estimator_.feature_importances_, index=X_train_lgb.columns
    ).sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    top_k = feat_imp.head(15)[::-1]
    plt.barh(top_k.index, top_k.values, color="#6a3d9a")
    plt.title("Constrained LGBM Feature Importances (Top 15)")
    plt.xlabel("Importance (gain-based)")
    plt.tight_layout()
    plt.show()

# Glassbox interpretation text block
print("""
Glassbox Summary:
 - Constrained model enforces non-decreasing relationship in BonusMalus, smoothing accidental dips.
 - Difference plot (Δ) shows where unconstrained model violated monotonicity (positive corrections applied).
 - Scatter highlights variance within bins; constraint acts on aggregate directional trend, not per-point smoothing.
 - Feature importance chart reveals whether BonusMalus remains among top drivers post-constraint.
""")

# %%
# =============================================================
# Ex 2: Learning Curve for Constrained LGBM
# =============================================================
# Objective: Refit constrained LGBM with eval_set to capture train/test metric
# evolution across boosting rounds and visualize convergence.

from lightgbm import plot_metric as lgb_plot_metric

# Refit best constrained estimator with evaluation tracking.
# We copy parameters to avoid altering the GridSearchCV internal state.
best_constrained_params = gs_constrained.best_estimator_.get_params()
learning_lgbm = LGBMRegressor(**best_constrained_params)

from lightgbm import log_evaluation

learning_lgbm.fit(
    X_train_lgb,
    y_train_t,
    sample_weight=w_train_t,
    eval_set=[(X_train_lgb, y_train_t), (X_test_lgb, y_test_t)],
    eval_sample_weight=[w_train_t, w_test_t],
    eval_metric="tweedie",
    callbacks=[log_evaluation(100)],  # log every 100 rounds; remove or adjust as needed
)

# Store round-wise predictions (optional if we want custom curve)
evals_result = learning_lgbm.evals_result_

# Plot using built-in helper
plt.figure(figsize=(8,5))
lgb_plot_metric(learning_lgbm, metric="tweedie")
plt.title("Learning Curve (Tweedie Metric) – Constrained LGBM")
plt.tight_layout()
plt.show()

# Manual overlay (gain insight into over/underfitting): lower metric is better for tweedie
train_metric = evals_result["training"]["tweedie"]
test_metric = evals_result["valid_1"]["tweedie"]
rounds = range(1, len(train_metric) + 1)

plt.figure(figsize=(8,5))
plt.plot(rounds, train_metric, label="Train Tweedie", color="#1b9e77")
plt.plot(rounds, test_metric, label="Test Tweedie", color="#d95f02")
plt.xlabel("Boosting Round")
plt.ylabel("Tweedie Metric")
plt.title("Learning Curve (Train vs Test) – Constrained LGBM")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Simple interpretation block
print("""
Learning Curve Interpretation:
 - Parallel descent then flattening indicates convergence of the constrained model.
 - If test metric starts increasing while train keeps decreasing, it signals overfitting.
 - If both plateau early at high values, consider more trees (n_estimators) or lower learning_rate.
 - Monotonic constraint can slightly slow initial improvement but may reduce overfitting in later rounds.
Next tuning ideas:
 - Try learning_rate=0.03 with higher n_estimators (e.g., 800) for smoother convergence.
 - Add early_stopping_rounds=50 to automatically choose optimal boosting iteration.
""")

# =============================================================
# Final Analysis Summary (Helper)
# -------------------------------------------------------------
# This section summarizes the key model diagnostics produced above.
# Generated artifacts when script runs:
#   1. GLM benchmark (t_glm1) train/test deviance + total predicted claims
#   2. GLM splines (pp_t_glm2) train/test deviance
#   3. Unconstrained & constrained LGBM losses + Gini indices
#   4. Exposure-weighted BonusMalus empirical & predicted curves
#   5. Lorenz curves (GLM / Splines / LGBM / Oracle)
#   6. Learning curve for constrained LGBM
#   7. Feature importances (constrained LGBM)
# Call print_final_summary() after all cells to re-display core numeric metrics.
# =============================================================

def print_final_summary():
    """Print consolidated numeric metrics for quick review.
    Recomputes deviances from stored prediction columns.
    """
    lines = []
    # GLM benchmark
    if all(col in df_train.columns for col in ["pp_t_glm1"]):
        glm1_train = TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)/np.sum(w_train_t)
        glm1_test = TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)/np.sum(w_test_t)
        lines.append(f"GLM Benchmark deviance (train/test): {glm1_train:.4f} / {glm1_test:.4f}")
    # GLM splines
    if all(col in df_train.columns for col in ["pp_t_glm2"]):
        glm2_train = TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)/np.sum(w_train_t)
        glm2_test = TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)/np.sum(w_test_t)
        lines.append(f"GLM Splines deviance (train/test): {glm2_train:.4f} / {glm2_test:.4f}")
    # LGBM unconstrained & constrained
    def maybe_gini(pred_col):
        if pred_col in df_test.columns:
            ordered_samples, cum_claims = lorenz_curve(df_test["PurePremium"], df_test[pred_col], df_test["Exposure"])
            return 1 - 2 * auc(ordered_samples, cum_claims)
        return None
    if "pp_t_lgbm" in df_train.columns:
        lgbm_train = TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)/np.sum(w_train_t)
        lgbm_test = TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)/np.sum(w_test_t)
        gini_u = maybe_gini("pp_t_lgbm")
        lines.append(f"LGBM Unconstrained deviance (train/test): {lgbm_train:.4f} / {lgbm_test:.4f} | Gini: {gini_u:.3f}")
    if "pp_t_lgbm_constrained" in df_train.columns:
        lgbmc_train = TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)/np.sum(w_train_t)
        lgbmc_test = TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)/np.sum(w_test_t)
        gini_c = maybe_gini("pp_t_lgbm_constrained")
        lines.append(f"LGBM Constrained deviance (train/test): {lgbmc_train:.4f} / {lgbmc_test:.4f} | Gini: {gini_c:.3f}")
    # Learning curve last values
    if 'evals_result' in globals():
        train_metric_last = evals_result['training']['tweedie'][-1]
        test_metric_last = evals_result['valid_1']['tweedie'][-1]
        lines.append(f"Learning curve final Tweedie (train/test): {train_metric_last:.4f} / {test_metric_last:.4f}")
    print("\n=== Final Summary ===")
    for l in lines:
        print(l)
    print("====================\n")

#%%
# =============================================================
# Evaluation Metrics (Consolidated) – Using evaluate_predictions
# =============================================================
# We compute exposure-weighted metrics for each model's test predictions.
# Deviance uses the Tweedie distribution object (power=1.5) we fit earlier.

evaluation_rows = []

def _add_metrics(label, pred_col):
    if pred_col in df_test.columns:
        m = evaluate_predictions(
            y_true=df_test["PurePremium"],
            y_pred=df_test[pred_col],
            sample_weight=df_test["Exposure"],
            distribution=TweedieDist,
        )
        evaluation_rows.append({"model": label, **m})

_add_metrics("GLM Benchmark", "pp_t_glm1")
_add_metrics("GLM Splines", "pp_t_glm2")
_add_metrics("LGBM Unconstrained", "pp_t_lgbm")
_add_metrics("LGBM Constrained", "pp_t_lgbm_constrained")

if evaluation_rows:
    eval_df = pd.DataFrame(evaluation_rows)
    # Order columns for readability
    cols = ["model", "deviance", "gini", "mae", "total_actual", "total_predicted"]
    eval_df = eval_df[[c for c in cols if c in eval_df.columns]]
    print("\n=== Evaluation Metrics (Test Set) ===")
    print(eval_df.to_string(index=False, formatters={
        "deviance": lambda v: f"{v:.4f}" if pd.notnull(v) else "-",
        "gini": lambda v: f"{v:.3f}" if pd.notnull(v) else "-",
        "mae": lambda v: f"{v:.4f}" if pd.notnull(v) else "-",
        "total_actual": lambda v: f"{v:,.2f}",
        "total_predicted": lambda v: f"{v:,.2f}",
    }))
    print("====================================\n")

# Usage example (textual):
print("Usage: from ps3.evaluation import evaluate_predictions; metrics = evaluate_predictions(y_true, y_pred, exposure, TweedieDist)")

# =============================================================
# Extended Diagnostics – Visual Comparison of Model Behaviors
# =============================================================
# Charts to deepen understanding of each model's strengths & weaknesses:
#   1. Residual scatter (predicted vs residual) – pattern or heteroskedasticity
#   2. Residual distribution comparison – bias & spread
#   3. Calibration (decile bins) – predicted vs actual exposure-weighted pure premium
#   4. Error by BonusMalus – where pricing signal differs most
#   5. Claim lift curve – cumulative claim capture vs exposure (ranking power)
#   6. Partial dependence approximations – aggregated prediction vs BonusMalus/Density
#   7. Portfolio bias bars – total actual vs total predicted per model

model_defs = [
    ("GLM Benchmark", "pp_t_glm1"),
    ("GLM Splines", "pp_t_glm2"),
    ("LGBM Unconstrained", "pp_t_lgbm"),
    ("LGBM Constrained", "pp_t_lgbm_constrained"),
]

available_models = [(lbl, col) for lbl, col in model_defs if col in df_test.columns]

def _residuals(df_src, pred_col):
    return df_src["PurePremium"] - df_src[pred_col]

# 1. Residual scatter plots
plt.figure(figsize=(10, 6))
for i, (lbl, col) in enumerate(available_models, 1):
    ax = plt.subplot(2, (len(available_models)+1)//2, i)
    resid = _residuals(df_test, col)
    ax.scatter(df_test[col], resid, s=8, alpha=0.25)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"Residuals vs Pred ({lbl})", fontsize=9)
    ax.set_xlabel("Predicted Pure Premium")
    ax.set_ylabel("Residual (Actual - Pred)")
plt.tight_layout(); plt.show()

# 2. Residual distributions (hist)
plt.figure(figsize=(10,5))
bins = 50
for lbl, col in available_models:
    resid = _residuals(df_test, col)
    plt.hist(resid, bins=bins, alpha=0.4, label=lbl)
plt.title("Residual Distribution Comparison (Test)")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.legend(); plt.tight_layout(); plt.show()

# 3. Calibration by prediction deciles
def _calibration_table(df_src, pred_col, weight_col="Exposure", actual_col="PurePremium", q=10):
    """Return exposure-weighted actual vs predicted per prediction decile.

    Safer handling of interval naming to avoid KeyError ('index') differences across pandas versions.
    """
    preds = df_src[pred_col]
    try:
        deciles = pd.qcut(preds, q, duplicates="drop")
    except ValueError:
        deciles = pd.cut(preds, q)
    grp = df_src.groupby(deciles).apply(
        lambda g: pd.Series({
            "pred": np.sum(g[pred_col] * g[weight_col]) / np.sum(g[weight_col]),
            "actual": np.sum(g[actual_col] * g[weight_col]) / np.sum(g[weight_col]),
            "exposure": np.sum(g[weight_col])
        })
    )
    grp = grp.reset_index()  # first column is the interval bins (may be named 'index')
    # Uniform column names
    if "index" in grp.columns:
        grp.rename(columns={"index": "bin_interval"}, inplace=True)
    else:
        grp.rename(columns={grp.columns[0]: "bin_interval"}, inplace=True)
    # Order value for plotting (0..n-1)
    grp["bin_order"] = np.arange(len(grp))
    grp["bin_mid"] = grp["bin_interval"].apply(lambda iv: getattr(iv, 'mid', np.nan))
    return grp

plt.figure(figsize=(9,6))
for lbl, col in available_models:
    calib = _calibration_table(df_test, col)
    plt.plot(calib["bin_order"], calib["actual"], marker="o", linestyle="-", label=f"Actual ({lbl})")
    plt.plot(calib["bin_order"], calib["pred"], marker="x", linestyle="--", label=f"Pred ({lbl})")
plt.title("Calibration: Exposure-weighted Actual vs Predicted by Decile")
plt.xlabel("Prediction Decile (ascending)")
plt.ylabel("Pure Premium (Exposure-weighted)")
plt.legend(ncol=2, fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 4. Error by BonusMalus
bm_vals = sorted(df_test["BonusMalus"].unique())
plt.figure(figsize=(10,6))
for lbl, col in available_models:
    df_grp = df_test.groupby("BonusMalus").apply(
        lambda g: np.sum(np.abs(g["PurePremium"] - g[col]) * g["Exposure"]) / np.sum(g["Exposure"])
    ).rename(lbl)
    plt.plot(df_grp.index, df_grp.values, marker="o", linewidth=1, label=lbl)
plt.title("Exposure-weighted Absolute Error vs BonusMalus (Test)")
plt.xlabel("BonusMalus")
plt.ylabel("Weighted Absolute Error")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 5. Claim lift curve (cumulative claims captured)
plt.figure(figsize=(8,6))
for lbl, col in available_models:
    cum_exposure, cum_claims = lorenz_curve(df_test["PurePremium"], df_test[col], df_test["Exposure"])  # already normalized
    plt.plot(cum_exposure, cum_claims, label=lbl)
plt.plot([0,1],[0,1], linestyle="--", color="black", label="Random")
plt.title("Claim Lift Curve (Lorenz) – Test")
plt.xlabel("Cumulative Exposure Fraction")
plt.ylabel("Cumulative Claim Fraction")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 6. Partial dependence approximations (binning numeric variables)
def _partial_dependence(df_src, feature, pred_col, bins=12):
    binned = pd.qcut(df_src[feature], bins, duplicates="drop")
    return df_src.groupby(binned).apply(
        lambda g: np.sum(g[pred_col]*g["Exposure"]) / np.sum(g["Exposure"])  # exposure-weighted mean prediction
    ).reset_index().rename(columns={0: "pred"})

for feature in ["BonusMalus", "Density"]:
    plt.figure(figsize=(8,5))
    for lbl, col in available_models:
        try:
            pdp = _partial_dependence(df_test, feature, col)
            # Extract midpoint for ordering if interval
            pdp["mid"] = pdp.iloc[:,0].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)
            plt.plot(pdp["mid"], pdp["pred"], marker="o", linewidth=1, label=lbl)
        except Exception:
            continue
    plt.title(f"Approx Partial Dependence – {feature}")
    plt.xlabel(feature)
    plt.ylabel("Exposure-weighted Predicted Pure Premium")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 7. Portfolio bias bars
plt.figure(figsize=(9,5))
bar_labels = []
actual_vals = []
pred_vals = []
for lbl, col in available_models:
    bar_labels.append(lbl)
    actual_vals.append(np.sum(df_test["PurePremium"] * df_test["Exposure"]))
    pred_vals.append(np.sum(df_test[col] * df_test["Exposure"]))
width = 0.35
idx = np.arange(len(bar_labels))
plt.bar(idx - width/2, actual_vals, width, label="Actual", color="#2c7fb8")
plt.bar(idx + width/2, pred_vals, width, label="Predicted", color="#fdae61")
plt.xticks(idx, bar_labels, rotation=20)
plt.title("Portfolio Claim Totals (Test)")
plt.ylabel("Total Claim Amount (Exposure-weighted)")
plt.legend(); plt.tight_layout(); plt.show()

print("Extended diagnostics generated: residuals, calibration, error curves, lift, partial dependence, portfolio bias.")

# Uncomment to print immediately when script runs:
# print_final_summary()

"""
The Lorenz curves show modest ranking power: Gini ≈ 0.310 (GLM Benchmark) and ≈ 0.307 (GLM Splines). 
Both are well above random (0) but far from oracle (~0.983).

The benchmark slightly edges the spline model; differences are minimal, so splines did not materially 
improve ranking on this test set.

The sharp rise near the right end indicates claims are highly concentrated in the riskiest tail 
(top ~10–20% of exposure).

Curves close to the diagonal imply limited discrimination overall; models capture some risk ordering, 
but much of claim variability remains unexplained.
"""
# %%
