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