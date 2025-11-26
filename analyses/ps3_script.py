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
# Explore the data: check dimensions and preview the first few rows
df_train.head()  # Displays the first 5 rows to understand the data structure
df_test.head()  # Displays the first 5 rows to understand the data structure

plt.figure(figsize=(10, 6))
plt.hist(df_test["pp_t_glm1"], bins=50, alpha=0.5, label="Train Predictions")
plt.xlabel("Predicted Pure Premium")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Pure Premiums")
plt.legend()
plt.show()

# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Build an improved model using splines for numeric variables
# Splines allow non-linear relationships (e.g., risk doesn't increase linearly with BonusMalus)
numeric_cols = ["BonusMalus", "Density"]

# ColumnTransformer applies different preprocessing to different column types
preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here (StandardScaler + SplineTransformer for numeric_cols)
        # OneHotEncoder converts categorical variables to binary columns (one per category)
        # drop="first" removes one category to avoid multicollinearity
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
# Set output format to pandas DataFrame (easier to work with than arrays)
preprocessor.set_output(transform="pandas")

# Pipeline chains preprocessing and modeling steps together
model_pipeline = Pipeline(
    # TODO: Define pipeline steps here
    # Should be: [("preprocessor", preprocessor), ("estimate", GLM model)]
)

# Display the pipeline structure to verify all steps are configured correctly
model_pipeline

# Test that preprocessing works: fit and transform training data
# [:-1] means "all steps except the last one" (i.e., just preprocessing, not the model)
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

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
# TODO: Let's use a GBM (Gradient Boosting Machine) instead as an estimator.
# GBMs often outperform GLMs because they can capture complex non-linear patterns
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.
#    Hint: Use objective="tweedie" to match our Tweedie distribution assumption

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Hyperparameter tuning finds the best combination of model settings
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
# GridSearchCV tries all combinations of parameters and picks the best based on cross-validation
cv = GridSearchCV(
    # TODO: Add estimator, param_grid, cv (cross-validation folds), and scoring metric
)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Compare model performance using Lorenz curves and Gini index
# The Lorenz curve shows how well a model ranks policies by risk
# A perfect model would identify all high-claim policies first

# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    """Calculate Lorenz curve for model evaluation.
    
    The Lorenz curve plots cumulative claims vs cumulative population fraction,
    ordered by predicted risk. Better models have curves further from the diagonal.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # Order policies from lowest to highest predicted risk
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    
    # Calculate cumulative claim amounts (as fraction of total)
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]  # Normalize to [0, 1]
    
    # Create evenly spaced points for x-axis (fraction of policies)
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


# Create a figure to compare all three models
fig, ax = plt.subplots(figsize=(8, 8))

# Plot Lorenz curves for each model
# The further the curve is from the diagonal, the better the model ranks policies
for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    # Gini index: ranges from 0 (random) to 1 (perfect). Higher is better.
    # It measures the area between the Lorenz curve and the diagonal
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: perfect predictions (y_pred == y_true)
# This represents the theoretical best possible performance
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline: diagonal line representing random predictions
# Any useful model should be above this line
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
