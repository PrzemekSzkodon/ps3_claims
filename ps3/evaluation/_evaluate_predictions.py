"""
Evaluation metrics for insurance claim prediction models.

This module provides a reusable function to compute various performance metrics
without having to recompute them from scratch every time.
"""

import numpy as np
from sklearn.metrics import auc


def evaluate_predictions(y_true, y_pred, sample_weight, distribution=None):
    """
    Evaluate model predictions with insurance-specific metrics.
    
    This function computes several metrics commonly used in insurance modeling:
    - Deviance (if a distribution is provided)
    - Gini coefficient (measures how well the model ranks policies by risk)
    - Mean Absolute Error (exposure-weighted)
    - Total actual vs predicted claims
    
    Parameters
    ----------
    y_true : array-like
        Actual outcome values (e.g., pure premium per policy)
    y_pred : array-like
        Model predictions (same shape as y_true)
    sample_weight : array-like
        Sample weights, typically exposure (fraction of year policy was active)
    distribution : object, optional
        Distribution object with a .deviance() method (e.g., TweedieDistribution).
        If provided, computes weighted deviance. If None, deviance is skipped.
        
    Returns
    -------
    dict
        Dictionary containing computed metrics:
        - 'deviance': Weighted deviance per unit exposure (if distribution provided)
        - 'gini': Gini coefficient from Lorenz curve (0=random, 1=perfect)
        - 'mae': Mean absolute error, weighted by exposure
        - 'total_actual': Sum of actual claim amounts (y_true * exposure)
        - 'total_predicted': Sum of predicted claim amounts (y_pred * exposure)
        
    Examples
    --------
    >>> from glum import TweedieDistribution
    >>> dist = TweedieDistribution(1.5)
    >>> metrics = evaluate_predictions(
    ...     y_true=y_test,
    ...     y_pred=model.predict(X_test),
    ...     sample_weight=exposure_test,
    ...     distribution=dist
    ... )
    >>> print(f"Gini: {metrics['gini']:.3f}")
    """
    # Convert inputs to numpy arrays for consistent operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sample_weight = np.asarray(sample_weight)
    
    # Initialize dictionary to store all metrics
    metrics = {}
    
    # ===== 1. Deviance (if distribution provided) =====
    # Deviance measures how well the model fits the data for a specific distribution.
    # Lower deviance = better fit. We normalize by total exposure.
    if distribution is not None:
        dev = distribution.deviance(y_true, y_pred, sample_weight=sample_weight)
        metrics['deviance'] = dev / np.sum(sample_weight)
    
    # ===== 2. Gini Coefficient via Lorenz Curve =====
    # The Gini coefficient measures how well the model ranks policies by risk.
    # Steps:
    #   a) Sort policies by predicted risk (low to high)
    #   b) Calculate cumulative claim amounts (weighted by exposure)
    #   c) Compare to perfect ranking (oracle) and random baseline
    
    # Sort policies from safest (lowest prediction) to riskiest (highest prediction)
    ranking = np.argsort(y_pred)
    ranked_exposure = sample_weight[ranking]
    ranked_pure_premium = y_true[ranking]
    
    # Calculate cumulative fraction of total claims (y-axis of Lorenz curve)
    cumulative_claims = np.cumsum(ranked_pure_premium * ranked_exposure)
    total_claims = cumulative_claims[-1]
    
    # Handle edge case: if no claims at all, set to zeros
    if total_claims > 0:
        cumulative_claims = cumulative_claims / total_claims  # Normalize to [0, 1]
    else:
        cumulative_claims = np.zeros_like(cumulative_claims, dtype=float)
    
    # Calculate cumulative fraction of exposure (x-axis of Lorenz curve)
    cumulative_exposure = np.cumsum(ranked_exposure) / np.sum(ranked_exposure)
    
    # Gini = 1 - 2*AUC (area under Lorenz curve)
    # Gini closer to 1 = better ranking, closer to 0 = random
    gini = 1 - 2 * auc(cumulative_exposure, cumulative_claims)
    metrics['gini'] = gini
    
    # ===== 3. Mean Absolute Error (weighted) =====
    # MAE measures average prediction error, weighted by exposure.
    # Policies with longer exposure contribute more to the metric.
    mae = np.average(np.abs(y_true - y_pred), weights=sample_weight)
    metrics['mae'] = mae
    
    # ===== 4. Total Claims (Actual vs Predicted) =====
    # Useful for checking if model is over/under-predicting total claims.
    # We multiply pure premium by exposure to get actual claim amounts.
    metrics['total_actual'] = np.sum(y_true * sample_weight)
    metrics['total_predicted'] = np.sum(y_pred * sample_weight)
    
    return metrics


