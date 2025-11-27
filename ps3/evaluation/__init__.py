"""Evaluation subpackage exports.

Provides a clean import surface:

	from ps3.evaluation import evaluate_predictions

Avoid importing analysis scripts here to keep dependency one-way (library -> analyses).
"""

from ._evaluate_predictions import evaluate_predictions

__all__ = ["evaluate_predictions"]

