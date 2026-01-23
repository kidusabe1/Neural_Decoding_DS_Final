"""
Evaluation metrics module.
"""

from neural_decoding.evaluation.metrics import get_R2, get_rho, evaluate_decoder


__all__ = [
    "evaluate_decoder",
    "get_rmse",
    "get_r2_score",
    "get_pearson_correlation",
]
