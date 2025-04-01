"""US Imputation Benchmarking Package

A package for benchmarking different imputation methods using US data.
"""

__version__ = "0.1.0"

# Import main models and utilities
from us_imputation_benchmarking.models import (
    Imputer, ImputerResults,
    OLS, QRF, QuantReg, Matching
)

# Import data handling functions
from us_imputation_benchmarking.comparisons.data import (
    prepare_scf_data, preprocess_data
)

# Import evaluation modules
from us_imputation_benchmarking.evaluations.cross_validation import cross_validate_model
from us_imputation_benchmarking.evaluations.train_test_performance import plot_train_test_performance

# Import comparison utilities
from us_imputation_benchmarking.comparisons.quantile_loss import quantile_loss
from us_imputation_benchmarking.comparisons.plot import plot_loss_comparison
from us_imputation_benchmarking.comparisons.imputations import get_imputations

# Main configuration
from us_imputation_benchmarking.config import RANDOM_STATE, QUANTILES, PLOT_CONFIG, validate_config