"""MicroImpute Package

A package for benchmarking different imputation methods using microdata.
"""

__version__ = "0.1.0"

# Import data handling functions
from microimpute.comparisons.data import prepare_scf_data, preprocess_data
from microimpute.comparisons.imputations import get_imputations

# Import comparison utilities
from microimpute.comparisons.quantile_loss import (
    compare_quantile_loss,
    compute_quantile_loss,
    quantile_loss,
)

# Main configuration
from microimpute.config import (
    PLOT_CONFIG,
    QUANTILES,
    RANDOM_STATE,
    VALIDATE_CONFIG,
)

# Import evaluation modules
from microimpute.evaluations.cross_validation import cross_validate_model

# Import main models and utilities
from microimpute.models import (
    OLS,
    QRF,
    Imputer,
    ImputerResults,
    Matching,
    QuantReg,
)

# Import visualization modules
from microimpute.visualizations.plotting import (
    method_comparison_results,
    model_performance_results,
)
