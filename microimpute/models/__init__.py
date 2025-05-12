"""Imputation Models

This module contains different imputation models for benchmarking.
"""

# Import base classes
from .imputer import Imputer, ImputerResults

try:
    from .matching import Matching
except ImportError:
    pass

# Import specific model implementations
from .ols import OLS
from .qrf import QRF
from .quantreg import QuantReg
