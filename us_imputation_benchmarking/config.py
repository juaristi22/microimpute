"""
Configuration module for US imputation benchmarking.

This module centralizes all constants and configuration parameters used across
the package.
"""
from typing import Any, Dict, List
import pandas as pd
from pydantic import ConfigDict, BaseModel, field_validator
from us_imputation_benchmarking.utils.logging_utils import configure_logging

# Define a configuration for pydantic validation that allows 
# arbitrary types like pd.DataFrame
validate_config = ConfigDict(arbitrary_types_allowed=True)

# Function to validate data frames are not empty
def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame is not None or empty."""
    if df is None or df.empty:
        raise ValueError("Data must not be None or empty")
    return df

class DataFrameModel(BaseModel):
    """
    A model enforcing that 'df' must be a non-empty pandas DataFrame.
    """
    model_config = validate_config

    df: pd.DataFrame

    @field_validator("df")
    def ensure_nonempty(cls, v: pd.DataFrame) -> pd.DataFrame:
        return validate_dataframe(v)

# Logging configuration
configure_logging()

# Data configuration
VALID_YEARS: List[int] = [
    1989,
    1992,
    1995,
    1998,
    2001,
    2004,
    2007,
    2010,
    2013,
    2016,
    2019,
]

train_size: float = 0.8
test_size: float = 0.2

# Analysis configuration
QUANTILES: List[float] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

# Random state for reproducibility
RANDOM_STATE: int = 42

# Model parameters
DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "qrf": {},
    "quantreg": {},
    "ols": {},
    "matching": {},
}

# Plotting configuration
PLOT_CONFIG: Dict[str, Any] = {
    "width": 1000,
    "height": 600,
    "colors": {},
}
