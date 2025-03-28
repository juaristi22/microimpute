from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import logging


class Imputer(ABC):
    """
    Abstract base class for imputation models.

    All imputation models should inherit from this class and implement
    the required methods.
    """

    def __init__(self) -> None:
        """Initialize the imputer model."""
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None
        self.logger = logging.getLogger(__name__)

    def _validate_data(self, 
        data: pd.DataFrame, 
        columns: List[str]
    ) -> None:
        """Validate that all required columns are in the data.

        Args:
            data: DataFrame to validate
            columns: Column names that should be present

        Raises:
            ValueError: If any columns are missing from the data
        """
        missing_columns: Set[str] = set(columns) - set(data.columns)
        if missing_columns:
            error_msg = f"Missing columns in data: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Any,
    ) -> "Imputer":
        """Fit the model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **kwargs: Additional model-specific parameters.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            RuntimeError: If model fitting fails.
            NotImplementedError: If method is not implemented by subclass.
        """
        try:
            # Validate data
            self._validate_data(X_train, predictors + imputed_variables)
        except Exception as e:
            raise ValueError(f"Invalid input data for model: {str(e)}") from e
        
        # Save predictors and imputed variables
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        
        # Defer actual training to subclass with all parameters
        self._fit(X_train, predictors, imputed_variables, **kwargs)
        return self

    @abstractmethod
    def _fit(
        self, 
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Any,
    ) -> None:
        """Actual model-fitting logic (overridden in method subclass).
        
        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **kwargs: Additional model-specific parameters.
            
        Raises:
            ValueError: If specific model parameters are invalid.
            RuntimeError: If model fitting fails.
        """
        raise NotImplementedError("Subclasses must implement `_fit`")

    @abstractmethod
    def predict(
        self, X_test: pd.DataFrame, 
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values at specified quantiles.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If model is not fitted or input data is invalid.
            RuntimeError: If prediction fails.
            NotImplementedError: If method is not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the predict method")
