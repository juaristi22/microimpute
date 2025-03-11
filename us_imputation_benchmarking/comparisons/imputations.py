import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any, Type, Callable, Tuple
from us_imputation_benchmarking.config import QUANTILES
from us_imputation_benchmarking.models.quantreg import QuantReg


def get_imputations(
    model_classes: List[Type],
    X: pd.DataFrame,
    test_X: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    quantiles: Optional[List[float]] = QUANTILES,
) -> Dict[str, Dict[float, Union[np.ndarray, pd.DataFrame]]]:
    """
    Generate imputations using multiple model classes for the specified variables.

    :param model_classes: List of model classes to use (e.g., QRF, OLS, QuantReg, Matching).
    :type model_classes: List[Type]
    :param X: Training data containing predictors and variables to impute.
    :type X: pd.DataFrame
    :param test_X: Test data on which to make imputations.
    :type test_X: pd.DataFrame
    :param predictors: Names of columns to use as predictors.
    :type predictors: List[str]
    :param imputed_variables: Names of columns to impute.
    :type imputed_variables: List[str]
    :returns: Nested dictionary mapping method names to dictionaries mapping quantiles to imputations.
    :rtype: Dict[str, Dict[float, Union[np.ndarray, pd.DataFrame]]]
    """
    method_imputations: Dict[str, Dict[float, Any]] = {}

    for model_class in model_classes:
        model_name = model_class.__name__
        method_imputations[model_name] = {}

        # Instantiate the model
        model = model_class()

        # Handle QuantReg which needs quantiles during fitting
        if model_class == QuantReg:
            model.fit(X, predictors, imputed_variables, quantiles)
        else:
            model.fit(X, predictors, imputed_variables)

        # Get predictions
        imputations = model.predict(test_X, quantiles)
        method_imputations[model_name] = imputations

    return method_imputations
