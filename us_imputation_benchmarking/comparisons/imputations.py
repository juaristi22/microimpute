from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.models.quantreg import QuantReg
from us_imputation_benchmarking.models.matching import Matching
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any, Type, Callable, Tuple


def get_imputations(
    model_classes: List[Type], 
    X: pd.DataFrame, 
    test_X: pd.DataFrame, 
    predictors: List[str], 
    imputed_variables: List[str],
    quantiles: Optional[List[float]] = None,
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
    # Set default quantiles if not provided
    if quantiles is None:
        QUANTILES: List[float] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    method_imputations: Dict[str, Dict[float, Any]] = {}
    
    for model_class in model_classes:
        model_name = model_class.__name__
        method_imputations[model_name] = {}
        
        # Instantiate the model
        model = model_class()
        
        # Handle QuantReg which needs quantiles during fitting
        if model_name == "QuantReg":
            model.fit(X, predictors, imputed_variables, QUANTILES)
        else:
            model.fit(X, predictors, imputed_variables)
            
        # Get predictions
        imputations = model.predict(test_X, QUANTILES)
        method_imputations[model_name] = imputations
        
    return method_imputations
