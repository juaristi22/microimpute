import statsmodels.api as sm
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Collection, Any


class QuantReg:
    """
    Quantile Regression model for imputation.
    
    This model uses statsmodels' QuantReg implementation to
    directly predict specific quantiles.
    """
    __name__ = "QuantReg"
    
    def __init__(self):
        """
        Initialize the Quantile Regression model.
        """
        self.models: Dict[float, Any] = {}
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None
        
    def fit(self, X: pd.DataFrame, predictors: List[str], imputed_variables: List[str], quantiles: List[float]) -> 'QuantReg':
        """
        Fit the Quantile Regression model to the training data.
        
        :param X: DataFrame containing the training data.
        :type X: pd.DataFrame
        :param predictors: List of column names to use as predictors.
        :type predictors: List[str]
        :param imputed_variables: List of column names to impute.
        :type imputed_variables: List[str]
        :param quantiles: List of quantiles to fit models for.
        :type quantiles: List[float]
        :returns: The fitted model instance.
        :rtype: QuantReg
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        
        Y = X[imputed_variables]
        X_with_const = sm.add_constant(X[predictors])
        
        for q in quantiles:
            self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)
            
        return self
        
    def predict(self, test_X: pd.DataFrame, quantiles: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        """
        Predict values at specified quantiles using the Quantile Regression model.
        
        :param test_X: DataFrame containing the test data.
        :type test_X: pd.DataFrame
        :param quantiles: List of quantiles to predict. If None, uses the quantiles
                        from training.
        :type quantiles: Optional[List[float]]
        :returns: Dictionary mapping quantiles to predicted values.
        :rtype: Dict[float, np.ndarray]
        :raises ValueError: If a requested quantile was not fitted during training.
        """
        imputations: Dict[float, np.ndarray] = {}
        test_X_with_const = sm.add_constant(test_X[self.predictors])
        
        if quantiles is None:
            quantiles = list(self.models.keys())
            
        for q in quantiles:
            if q not in self.models:
                raise ValueError(f"Model for quantile {q} not fitted. Available quantiles: {list(self.models.keys())}")
                
            imputation = self.models[q].predict(test_X_with_const)
            imputations[q] = imputation
            
        return imputations
