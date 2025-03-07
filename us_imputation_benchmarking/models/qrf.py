from us_imputation_benchmarking.utils import qrf
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union


class QRF:
    """
    Quantile Random Forest model for imputation.
    
    This model uses a Quantile Random Forest to predict quantiles.
    The underlying QRF implementation is from utils.qrf.
    """
    
    def __init__(self, seed: int = 0):
        """
        Initialize the QRF model.
        
        :param seed: Random seed for reproducibility.
        :type seed: int
        """
        self.qrf = qrf.QRF(seed=seed)
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None
        
    def fit(self, X: pd.DataFrame, predictors: List[str], imputed_variables: List[str], **qrf_kwargs: Any) -> 'QRF':
        """
        Fit the QRF model to the training data.
        
        :param X: DataFrame containing the training data.
        :type X: pd.DataFrame
        :param predictors: List of column names to use as predictors.
        :type predictors: List[str]
        :param imputed_variables: List of column names to impute.
        :type imputed_variables: List[str]
        :param qrf_kwargs: Additional keyword arguments to pass to QRF.
        :type qrf_kwargs: Any
        :returns: The fitted model instance.
        :rtype: QRF
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        
        self.qrf.fit(X[predictors], X[imputed_variables], **qrf_kwargs)
        return self
        
    def predict(self, test_X: pd.DataFrame, quantiles: List[float]) -> Dict[float, np.ndarray]:
        """
        Predict values at specified quantiles using the QRF model.
        
        :param test_X: DataFrame containing the test data.
        :type test_X: pd.DataFrame
        :param quantiles: List of quantiles to predict.
        :type quantiles: List[float]
        :returns: Dictionary mapping quantiles to predicted values.
        :rtype: Dict[float, np.ndarray]
        """
        imputations: Dict[float, np.ndarray] = {}
        
        for q in quantiles:
            imputation = self.qrf.predict(test_X[self.predictors], mean_quantile=q)
            imputations[q] = imputation
            
        return imputations
