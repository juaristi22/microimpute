from us_imputation_benchmarking.utils import qrf
import numpy as np


class QRF:
    """
    Quantile Random Forest model for imputation.
    
    This model uses a Quantile Random Forest to predict quantiles.
    The underlying QRF implementation is from utils.qrf.
    """
    
    def __init__(self, seed=0):
        """
        Initialize the QRF model.
        
        Args:
            seed: Random seed for reproducibility. Defaults to 0.
        """
        self.qrf = qrf.QRF(seed=seed)
        self.predictors = None
        self.imputed_variables = None
        
    def fit(self, X, predictors, imputed_variables, **qrf_kwargs):
        """
        Fit the QRF model to the training data.
        
        Args:
            X: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **qrf_kwargs: Additional keyword arguments to pass to QRF.
            
        Returns:
            self: The fitted model instance.
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        
        self.qrf.fit(X[predictors], X[imputed_variables], **qrf_kwargs)
        return self
        
    def predict(self, test_X, quantiles):
        """
        Predict values at specified quantiles using the QRF model.
        
        Args:
            test_X: DataFrame containing the test data.
            quantiles: List of quantiles to predict.
            
        Returns:
            Dict: Mapping of quantiles to predicted values.
        """
        imputations = {}
        
        for q in quantiles:
            imputation = self.qrf.predict(test_X[self.predictors], mean_quantile=q)
            imputations[q] = imputation
            
        return imputations


# Legacy function for backward compatibility
def impute_qrf(X, test_X, predictors, imputed_variables, quantiles):
    """
    Legacy function for QRF imputation.
    
    Args:
        X: Training data.
        test_X: Test data.
        predictors: List of predictor column names.
        imputed_variables: List of column names to impute.
        quantiles: List of quantiles to predict.
        
    Returns:
        Dict: Mapping of quantiles to predicted values.
    """
    model = QRF()
    model.fit(X, predictors, imputed_variables)
    return model.predict(test_X, quantiles)