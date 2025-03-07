import statsmodels.api as sm
import numpy as np
from scipy.stats import norm


class OLS:
    """
    Ordinary Least Squares regression model for imputation.
    
    This model predicts different quantiles by assuming normally 
    distributed residuals.
    """
    
    def __init__(self):
        """Initialize the OLS model."""
        self.model = None
        self.predictors = None
        self.imputed_variables = None
        
    def fit(self, X, predictors, imputed_variables):
        """
        Fit the OLS model to the training data.
        
        Args:
            X: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            
        Returns:
            self: The fitted model instance.
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        
        Y = X[imputed_variables]
        X_with_const = sm.add_constant(X[predictors])
        
        self.model = sm.OLS(Y, X_with_const).fit()
        return self
        
    def predict(self, test_X, quantiles):
        """
        Predict values at specified quantiles using the OLS model.
        
        Args:
            test_X: DataFrame containing the test data.
            quantiles: List of quantiles to predict.
            
        Returns:
            Dict: Mapping of quantiles to predicted values.
        """
        imputations = {}
        test_X_with_const = sm.add_constant(test_X[self.predictors])
        
        for q in quantiles:
            imputation = self._predict_quantile(test_X_with_const, q)
            imputations[q] = imputation
            
        return imputations
    
    def _predict_quantile(self, X, q):
        """
        Predict values at a specified quantile.
        
        Args:
            X: Feature matrix with constant.
            q: Quantile to predict.
            
        Returns:
            Array of predicted values at the specified quantile.
        """
        mean_pred = self.model.predict(X)
        se = np.sqrt(self.model.scale)
        return mean_pred + norm.ppf(q) * se


# Legacy functions for backward compatibility
def ols_quantile(m, X, q):
    """
    Predict values at a specified quantile using an OLS model.
    
    Args:
        m: OLS model.
        X: X matrix.
        q: Quantile.
        
    Returns:
        Array of predicted values at the specified quantile.
    """
    mean_pred = m.predict(X)
    se = np.sqrt(m.scale)
    return mean_pred + norm.ppf(q) * se


def impute_ols(X, test_X, predictors, imputed_variables, quantiles):
    """
    Legacy function for OLS imputation.
    
    Args:
        X: Training data.
        test_X: Test data.
        predictors: List of predictor column names.
        imputed_variables: List of column names to impute.
        quantiles: List of quantiles to predict.
        
    Returns:
        Dict: Mapping of quantiles to predicted values.
    """
    model = OLS()
    model.fit(X, predictors, imputed_variables)
    return model.predict(test_X, quantiles)