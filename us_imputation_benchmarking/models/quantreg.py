import statsmodels.api as sm


class QuantReg:
    """
    Quantile Regression model for imputation.
    
    This model uses statsmodels' QuantReg implementation to
    directly predict specific quantiles.
    """
    
    def __init__(self):
        """Initialize the Quantile Regression model."""
        self.models = {}
        self.predictors = None
        self.imputed_variables = None
        
    def fit(self, X, predictors, imputed_variables, quantiles):
        """
        Fit the Quantile Regression model to the training data.
        
        Args:
            X: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            quantiles: List of quantiles to fit models for.
            
        Returns:
            self: The fitted model instance.
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        
        Y = X[imputed_variables]
        X_with_const = sm.add_constant(X[predictors])
        
        for q in quantiles:
            self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)
            
        return self
        
    def predict(self, test_X, quantiles=None):
        """
        Predict values at specified quantiles using the Quantile Regression model.
        
        Args:
            test_X: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses the quantiles
                from training. Defaults to None.
            
        Returns:
            Dict: Mapping of quantiles to predicted values.
        """
        imputations = {}
        test_X_with_const = sm.add_constant(test_X[self.predictors])
        
        if quantiles is None:
            quantiles = self.models.keys()
            
        for q in quantiles:
            if q not in self.models:
                raise ValueError(f"Model for quantile {q} not fitted. Available quantiles: {list(self.models.keys())}")
                
            imputation = self.models[q].predict(test_X_with_const)
            imputations[q] = imputation
            
        return imputations
