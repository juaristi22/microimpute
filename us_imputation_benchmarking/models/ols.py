import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Union, Optional


class OLS:
    """
    Ordinary Least Squares regression model for imputation.

    This model predicts different quantiles by assuming normally
    distributed residuals.
    """
    def __init__(self):
        """
        Initialize the OLS model.
        """
        self.model = None
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None

    def fit(
        self,
        X: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> "OLS":
        """
        Fit the OLS model to the training data.

        :param X: DataFrame containing the training data.
        :type X: pd.DataFrame
        :param predictors: List of column names to use as predictors.
        :type predictors: List[str]
        :param imputed_variables: List of column names to impute.
        :type imputed_variables: List[str]
        :returns: The fitted model instance.
        :rtype: OLS
        """
        self.predictors = predictors
        self.imputed_variables = imputed_variables

        Y = X[imputed_variables]
        X_with_const = sm.add_constant(X[predictors])

        self.model = sm.OLS(Y, X_with_const).fit()
        return self

    def predict(
        self, test_X: pd.DataFrame, quantiles: List[float]
    ) -> Dict[float, np.ndarray]:
        """
        Predict values at specified quantiles using the OLS model.

        :param test_X: DataFrame containing the test data.
        :type test_X: pd.DataFrame
        :param quantiles: List of quantiles to predict.
        :type quantiles: List[float]
        :returns: Dictionary mapping quantiles to predicted values.
        :rtype: Dict[float, np.ndarray]
        """
        imputations: Dict[float, np.ndarray] = {}
        test_X_with_const = sm.add_constant(test_X[self.predictors])

        for q in quantiles:
            imputation = self._predict_quantile(test_X_with_const, q)
            imputations[q] = imputation

        return imputations

    def _predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        """
        Predict values at a specified quantile.

        :param X: Feature matrix with constant.
        :type X: pd.DataFrame
        :param q: Quantile to predict.
        :type q: float
        :returns: Array of predicted values at the specified quantile.
        :rtype: np.ndarray
        """
        mean_pred = self.model.predict(X)
        se = np.sqrt(self.model.scale)
        return mean_pred + norm.ppf(q) * se
