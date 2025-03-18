from us_imputation_benchmarking.comparisons.data import preprocess_data, prepare_scf_data
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.config import RANDOM_STATE, QUANTILES
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np 
from us_imputation_benchmarking.evaluations.cross_validation import (
    cross_validate_model,
)
from us_imputation_benchmarking.evaluations.train_test_performance import (
    plot_train_test_performance,
)

# Shrink down the data by sampling
data, PREDICTORS, IMPUTED_VARIABLES = prepare_scf_data(full_data=True)
data = data.sample(frac=0.01, random_state=RANDOM_STATE)

ols_results = cross_validate_model(
        OLS, data, PREDICTORS, IMPUTED_VARIABLES
    )
ols_results.to_csv("ols_results.csv")

assert not ols_results.isna().any().any()

#plot_train_test_performance(ols_results, save_path="ols_train_test_performance.png")


# Test Method on iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
imputed_variables = ['petal width (cm)']

iris_df = iris_df[predictors + imputed_variables]

def test_ols_example(data=iris_df, 
                    predictors=predictors, 
                    imputed_variables=imputed_variables,
                    quantiles=QUANTILES):
        """
        Example of how to use the OLS imputer model.
        
        This example demonstrates:
        - Initializing an OLS model
        - Fitting the model to training data
        - Predicting quantiles on test data
        - How OLS models assume normally distributed residuals
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        X_train, X_test = preprocess_data(data)
        
        # Initialize OLS model
        model = OLS()
        
        # Fit the model
        model.fit(X_train, predictors, imputed_variables)
        
        # Predict at multiple quantiles
        predictions = model.predict(X_test, quantiles)
        
        # Check structure of predictions
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == set(quantiles)
        
        # Demonstrate how OLS uses normal distribution assumption
        median_pred = predictions[0.5]
        q10_pred = predictions[0.1]
        q90_pred = predictions[0.9]
        
        # The difference between q90 and median should approximately equal
        # the difference between median and q10 for OLS (symmetric distribution)
        upper_diff = q90_pred - median_pred
        lower_diff = median_pred - q10_pred
        
        # Allow some numerical error
        np.testing.assert_allclose(
            upper_diff.mean(), 
            lower_diff.mean(), 
            rtol=0.1,
            err_msg="OLS should have symmetric quantile predictions around the median"
        )