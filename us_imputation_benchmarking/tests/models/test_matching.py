from us_imputation_benchmarking.comparisons.data import preprocess_data, prepare_scf_data
from us_imputation_benchmarking.models.matching import Matching
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

matching_results = cross_validate_model(
        Matching, data, PREDICTORS, IMPUTED_VARIABLES
    )
matching_results.to_csv("matching_results.csv")

assert not matching_results.isna().any().any()

#plot_train_test_performance(matching_results, save_path="matching_train_test_performance.png")


# Test Method on iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
imputed_variables = ['petal width (cm)']

iris_df = iris_df[predictors + imputed_variables]

def test_matching_example(data=iris_df, 
                    predictors=predictors, 
                    imputed_variables=imputed_variables, 
                    quantiles = QUANTILES):
        """
        Example of how to use the Statistical Matching imputer model.
        
        This example demonstrates:
        - Initializing a Matching model
        - Fitting the model to donor data
        - Predicting values for recipient data
        - How matching uses nearest neighbors for imputation
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        X_train, X_test = preprocess_data(data)
        
        # Initialize Matching model
        model = Matching()
        
        # Fit the model (stores donor data)
        model.fit(X_train, predictors, imputed_variables)
        
        # Predict for the test data
        # For matching, quantiles don't have the same meaning as in regression
        # The same matched value is used for all quantiles
        quantiles = [0.5]  # Just use one quantile for simplicity
        predictions = model.predict(X_test, quantiles)
        
        # Check structure of predictions
        assert isinstance(predictions, dict)
        assert 0.5 in predictions
        
        # Check that predictions are pandas DataFrame for matching model
        assert isinstance(predictions[0.5], pd.DataFrame)