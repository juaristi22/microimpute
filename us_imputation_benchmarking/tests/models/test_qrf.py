from us_imputation_benchmarking.comparisons.data import preprocess_data, prepare_scf_data
from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.config import RANDOM_STATE, QUANTILES
from sklearn.datasets import load_iris
import pandas as pd
from us_imputation_benchmarking.evaluations.cross_validation import (
    cross_validate_model,
)
from us_imputation_benchmarking.evaluations.train_test_performance import (
    plot_train_test_performance,
)

# Shrink down the data by sampling
data, PREDICTORS, IMPUTED_VARIABLES = prepare_scf_data(full_data=True)
data = data.sample(frac=0.01, random_state=RANDOM_STATE)

qrf_results = cross_validate_model(
        QRF, data, PREDICTORS, IMPUTED_VARIABLES
    )
qrf_results.to_csv("qrf_results.csv")

assert not qrf_results.isna().any().any()

#plot_train_test_performance(qrf_results, save_path="qrf_train_test_performance.png")

# Test Method on iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
imputed_variables = ['petal width (cm)']

iris_df = iris_df[predictors + imputed_variables]

def test_qrf_example(data=iris_df, 
                    predictors=predictors, 
                    imputed_variables=imputed_variables, 
                    quantiles = QUANTILES):
        """
        Example of how to use the Quantile Random Forest imputer model.
        
        This example demonstrates:
        - Initializing a QRF model
        - Fitting the model with optional hyperparameters
        - Predicting quantiles on test data
        - How QRF can capture complex nonlinear relationships
        
        Args:
            train_test_data: Tuple of (train_data, test_data)
        """
        X_train, X_test = preprocess_data(data)
        
        # Initialize QRF model
        model = QRF()
        
        # Fit the model with RF hyperparameters
        model.fit(
            X_train, 
            predictors, 
            imputed_variables,
            n_estimators=100,  # Number of trees
            min_samples_leaf=5  # Min samples in leaf nodes
        )
        
        # Predict at multiple quantiles
        predictions = model.predict(X_test, quantiles)
        
        # Check structure of predictions
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == set(quantiles)
        
        # QRF should capture nonlinear relationships
        # We'd need more complex tests to verify this in detail