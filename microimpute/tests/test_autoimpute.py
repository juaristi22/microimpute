"""
Test the autoimpute function.
"""

import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons.autoimpute import autoimpute
from microimpute.comparisons.plot import (
    plot_autoimpute_method_comparison,
    plot_loss_comparison,
)


def test_autoimpute_basic():
    """Test that autoimpute returns expected data structures."""
    diabetes = load_diabetes()
    diabetes_donor = pd.DataFrame(
        diabetes.data, columns=diabetes.feature_names
    )
    diabetes_receiver = pd.DataFrame(
        diabetes.data, columns=diabetes.feature_names
    )

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    imputations, fitted_model, method_results_df = autoimpute(
        donor_data=diabetes_donor,
        receiver_data=diabetes_receiver,
        predictors=predictors,
        imputed_variables=imputed_variables,
        hyperparameters={"QRF": {"n_estimators": 100}},
    )

    # Check that the imputations is a dictionary of dataframes
    assert isinstance(imputations, dict)
    for quantile, df in imputations.items():
        assert isinstance(df, pd.DataFrame)
        # Check that the imputed variables are in the dataframe
        for var in imputed_variables:
            assert var in df.columns

    # Check that the method_results_df has the expected structure
    assert isinstance(method_results_df, pd.DataFrame)
    # method_results_df will have quantiles as columns and model names as indices
    assert "mean_loss" in method_results_df.columns
    assert 0.05 in method_results_df.columns  # First quantile
    assert 0.95 in method_results_df.columns  # Last quantile

    quantiles = [q for q in method_results_df.columns if isinstance(q, float)]

    imputations[0.5].to_csv("autoimpute_bestmodel_median_imputations.csv")

    method_results_df.to_csv("autoimpute_model_comparison_results.csv")

    plot_autoimpute_method_comparison(
        method_results_df=method_results_df,
        quantiles=quantiles,
        metric_name="Quantile Loss",
        save_path="autoimpute_model_comparison.jpg",
    )
