"""
Tests for the plot interface of the microimpute package.
"""

import os
from typing import Dict, List, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from microimpute.comparisons import compare_quantile_loss, get_imputations
from microimpute.config import QUANTILES, RANDOM_STATE
from microimpute.evaluations import cross_validate_model
from microimpute.models import OLS, QRF, Imputer, Matching, QuantReg
from microimpute.visualizations.plotting import *


def test_model_cv_visualization():
    """Test visualization of cross-validation results using the statsmodels-like interface."""
    # Load diabetes dataset
    diabetes_data = load_diabetes()
    diabetes_df = pd.DataFrame(
        diabetes_data.data, columns=diabetes_data.feature_names
    )

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    # Prepare data for modeling
    X = diabetes_df[predictors + imputed_variables]

    # Run cross-validation for OLS model
    ols_results = cross_validate_model(OLS, X, predictors, imputed_variables)

    # Validate results are complete
    assert not ols_results.isna().any().any()

    # Create a PerformanceResults object for visualization
    perf_results_viz = model_performance_results(
        results=ols_results,
        model_name="OLS",
        method_name="Cross-Validation Quantile Loss Average",
    )

    # Create a plot with the statsmodels-like interface
    fig = perf_results_viz.plot(
        title="OLS Cross-Validation Performance",
    )

    # Generate summary statistics
    summary = perf_results_viz.summary()

    # Validate output types
    assert isinstance(fig, go.Figure)
    assert isinstance(summary, pd.DataFrame)
    assert "Mean Train Loss" in summary.columns
    assert "Mean Test Loss" in summary.columns

    return


def test_model_performance_comparison():
    """Test comparison of multiple imputation methods using the statsmodels-like interface."""
    # 1. Prepare data
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    X_train, X_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE
    )

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    Y_test: pd.DataFrame = X_test[imputed_variables]

    # 2. Run imputation methods
    model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    # 3. Compare imputation methods
    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # 4. Create visualization using the statsmodels-like interface
    # The loss_comparison_df is in long format suitable for the MethodComparison class

    # Create a MethodComparison object using the factory function
    comparison = method_comparison_results(
        data=loss_comparison_df,
        metric_name="Quantile Loss",
        data_format="long",
    )

    # Create a plot comparing methods across quantiles
    fig = comparison.plot(title="Imputation Method Comparison", show_mean=True)

    # Get summary statistics
    summary = comparison.summary()

    # Validate output
    assert isinstance(fig, go.Figure)
    assert isinstance(summary, pd.DataFrame)
    assert "Method" in summary.columns
    assert (
        "Mean Quantile Loss" in summary.columns
        or "Mean Loss" in summary.columns
    )

    return


def test_autoimpute_wide_format_visualization():
    """Test visualization of autoimpute-style method comparison results using wide format."""
    # 1. Prepare data
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # Split into donor (training) and receiver (imputation target) data
    donor_data, receiver_data = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE
    )

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    # 2. Simulate the output of autoimpute cross-validation
    # Create a DataFrame with methods as index and quantiles as columns (wide format)
    # This simulates the method_results_df from autoimpute

    # Define methods and their performance at different quantiles
    methods = {
        "OLS": [
            0.54,
            0.71,
            0.60,
            0.54,
            0.42,
            0.64,
            0.43,
            0.89,
            0.96,
            0.38,
            0.79,
            0.52,
            0.56,
            0.92,
            0.17,
            0.28,
            0.09,
            0.83,
            0.77,
        ],
        "QRF": [
            0.87,
            0.97,
            0.79,
            0.46,
            0.78,
            0.11,
            0.68,
            0.14,
            0.94,
            0.52,
            0.41,
            0.26,
            0.77,
            0.45,
            0.56,
            0.06,
            0.61,
            0.61,
            0.61,
        ],
        "QuantReg": [
            0.94,
            0.68,
            0.35,
            0.43,
            0.69,
            0.060,
            0.66,
            0.67,
            0.21,
            0.12,
            0.31,
            0.36,
            0.57,
            0.43,
            0.88,
            0.10,
            0.20,
            0.16,
            0.65,
        ],
        "Matching": [
            0.25,
            0.46,
            0.24,
            0.15,
            0.11,
            0.65,
            0.13,
            0.19,
            0.36,
            0.82,
            0.09,
            0.83,
            0.09,
            0.92,
            0.46,
            0.91,
            0.60,
            0.73,
            0.08,
        ],
    }

    # Create wide format DataFrame with quantiles as columns
    quantile_labels = [f"{int(q * 100)}" for q in QUANTILES]
    wide_format_df = pd.DataFrame(methods, index=quantile_labels).T

    # Add mean_loss column as autoimpute does
    wide_format_df["mean_loss"] = wide_format_df.mean(axis=1)

    # 3. Create visualization using the statsmodels-like interface
    comparison = method_comparison_results(
        data=wide_format_df,
        metric_name="Test Quantile Loss",
        data_format="wide",  # Explicitly using wide format
    )

    # 4. Create visualization
    fig = comparison.plot(
        title="Autoimpute Method Comparison (Wide Format)", show_mean=True
    )

    # 5. Get summary statistics
    summary = comparison.summary()

    # Verify output
    assert isinstance(fig, go.Figure)
    assert isinstance(summary, pd.DataFrame)
    assert "Method" in summary.columns
    assert (
        "Mean Test Quantile Loss" in summary.columns
        or "Mean Quantile Loss" in summary.columns
    )

    # 6. Create a focused comparison of top methods
    # Select just the top 2 methods (QRF and OLS in this case)
    top_methods_df = wide_format_df.loc[["QRF", "OLS"]]

    # Create a new comparison object
    top_comparison = method_comparison_results(
        data=top_methods_df,
        metric_name="Test Quantile Loss",
        data_format="wide",
    )

    # Generate plot for just the top methods
    top_fig = top_comparison.plot(
        title="Top Methods Comparison", show_mean=True
    )

    # Verify output
    assert isinstance(top_fig, go.Figure)
    assert len(top_comparison.methods) == 2

    return
