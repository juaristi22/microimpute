"""
Pipeline for autoimputation of missing values in a dataset.
This module integrates all steps necessary for method selection and imputation of missing values.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from microimpute.comparisons import *
from microimpute.config import RANDOM_STATE, QUANTILES, TRAIN_SIZE
from microimpute. models import *

import pandas as pd

log = logging.getLogger(__name__)

def autoimpute(donor_data: pd.DataFrame,
               receiver_data: pd.DataFrame,
               predictors: List[str],
               imputed_variables: List[str], 
               models: Optional[List["Imputer"]] = None,
               quantiles: Optional[List[float]] = QUANTILES,
               hyperparameters: Optional[Dict[str, Any]] = None,
               random_state: Optional[int] = RANDOM_STATE,
               train_size: Optional[float] = TRAIN_SIZE,
               k_folds: Optional[int] = 5,
               ) -> Any:
    """
    """
    # Step 0: Input validation
    try: 
        # Validate quantiles if provided
        if quantiles:
            invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
            if invalid_quantiles:
                error_msg = f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Validate that predictor and imputed variable columns exist in donor data
        missing_predictors_donor = [
            col for col in predictors if col not in donor_data.columns
        ]
        if missing_predictors_donor:
            error_msg = f"Missing predictor columns in donor data: {missing_predictors_donor}"
            log.error(error_msg)
            raise ValueError(error_msg)
        
        missing_predictors_receiver = [
            col for col in predictors if col not in receiver_data.columns
        ]
        if missing_predictors_receiver:
            error_msg = f"Missing predictor columns in reciver data: {missing_predictors_receiver}"
            log.error(error_msg)
            raise ValueError(error_msg)

        missing_imputed_donor = [
            col for col in imputed_variables if col not in donor_data.columns
        ]
        if missing_imputed_donor:
            error_msg = f"Missing imputed variable columns in donor data: {missing_imputed_donor}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate that predictor columns exist in receiver data (imputed variables may not be present in receiver data)
        missing_predictors_receiver = [
            col for col in predictors if col not in receiver_data.columns
        ]
        if missing_predictors_receiver:
            error_msg = f"Missing predictor columns in test data: {missing_predictors_receiver}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(
            f"Generating imputations to impute from {len(donor_data)} donor data to {len(receiver_data)} receiver data for variables {imputed_variables} with predictors {predictors}. "
        )

        # Step 1: Data preparation

        ## Normalizing ?? should we drop the columns that are not predictors or imputed variables? 

        X_train, X_test, dummy_info = preprocess_data(donor_data[predictors + imputed_variables], train_size=train_size, test_size=(1-train_size))
        receiver_data, dummy_info = preprocess_data(receiver_data[predictors], full_data=True, train_size=train_size, test_size=(1-train_size))

        if dummy_info:
            # Retrieve new predictors and imputed variables after processed data
            for orig_col, dummy_cols in dummy_info.items():
                if orig_col in predictors:
                    predictors.remove(orig_col)
                    predictors + dummy_cols
                elif orig_col in imputed_variables:
                    imputed_variables.remove(orig_col)
                    imputed_variables + dummy_cols

        Y_test = X_test[imputed_variables]

        # If imputed variables are in receiver data, remove them
        receiver_data = receiver_data.drop(columns=imputed_variables, errors='ignore')

        # Step 2: Imputation with each method

        ## We evaluate imputation methods with a test donor subset right?

        if not models:
            # If no models are provided, use default models
            model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
        method_imputations, fitted_models = get_imputations(
            model_classes, X_train, X_test, predictors, imputed_variables, quantiles,
        )

        # Step 3: Compare imputation methods
        log.info(
            f"Comparing across {model_classes} methods. "
        )

        ## And do we want to compare only imputations once as per our compare_quantile_loss function? Or do we also want to include comparisons running cv for each method? Do we want to compare the mean or do a more specific evaluation across quantiles? Should this be a parameter?

        loss_comparison_df = compare_quantile_loss(
            Y_test, method_imputations, imputed_variables
        )

        # Step 4: Select best method
        average_losses = loss_comparison_df.query("`Imputed Variable` == 'average' and Percentile == 'average'")
        best_row = average_losses.loc[average_losses["Loss"].idxmin()]
        best_method = best_row["Method"]

        log.info(
            f"The method with the lowest average loss is {best_method}, with an average loss across variables and quantiles of {best_row['Loss']}. "
        )

        # Step 5: Generate imputations with the best method on the receiver data
        fitted_best = fitted_models[best_method]

        imputations_per_quantile = fitted_best.predict(
            X_test=receiver_data,
            quantiles=quantiles,
        )

        log.info(
            f"Imputation generation completed for {len(receiver_data)} samples using the best method: {best_method} and {len(imputations_per_quantile)} quantiles. "
        )

        return imputations_per_quantile, fitted_best, average_losses, loss_comparison_df

    except ValueError as e:
        # Re-raise validation errors directly
        raise e
    except Exception as e:
        log.error(f"Unexpected error during autoimputation: {str(e)}")
        raise RuntimeError(f"Failed to generate imputations: {str(e)}") from e
