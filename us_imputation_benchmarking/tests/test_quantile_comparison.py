from typing import List, Type

import pandas as pd

from us_imputation_benchmarking.comparisons import *
from us_imputation_benchmarking.models import *
from us_imputation_benchmarking.config import RANDOM_STATE


def test_quantile_comparison() -> None:
    """Test the end-to-end quantile loss comparison workflow."""
    X_train, X_test, PREDICTORS, IMPUTED_VARIABLES = prepare_scf_data(
        full_data=False, years=2019
    )
    # Shrink down the data by sampling
    X_train = X_train.sample(frac=0.01, random_state=RANDOM_STATE)
    X_test = X_test.sample(frac=0.01, random_state=RANDOM_STATE)

    Y_test: pd.DataFrame = X_test[IMPUTED_VARIABLES]

    model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, PREDICTORS, IMPUTED_VARIABLES
    )

    loss_comparison_df = compare_quantile_loss(Y_test, method_imputations)

    assert not loss_comparison_df.isna().any().any()

    plot_loss_comparison(loss_comparison_df, save_path="loss_comparison.png")
