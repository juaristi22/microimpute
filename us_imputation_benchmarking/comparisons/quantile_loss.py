import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from us_imputation_benchmarking.config import QUANTILES


def quantile_loss(q: float, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Calculate the quantile loss.

    Args:
        q: Quantile to be evaluated, e.g., 0.5 for median.
        y: True value.
        f: Fitted or predicted value.

    Returns:
        Array of quantile losses.
    """
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


def compute_quantile_loss(
    test_y: np.ndarray, 
    imputations: np.ndarray, 
    q: float
) -> np.ndarray:
    """Compute quantile loss for given true values and imputations.

    Args:
        test_y: Array of true values.
        imputations: Array of predicted/imputed values.
        q: Quantile value.

    Returns:
        np.ndarray: Element-wise quantile loss values between true values and predictions.
    """
    losses = quantile_loss(q, test_y, imputations)
    return losses

quantiles_legend: List[str] = [
    str(int(q * 100)) + "th percentile" for q in QUANTILES
]


def compare_quantile_loss(
    test_y: pd.DataFrame,
    method_imputations: Dict[
        str, Dict[float, Union[np.ndarray, pd.DataFrame]]
    ],
) -> pd.DataFrame:
    """Compare quantile loss across different imputation methods.

    Args:
        test_y: DataFrame containing true values.
        method_imputations: Nested dictionary mapping method names to dictionaries
                          mapping quantiles to imputation values.

    Returns:
        pd.DataFrame: Results dataframe with columns 'Method', 'Percentile', and 'Loss'
                     containing the mean quantile loss for each method and percentile.
    """

    # Initialize empty dataframe with method names, quantile, and loss columns
    results_df: pd.DataFrame = pd.DataFrame(
        columns=["Method", "Percentile", "Loss"]
    )

    for method, imputation in method_imputations.items():
        for quantile in QUANTILES:
            q_loss = compute_quantile_loss(
                test_y.values.flatten(),
                imputation[quantile].values.flatten(),
                quantile,
            )
            new_row = {
                "Method": method,
                "Percentile": str(int(quantile * 100)) + "th percentile",
                "Loss": q_loss.mean(),
            }

            results_df = pd.concat(
                [results_df, pd.DataFrame([new_row])], ignore_index=True
            )

    return results_df
