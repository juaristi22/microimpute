import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union


def quantile_loss(q: float, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the quantile loss.
    
    :param q: Quantile to be evaluated, e.g., 0.5 for median.
    :type q: float
    :param y: True value.
    :type y: np.ndarray
    :param f: Fitted or predicted value.
    :type f: np.ndarray
    :returns: Array of quantile losses.
    :rtype: np.ndarray
    """
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


def compute_quantile_loss(test_y: np.ndarray, imputations: np.ndarray, q: float) -> np.ndarray:
    """
    Compute quantile loss for given true values and imputations.
    
    :param test_y: Array of true values.
    :type test_y: np.ndarray
    :param imputations: Array of predicted/imputed values.
    :type imputations: np.ndarray
    :param q: Quantile value.
    :type q: float
    :returns: Array of computed losses.
    :rtype: np.ndarray
    """
    losses = quantile_loss(q, test_y, imputations)
    return losses


    
QUANTILES: List[float] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
quantiles_legend: List[str] = [str(int(q * 100)) + 'th percentile' for q in QUANTILES]

def compare_quantile_loss(
    test_y: pd.DataFrame, 
    method_imputations: Dict[str, Dict[float, Union[np.ndarray, pd.DataFrame]]]
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Compare quantile loss across different imputation methods.
    
    :param test_y: DataFrame containing true values.
    :type test_y: pd.DataFrame
    :param method_imputations: Nested dictionary mapping method names to dictionaries 
                              mapping quantiles to imputation values.
    :type method_imputations: Dict[str, Dict[float, Union[np.ndarray, pd.DataFrame]]]
    :returns: A tuple containing:
              - DataFrame with columns 'Method', 'Percentile', and 'Loss'
              - List of quantile values used
    :rtype: Tuple[pd.DataFrame, List[float]]
    """

    # Initialize empty dataframe with method names, quantile, and loss columns
    results_df: pd.DataFrame = pd.DataFrame(columns=['Method', 'Percentile', 'Loss'])

    for method, imputation in method_imputations.items():
        for quantile in QUANTILES:
            q_loss = compute_quantile_loss(test_y.values.flatten(), imputation[quantile].values.flatten(), quantile)
            new_row = {
                'Method': method,
                'Percentile': str(int(quantile * 100)) + 'th percentile',
                'Loss': q_loss.mean()}
            
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    return results_df
