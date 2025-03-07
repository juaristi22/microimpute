import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union


def quantile_loss(q: float, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the quantile loss.
    
    Args:
        q: Quantile to be evaluated, e.g., 0.5 for median.
        y: True value.
        f: Fitted or predicted value.
        
    Returns:
        Array of quantile losses.
    """
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


def compute_quantile_loss(test_y: np.ndarray, imputations: np.ndarray, q: float) -> np.ndarray:
    """
    Compute quantile loss for given true values and imputations.
    
    Args:
        test_y: Array of true values.
        imputations: Array of predicted/imputed values.
        q: Quantile value.
        
    Returns:
        Array of computed losses.
    """
    losses = quantile_loss(q, test_y, imputations)
    return losses


def compare_quantile_loss(
    test_y: pd.DataFrame, 
    method_imputations: Dict[str, Dict[float, Union[np.ndarray, pd.DataFrame]]]
) -> Tuple[pd.DataFrame, List[float]]:
    
    QUANTILES: List[float] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    quantiles_legend: List[str] = [str(int(q * 100)) + 'th percentile' for q in QUANTILES]

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

    return results_df, QUANTILES

