from quantile_forest import RandomForestQuantileRegressor
import pandas as pd
import numpy as np
import pickle
from typing import List, Optional, Dict, Any, Union, Tuple
from us_imputation_benchmarking.config import RANDOM_STATE


class QRF:
    categorical_columns: Optional[List[str]] = None
    encoded_columns: Optional[List[str]] = None
    output_columns: Optional[List[str]] = None

    def __init__(self, 
                 seed: int = RANDOM_STATE, 
                 file_path: Optional[str] = None):
        """Initialize Quantile Random Forest.

        Args:
            seed: Random seed for reproducibility.
            file_path: Path to a pickled model file to load.
        """
        self.seed = seed
        self.qrf = None

        if file_path is not None:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.seed = data["seed"]
            self.categorical_columns = data["categorical_columns"]
            self.encoded_columns = data["encoded_columns"]
            self.output_columns = data["output_columns"]
            self.qrf = data["qrf"]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **qrf_kwargs: Any) -> None:
        """Fit the Quantile Random Forest model.

        Args:
            X: Feature DataFrame.
            y: Target DataFrame.
            **qrf_kwargs: Additional keyword arguments to pass to RandomForestQuantileRegressor.
        """
        self.categorical_columns = X.select_dtypes(include=["object"]).columns
        X = pd.get_dummies(
            X, columns=self.categorical_columns, drop_first=True
        )
        self.encoded_columns = X.columns
        self.output_columns = y.columns
        self.qrf = RandomForestQuantileRegressor(
            random_state=self.seed, **qrf_kwargs
        )
        self.qrf.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame,
        count_samples: int = 10,
        mean_quantile: float = 0.5,
    ) -> pd.DataFrame:
        """Make predictions with the Quantile Random Forest model.

        Args:
            X: Feature DataFrame.
            count_samples: Number of quantile samples.
            mean_quantile: Target quantile for predictions.

        Returns:
            DataFrame with predictions.
        """
        X = pd.get_dummies(
            X, columns=self.categorical_columns, drop_first=True
        )
        X = X[self.encoded_columns]
        pred = self.qrf.predict(
            X, quantiles=list(np.linspace(0, 1, count_samples))
        )
        random_generator = np.random.default_rng(self.seed)
        a = mean_quantile / (1 - mean_quantile)
        input_quantiles = (
            random_generator.beta(a, 1, size=len(X)) * count_samples
        )
        input_quantiles = input_quantiles.astype(int)
        if len(pred.shape) == 2:
            predictions = pred[np.arange(len(pred)), input_quantiles]
        else:
            predictions = pred[np.arange(len(pred)), :, input_quantiles]
        return pd.DataFrame(predictions, columns=self.output_columns)

    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: File path to save the pickled model.
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "seed": self.seed,
                    "categorical_columns": self.categorical_columns,
                    "encoded_columns": self.encoded_columns,
                    "output_columns": self.output_columns,
                    "qrf": self.qrf,
                },
                f,
            )
