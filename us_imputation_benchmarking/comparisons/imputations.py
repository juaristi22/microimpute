from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.models.quantreg import QuantReg
from us_imputation_benchmarking.models.matching import Matching
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any


def get_imputations(
    methods: List[str], 
    X: pd.DataFrame, 
    test_X: pd.DataFrame, 
    predictors: List[str], 
    imputed_variables: List[str], 
    data: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[float, Union[np.ndarray, pd.DataFrame]]]:

    QUANTILES: List[float] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    method_imputations: Dict[str, Dict[float, Any]] = {}
    for method in methods:
        method_imputations[method] = {}

    # Cross-validation option
    if data:
        for method in methods:
            for q in QUANTILES:
                method_imputations[method][q] = []
        K = 5
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(data):
            X, test_X = data.iloc[train_idx], data.iloc[test_idx]
            
            # QRF
            if "QRF" in methods:
                model = QRF()
                model.fit(X, predictors, imputed_variables)
                imputations = model.predict(test_X, QUANTILES)
                for q in QUANTILES:
                    method_imputations["QRF"][q].append(imputations[q])

            # Matching
            if "Matching" in methods:   
                model = Matching()
                model.fit(X, predictors, imputed_variables)
                imputations = model.predict(test_X, QUANTILES)
                for q in QUANTILES:
                    method_imputations["Matching"][q].append(imputations[q])

            # OLS
            if "OLS" in methods:
                model = OLS()
                model.fit(X, predictors, imputed_variables)
                imputations = model.predict(test_X, QUANTILES)
                for q in QUANTILES:
                    method_imputations["OLS"][q].append(imputations[q])

            # QuantReg
            if "QuantReg" in methods:
                model = QuantReg()
                model.fit(X, predictors, imputed_variables, QUANTILES)
                imputations = model.predict(test_X, QUANTILES)
                for q in QUANTILES:
                    method_imputations["QuantReg"][q].append(imputations[q])
                    
        for method in methods:
            for q in QUANTILES:
                method_imputations[method][q] = np.mean(method_imputations[method][q])
    
    else:
        # QRF
        if "QRF" in methods:
            model = QRF()
            model.fit(X, predictors, imputed_variables)
            imputations = model.predict(test_X, QUANTILES)
            method_imputations["QRF"] = imputations

        # Matching
        if "Matching" in methods:   
            model = Matching()
            model.fit(X, predictors, imputed_variables)
            imputations = model.predict(test_X, QUANTILES)
            method_imputations["Matching"] = imputations

        # OLS
        if "OLS" in methods:
            model = OLS()
            model.fit(X, predictors, imputed_variables)
            imputations = model.predict(test_X, QUANTILES)
            method_imputations["OLS"] = imputations

        # QuantReg
        if "QuantReg" in methods:
            model = QuantReg()
            model.fit(X, predictors, imputed_variables, QUANTILES)
            imputations = model.predict(test_X, QUANTILES)
            method_imputations["QuantReg"] = imputations

    return method_imputations