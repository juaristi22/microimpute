from sklearn.model_selection import train_test_split
import pandas as pd
import io
import requests
import zipfile
from tqdm import tqdm
from typing import List, Union, Optional, Tuple, Set, Dict, Any

from us_imputation_benchmarking.config import VALID_YEARS, RANDOM_STATE


def scf_url(year: int) -> str:
    """Return the URL of the SCF summary microdata zip file for a year.

    Args:
        year: Year of SCF summary microdata to retrieve.

    Returns:
        URL of summary microdata zip file for the given year.

    Raises:
        AssertionError: If the year is not in VALID_YEARS.
    """
    assert year in VALID_YEARS, "The SCF is not available for " + str(year)
    return (
        "https://www.federalreserve.gov/econres/files/scfp"
        + str(year)
        + "s.zip"
    )


def _load(
    years: Optional[Union[int, List[int]]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load Survey of Consumer Finances data for specified years and columns.

    Args:
        years: Year or list of years to load data for.
        columns: List of column names to load.

    Returns:
        DataFrame containing the requested data.

    Raises:
        ValueError: If no Stata files are found in the downloaded zip.
    """
    if years is None:
        years = VALID_YEARS

    if isinstance(years, int):
        years = [years]

    all_data: List[pd.DataFrame] = []

    for year in tqdm(years):
        # Download zip file
        response = requests.get(scf_url(year))
        z = zipfile.ZipFile(io.BytesIO(response.content))

        # Find the .dta file in the zip
        dta_files: List[str] = [f for f in z.namelist() if f.endswith(".dta")]
        if not dta_files:
            raise ValueError(f"No Stata files found in zip for year {year}")

        # Read the Stata file
        with z.open(dta_files[0]) as f:
            df = pd.read_stata(io.BytesIO(f.read()), columns=columns)

        # Ensure 'wgt' is included
        if columns is not None and "wgt" not in df.columns:
            # Re-read to include weights
            with z.open(dta_files[0]) as f:
                cols_with_weight: List[str] = list(set(columns) | {"wgt"})
                df = pd.read_stata(
                    io.BytesIO(f.read()), columns=cols_with_weight
                )

        # Add year column
        df["year"] = year
        all_data.append(df)

    # Combine all years
    if len(all_data) > 1:
        return pd.concat(all_data)
    else:
        return all_data[0]


def preprocess_data(
    full_data: bool = False, years: Optional[Union[int, List[int]]] = None
) -> Union[
    Tuple[pd.DataFrame, List[str], List[str]],  # when full_data=True
    Tuple[
        pd.DataFrame, pd.DataFrame, List[str], List[str]
    ],  # when full_data=False
]:
    """Preprocess the Survey of Consumer Finances data for model training and testing.

    Args:
        full_data: Whether to return the complete dataset without splitting.
        years: Year or list of years to load data for.

    Returns:
        Different tuple formats depending on the value of full_data:
          - If full_data=True: (data, predictor_columns, imputed_columns)
          - If full_data=False: (train_data, test_data, predictor_columns, imputed_columns)
    """
    data = _load(years=years)

    # predictors shared with cps data

    PREDICTORS: List[str] = [
        "hhsex",  # sex of head of household
        "age",  # age of respondent
        "married",  # marital status of respondent
        "kids",  # number of children in household
        "educ",  # highest level of education
        "race",  # race of respondent
        "income",  # total annual income of household
        "wageinc",  # income from wages and salaries
        "bussefarminc",  # income from business, self-employment or farm
        "intdivinc",  # income from interest and dividends
        "ssretinc",  # income from social security and retirement accounts
        "lf",  # labor force status
    ]

    IMPUTED_VARIABLES: List[str] = [
        "networth"
    ]  # some property also captured in cps data (HPROP_VAL)

    data = data[PREDICTORS + IMPUTED_VARIABLES]
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std

    if full_data:
        return data, PREDICTORS, IMPUTED_VARIABLES
    else:
        X, test_X = train_test_split(
            data, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE
        )
        return X, test_X, PREDICTORS, IMPUTED_VARIABLES
