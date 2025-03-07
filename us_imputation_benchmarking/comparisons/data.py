from sklearn.model_selection import train_test_split
import pandas as pd
import io
import requests
import zipfile
from typing import List, Union, Optional, Tuple, Set, Dict, Any

VALID_YEARS: List[int] = [
    1989,
    1992,
    1995,
    1998,
    2001,
    2004,
    2007,
    2010,
    2013,
    2016,
    2019,
]

def scf_url(year: int) -> str:
    """ Returns the URL of the SCF summary microdata zip file for a year.

    :param year: Year of SCF summary microdata to retrieve.
    :type year: int
    :return: URL of summary microdata zip file for the given year.
    :rtype: str
    """
    assert year in VALID_YEARS, "The SCF is not available for " + str(year)
    return (
        "https://www.federalreserve.gov/econres/files/scfp"
        + str(year)
        + "s.zip"
    )


def _load(years: Optional[Union[int, List[int]]] = None, 
           columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load Survey of Consumer Finances data for specified years and columns.
    
    Args:
        years: Year or list of years to load data for. Defaults to all valid years.
        columns: List of column names to load. Defaults to all columns.
        
    Returns:
        DataFrame containing the requested data.
    """
    if years is None:
        years = VALID_YEARS

    if isinstance(years, int):
        years = [years]

    all_data: List[pd.DataFrame] = []

    for year in years:
        # Download zip file
        response = requests.get(scf_url(year))
        z = zipfile.ZipFile(io.BytesIO(response.content))

        # Find the .dta file in the zip
        dta_files: List[str] = [f for f in z.namelist() if f.endswith('.dta')]
        if not dta_files:
            raise ValueError(f"No Stata files found in zip for year {year}")

        # Read the Stata file
        with z.open(dta_files[0]) as f:
            df = pd.read_stata(io.BytesIO(f.read()), columns=columns)

        # Ensure 'wgt' is included
        if columns is not None and 'wgt' not in df.columns:
            # Re-read to include weights
            with z.open(dta_files[0]) as f:
                cols_with_weight: List[str] = list(set(columns) | {'wgt'})
                df = pd.read_stata(io.BytesIO(f.read()), columns=cols_with_weight)

        # Add year column
        df['year'] = year
        all_data.append(df)

    # Combine all years
    if len(all_data) > 1:
        return pd.concat(all_data)
    else:
        return all_data[0]


def preprocess_data(
    full_data: bool = False, 
    years: Optional[Union[int, List[int]]] = None
) -> Union[
    Tuple[pd.DataFrame, List[str], List[str]],  # when full_data=True
    Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]  # when full_data=False
]:
    """
    Preprocess the Survey of Consumer Finances data for model training and testing.
    
    Args:
        full_data: If True, returns the entire dataset without splitting. If False,
                  splits the data into training and test sets. Defaults to False.
        years: Year or list of years to load data for. Defaults to all valid years.
        
    Returns:
        If full_data=True:
            Tuple containing (data, predictor_columns, imputed_columns)
        If full_data=False:
            Tuple containing (train_data, test_data, predictor_columns, imputed_columns)
    """
    data = _load(years=years)

    # predictors shared with cps data

    PREDICTORS: List[str] = ["hhsex",      # sex of head of household
                "age",          # age of respondent
                "married",      # marital status of respondent
                "kids",         # number of children in household
                "educ",         # highest level of education
                "race",         # race of respondent 
                "income",       # total annual income of household  
                "wageinc",      # income from wages and salaries
                "bussefarminc", # income from business, self-employment or farm
                "intdivinc",    # income from interest and dividends
                "ssretinc",     # income from social security and retirement accounts
                "lf",           # labor force status
                ]   

    IMPUTED_VARIABLES: List[str] = ["networth"] # some property also captured in cps data (HPROP_VAL)

    data = data[PREDICTORS + IMPUTED_VARIABLES]
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std

    if full_data:
        return data, PREDICTORS, IMPUTED_VARIABLES
    else:
        X, test_X = train_test_split(data, test_size=0.2, train_size=0.8, random_state=42)
        return X, test_X, PREDICTORS, IMPUTED_VARIABLES