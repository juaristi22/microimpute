"""Statistical matching hot deck imputation utilities.

This module provides an interface to R's StatMatch package for performing nearest neighbor
distance hot deck matching.
"""

import logging
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import VALIDATE_CONFIG

log = logging.getLogger(__name__)

"""
data.rec: A matrix or data frame that plays the role of recipient in the
statistical matching application.

data.don: A matrix or data frame that that plays the role of donor in the statistical matching application.

mtc.ids: A matrix with two columns. Each row must contain the name or the index of the recipient record (row) in data.don and the name or the index of the corresponding donor record (row) in data.don. Note that this type of matrix is returned by the functions NND.hotdeck, RANDwNND.hotdeck, rankNND.hotdeck, and mixed.mtc.

z.vars: A character vector with the names of the variables available only in data.don that should be "donated" to data.rec.
"""

import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


@validate_call(config=VALIDATE_CONFIG)
def nnd_hotdeck_using_rpy2(
    receiver: pd.DataFrame,
    donor: pd.DataFrame,
    matching_variables: List[str],
    z_variables: List[str],
    **matching_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform nearest neighbor distance hot deck matching using R's StatMatch package.

    Args:
        receiver: DataFrame containing recipient data.
        donor: DataFrame containing donor data.
        matching_variables: List of column names to use for matching.
        z_variables: List of column names to donate from donor to recipient.
        **matching_kwargs: Optional hyperparameters for matching.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two fused DataFrames:
            - First without duplication of matching variables
            - Second with duplication of matching variables


    Raises:
        ValueError: If any of the matching variables or z variables are not found in the respective DataFrames.
        RuntimeError: If there is an unexpected error during the statistical matching process.
    """

    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    StatMatch = importr("StatMatch")

    try:
        missing_in_receiver = [
            v for v in matching_variables if v not in receiver.columns
        ]
        missing_in_donor = [
            v for v in matching_variables if v not in donor.columns
        ]
        if missing_in_receiver:
            msg = f"Matching variables missing in receiver: {missing_in_receiver}"
            log.error(msg)
            raise ValueError(msg)
        if missing_in_donor:
            msg = f"Matching variables missing in donor: {missing_in_donor}"
            log.error(msg)
            raise ValueError(msg)
        missing_z = [v for v in z_variables if v not in donor.columns]
        if missing_z:
            msg = f"Z variables missing in donor: {missing_z}"
            log.error(msg)
            raise ValueError(msg)

        with localconverter(
            default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_receiver = conversion.py2rpy(receiver)
            r_donor = conversion.py2rpy(donor)
            r_match = ro.StrVector(matching_variables)
            r_z = ro.StrVector(z_variables)

        if matching_kwargs:
            out_NND = StatMatch.NND_hotdeck(
                data_rec=r_receiver,
                data_don=r_donor,
                match_vars=r_match,
                **matching_kwargs,
            )
        else:
            out_NND = StatMatch.NND_hotdeck(
                data_rec=r_receiver,
                data_don=r_donor,
                match_vars=r_match,
            )

        # Create the correct matching indices matrix for StatMatch.create_fused
        recipient_indices = np.arange(1, len(receiver) + 1)
        mtc_ids_r = out_NND.rx2("mtc.ids")
        # Create the properly formatted 2-column matrix that create_fused expects
        if hasattr(mtc_ids_r, "ncol") and mtc_ids_r.ncol == 2:
            # Already a matrix with the right shape, use it directly
            mtc_ids = mtc_ids_r
        else:
            mtc_array = np.array(mtc_ids_r)
            # If we have a 1D array with strings, convert to integers
            if mtc_array.dtype.kind in ["U", "S"]:
                mtc_array = np.array([int(x) for x in mtc_array])
            # If the mtc.ids array has 2 values per recipient (recipient_idx, donor_idx pairs)
            if len(mtc_array) == 2 * len(receiver):
                donor_indices = mtc_array.reshape(-1, 2)[:, 1]
                # Make sure these indices are within the valid range (1 to donor dataset size)
                donor_indices_valid = (
                    np.remainder(donor_indices - 1, len(donor)) + 1
                )
            else:
                if len(mtc_array) >= len(receiver):
                    # Use the indices directly (up to the length of receiver)
                    donor_indices_valid = mtc_array[: len(receiver)]
                else:
                    # If we have too few indices, repeat the last ones to match length
                    donor_indices_valid = np.concatenate(
                        [
                            mtc_array,
                            np.repeat(
                                mtc_array[-1], len(receiver) - len(mtc_array)
                            ),
                        ]
                    )
            # Create the final mtc.ids matrix required by create_fused
            mtc_matrix = np.column_stack(
                (recipient_indices, donor_indices_valid)
            )
            # Convert to R matrix
            mtc_ids = ro.r.matrix(
                ro.IntVector(mtc_matrix.flatten()),
                nrow=len(recipient_indices),
                ncol=2,
            )

        fused_0_r = StatMatch.create_fused(
            data_rec=r_receiver,
            data_don=r_donor,
            mtc_ids=mtc_ids,
            z_vars=r_z,
        )
        fused_1_r = StatMatch.create_fused(
            data_rec=r_receiver,
            data_don=r_donor,
            mtc_ids=mtc_ids,
            z_vars=r_z,
            dup_x=False,
            match_vars=r_match,
        )

        with localconverter(
            default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            fused_0 = conversion.rpy2py(fused_0_r)
            fused_1 = conversion.rpy2py(fused_1_r)

        return fused_0, fused_1

    except ValueError:
        raise
    except Exception as e:
        log.error(f"Unexpected error in statistical matching: {e}")
        raise RuntimeError(
            f"Statistical matching failed with unexpected error: {e}"
        ) from e
