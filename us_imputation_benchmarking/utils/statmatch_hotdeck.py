import numpy as np
import pandas as pd
import logging
import os
import pkg_resources
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from typing import List, Dict, Optional, Union, Any, Tuple
# Enable R-Python DataFrame and array conversion
pandas2ri.activate()
numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
StatMatch = importr("StatMatch")


log = logging.getLogger(__name__)

"""
data.rec: A matrix or data frame that plays the role of recipient in the statistical matching application.

data.don: A matrix or data frame that that plays the role of donor in the statistical matching application.

mtc.ids: A matrix with two columns. Each row must contain the name or the index of the recipient record (row) 
          in data.don and the name or the index of the corresponding donor record (row) in data.don. Note that 
          this type of matrix is returned by the functions NND.hotdeck, RANDwNND.hotdeck, rankNND.hotdeck, and 
          mixed.mtc.

z.vars: A character vector with the names of the variables available only in data.don that should be "donated" to data.rec.
networth

dup.x: Logical. When TRUE the values of the matching variables in data.don are also "donated" to data.rec. 
        The names of the matching variables have to be specified with the argument match.vars. To avoid confusion, 
        the matching variables added to data.rec are renamed by adding the suffix "don". By default dup.x=FALSE.

match.vars: A character vector with the names of the matching variables. It has to be specified only when dup.x=TRUE. 
All other vars (no need to specify)     
"""

def nnd_hotdeck_using_rpy2(
    receiver: Optional[pd.DataFrame] = None, 
    donor: Optional[pd.DataFrame] = None, 
    matching_variables: Optional[List[str]] = None,
    z_variables: Optional[List[str]] = None, 
    donor_classes: Optional[Union[str, List[str]]] = None
) -> Tuple[Any, Any]:
    """
    Perform nearest neighbor distance hot deck matching using R's StatMatch package.
    
    :param receiver: DataFrame containing recipient data.
    :type receiver: Optional[pd.DataFrame]
    :param donor: DataFrame containing donor data.
    :type donor: Optional[pd.DataFrame]
    :param matching_variables: List of column names to use for matching.
    :type matching_variables: Optional[List[str]]
    :param z_variables: List of column names to donate from donor to recipient.
    :type z_variables: Optional[List[str]]
    :param donor_classes: Column name(s) used to define classes in the donor data.
    :type donor_classes: Optional[Union[str, List[str]]]
    :returns: Tuple containing two fused datasets:
              - First without duplication of matching variables
              - Second with duplication of matching variables
    :rtype: Tuple[Any, Any]
    :raises AssertionError: If receiver, donor, or matching_variables are not provided.
    """
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    assert receiver is not None and donor is not None, "Receiver and donor must be provided"
    assert matching_variables is not None, "Matching variables must be provided"

    pandas2ri.activate()
    StatMatch = importr("StatMatch")

    if isinstance(donor_classes, str):
        assert donor_classes in receiver, 'Donor class not present in receiver'
        assert donor_classes in donor, 'Donor class not present in donor'

    try:
        if donor_classes:
            out_NND = StatMatch.NND_hotdeck(
                data_rec=receiver,
                data_don=donor,
                match_vars=pd.Series(matching_variables),
                don_class=pd.Series(donor_classes)
                )
        else:
            out_NND = StatMatch.NND_hotdeck(
                data_rec=receiver,
                data_don=donor,
                match_vars=pd.Series(matching_variables),
                # don_class = pd.Series(donor_classes)
                )
            
    except Exception as e:
        print(1)
        print(receiver)
        print(2)
        print(donor)
        print(3)
        print(pd.Series(matching_variables))
        print(e)

    # Convert out_NND[0] to a NumPy array and reshape to 2 columns
    # To address shape errors when using hotdeck for mtc_ids
    mtc_ids_array = np.array(out_NND[0])
    mtc_ids_2d = mtc_ids_array.reshape(-1, 2)
    # Convert the 2D NumPy array to an R matrix using py2rpy
    mtc_ids = ro.conversion.py2rpy(mtc_ids_2d)

    print("mtc_ids:", mtc_ids)
    print("mtc_ids type:", type(mtc_ids))

    # create synthetic data.set, without the
    # duplication of the matching variables
    fused_0 = StatMatch.create_fused(
            data_rec=receiver,
            data_don=donor,
            mtc_ids=mtc_ids,
            z_vars=pd.Series(z_variables)
            )

    # create synthetic data.set, with the "duplication"
    # of the matching variables
    fused_1 = StatMatch.create_fused(
            data_rec=receiver,
            data_don=donor,
            mtc_ids=mtc_ids,
            z_vars=pd.Series(z_variables),
            dup_x=False,
            match_vars=pd.Series(matching_variables)
            )

    return fused_0, fused_1
