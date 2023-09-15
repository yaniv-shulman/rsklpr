from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

_dir: Path = Path(__file__).parent


def noaa_data() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads the public dataset noaa https://www.ncdc.noaa.gov/cag/global/time-series/globe/land/12/4/1880-2019. Note this
    dataset does not contain ground truth.

    Returns:
        The predictor and response values.
    """
    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=_dir.joinpath("noaa_data.csv"))
    return df.Year.values, df.Value.values, None


def tsla_data() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads the public dataset tsla https://finance.yahoo.com/quote/TSLA/history?p=TSLA. Note this dataset does not
    contain ground truth.

    Returns:
        The predictor and response values.
    """
    ds: pd.Series = pd.read_csv(filepath_or_buffer=_dir.joinpath("TSLA.csv")).squeeze()
    return np.asarray(ds.index), ds.values, None


def anti_diabetic_data() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads the public dataset anti-diabetic https://raw.githubusercontent.com/selva86/datasets/master/a10.csv. Note this
    dataset does not contain ground truth.

    Returns:
        The predictor and response values.
    """
    ds: pd.Series = pd.read_csv(
        filepath_or_buffer=_dir.joinpath("anti-diabetic.csv")
    ).squeeze()
    return np.asarray(ds.index), ds.values, None


def elecequip_data() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads the public dataset elecequip https://raw.githubusercontent.com/selva86/datasets/master/elecequip.csv. Note
    this dataset does not contain ground truth.

    Returns:
        The predictor and response values.
    """
    ds: pd.Series = pd.read_csv(
        filepath_or_buffer=_dir.joinpath("elecequip.csv")
    ).squeeze()
    del ds["date"]
    return np.asarray(ds.index), ds["value"].values, None
