__all__ = ['mstl']


import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _has_two_full_periods(periods: List[int], n: int) -> List[bool]:
    # R's forecast::mstl() drops the seasonal periods that don't have at least
    # two full cycles in the data (period >= n / 2), see
    # https://github.com/robjhyndman/forecast/blob/master/R/mstl.R#L37
    return [p == 1 or p < n / 2 for p in periods]


def mstl(
    x: np.ndarray,  # time series
    period: Union[int, List[int]],  # season length
    blambda: Optional[float] = None,  # box-cox transform
    iterate: int = 2,  # number of iterations
    s_window: Optional[np.ndarray] = None,  # seasonal window
    stl_kwargs: Dict = dict(),
):
    origx = x
    n = len(x)
    if isinstance(period, int):
        msts = [period]
    else:
        msts = sorted(period)
    keep = _has_two_full_periods(msts, n)
    if not all(keep):
        warnings.warn("Dropping seasonal components with fewer than two full periods.")
        if s_window is not None and np.size(s_window) == len(msts):
            s_window = np.ravel(s_window)[keep]
        msts = [p for p, k in zip(msts, keep) if k]
    if s_window is None:
        # default seasonal windows 11, 15, 19, ... following R's
        # forecast::mstl() (7 + 4*k), see
        # https://github.com/robjhyndman/forecast/blob/master/R/mstl.R#L29
        # R hard-codes six windows and breaks beyond that; here we extend the
        # pattern to len(msts) so any number of seasonal periods works.
        # Values match R for up to six periods.
        s_window = 7 + 4 * np.arange(1, len(msts) + 1)
    if len(msts) == 1:
        iterate = 1
    if x.ndim == 2:
        x = x[:, 0]
    if np.isnan(x).any():
        raise Exception(
            "`mstl` cannot handle missing values. "
            "Please raise an issue to include this feature."
        )
    if blambda is not None:
        raise Exception(
            "`blambda` not implemented yet. "
            "Please raise an issue to include this feature."
        )
    stl_kwargs = {"seasonal_deg": 0, **stl_kwargs}
    if not msts:
        # no seasonal period can be estimated, so there's nothing to remove
        # from the series
        deseas = np.copy(x)
        trend = np.copy(x)
    elif msts[0] > 1:
        seas = np.zeros((len(msts), n))
        deseas = np.copy(x)
        if len(s_window) == 1:
            s_window = np.repeat(s_window, len(msts))
        for j in range(iterate):
            for i, seas_ in enumerate(msts, start=0):
                deseas = deseas + seas[i]
                fit = sm.tsa.STL(
                    deseas, period=seas_, seasonal=s_window[i], **stl_kwargs
                ).fit()
                seas[i] = fit.seasonal
                deseas = deseas - seas[i]
        trend = fit.trend
    else:
        try:
            from supersmoother import SuperSmoother
        except ImportError as e:
            print("supersmoother is required for mstl with period=1")
            raise e
        deseas = x
        t = 1 + np.arange(n)
        trend = SuperSmoother().fit(t, x).predict(t)
    deseas[np.isnan(origx)] = np.nan
    remainder = deseas - trend
    output = {"data": origx, "trend": trend}
    if msts and msts[0] > 1:
        if len(msts) == 1:
            output["seasonal"] = seas[0]
        else:
            for i, seas_ in enumerate(msts, start=0):
                output[f"seasonal{seas_}"] = seas[i]
    output["remainder"] = remainder
    return pd.DataFrame(output)
