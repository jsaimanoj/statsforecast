import warnings

import numpy as np
import pandas as pd
import pytest
from statsforecast.mstl import mstl


def test_mstl():
    """Test MSTL decomposition with electricity demand data."""

    # Load electricity demand data from online source
    url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
    df = pd.read_csv(url)

    df["Date"] = df["Date"].apply(
        lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
    )
    df["ds"] = df["Date"] + pd.to_timedelta((df["Period"] - 1) * 30, unit="m")

    electricity_data = df[["ds", "OperationalLessIndustrial"]]
    electricity_data.columns = ["ds", "y"]

    # Filter to first 149 days of 2012 for consistent test data
    start_date = pd.to_datetime("2012-01-01")
    end_date = start_date + pd.Timedelta("149D")
    mask = (electricity_data["ds"] >= start_date) & (electricity_data["ds"] < end_date)
    electricity_data = electricity_data[mask]

    # Resample to hourly frequency for analysis
    hourly_data = electricity_data.set_index("ds").resample("h").sum()

    seasonal_periods = [24, 24 * 7]  # Daily and weekly patterns
    decomposition = mstl(hourly_data["y"].values, seasonal_periods)

    assert decomposition is not None, "MSTL decomposition should return a result"
    assert hasattr(decomposition, "plot"), "Decomposition should have a plot method"


def _seasonal_series(n, periods=(7,), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 20 + 0.05 * t + rng.normal(size=n)
    for p in periods:
        y += np.sin(2 * np.pi * t / p)
    return y


def test_mstl_drops_periods_without_two_full_cycles():
    # https://github.com/Nixtla/statsforecast/issues/969
    # as in R's forecast::mstl(), periods with period >= n / 2 are dropped
    y = _seasonal_series(20, periods=(12,))
    with pytest.warns(UserWarning, match="fewer than two full periods"):
        res = mstl(y, period=12)
    assert res.columns.tolist() == ["data", "trend", "remainder"]
    np.testing.assert_allclose(res["trend"] + res["remainder"], y)


def test_mstl_keeps_periods_with_two_full_cycles():
    y = _seasonal_series(60, periods=(7, 30))
    with pytest.warns(UserWarning, match="fewer than two full periods"):
        res = mstl(y, period=[30, 7])
    pd.testing.assert_frame_equal(res, mstl(y, period=7))
    # the s_window of a dropped period is dropped with it
    with pytest.warns(UserWarning, match="fewer than two full periods"):
        res = mstl(y, period=[7, 30], s_window=np.array([9, 13]))
    pd.testing.assert_frame_equal(res, mstl(y, period=7, s_window=np.array([9])))
    # series with at least two full cycles of every period are unchanged
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = mstl(_seasonal_series(61, periods=(7, 30)), period=[7, 30])
    assert res.columns.tolist() == [
        "data",
        "trend",
        "seasonal7",
        "seasonal30",
        "remainder",
    ]
