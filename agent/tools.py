"""
Data analysis tools for the battery performance agent.

Each function accepts a validated pandas DataFrame and returns a plain Python
dict or list that is JSON-serialisable. Raw row-level data never leaves these
functions — only aggregated summaries are returned to the LLM.

All tools filter to SCHEDULE_TYPE == "cleared" by default, which represents
realized dispatch at actual market prices. The forecast-accuracy tool is the
exception: it compares "expected" vs "cleared" within the Historical scenario.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Required schema
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "SCENARIO_NAME",
    "SCHEDULE_TYPE",
    "START_DATETIME",
    "SOC",
    "CHARGE_ENERGY",
    "DISCHARGE_ENERGY",
    "PRICE_ENERGY",
    "REVENUE",
}

REQUIRED_SCENARIOS = {"historical", "perfect"}
REQUIRED_SCHEDULE_TYPES = {"cleared", "expected"}


def load_and_validate(filepath: str) -> pd.DataFrame:
    """Load a CSV and validate that it matches the expected schema.

    Raises ValueError with a descriptive message if validation fails so the
    caller gets a fast, clear error before any tool is called.
    """
    df = pd.read_csv(filepath, parse_dates=["START_DATETIME"])

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df["SCENARIO_NAME"] = df["SCENARIO_NAME"].str.strip().str.lower()
    df["SCHEDULE_TYPE"] = df["SCHEDULE_TYPE"].str.strip().str.lower()

    found_scenarios = set(df["SCENARIO_NAME"].unique())
    if not REQUIRED_SCENARIOS.issubset(found_scenarios):
        missing = REQUIRED_SCENARIOS - found_scenarios
        raise ValueError(
            f"Dataset must contain both 'historical' and 'perfect' scenarios. "
            f"Missing: {missing}"
        )

    found_types = set(df["SCHEDULE_TYPE"].unique())
    if "cleared" not in found_types:
        raise ValueError(
            "Dataset must contain 'cleared' schedule type for revenue analysis."
        )

    numeric_cols = ("SOC", "CHARGE_ENERGY", "DISCHARGE_ENERGY", "PRICE_ENERGY", "REVENUE")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    null_counts = df[list(numeric_cols)].isnull().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        total_rows = len(df)
        if any(v / total_rows > 0.05 for v in bad.values()):
            raise ValueError(
                f"Too many non-numeric values in columns {bad}. "
                "Check the dataset for corrupted rows."
            )
        # Small number of NaNs — forward-fill then back-fill within each scenario/schedule group
        df[list(numeric_cols)] = (
            df.groupby(["SCENARIO_NAME", "SCHEDULE_TYPE"])[list(numeric_cols)]
            .transform(lambda s: s.ffill().bfill())
        )

    df = df.sort_values("START_DATETIME")
    return df


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_float(value: Any, decimals: int = 2) -> float:
    """Round to decimals and replace NaN/Inf with 0.0."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 0.0
    return round(float(value), decimals)


def _cleared(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["SCHEDULE_TYPE"] == "cleared"].copy()


def _scenario(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["SCENARIO_NAME"] == name].copy()


# ---------------------------------------------------------------------------
# Tool 1: Financial summary
# ---------------------------------------------------------------------------

def get_financial_summary(df: pd.DataFrame) -> dict:
    """Return total revenue for each scenario and the absolute/relative gap.

    Uses cleared rows only (realized dispatch at actual market prices).

    Returns
    -------
    dict with keys:
        historical_revenue   : float ($)
        perfect_revenue      : float ($)
        gap_abs              : float ($ — Perfect minus Historical)
        gap_pct              : float (% of Perfect revenue)
        interval_count       : int (number of cleared intervals per scenario)
        analysis_period      : dict with start and end timestamps
    """
    cleared = _cleared(df)
    hist = _scenario(cleared, "historical")
    perf = _scenario(cleared, "perfect")

    hist_rev = hist["REVENUE"].sum()
    perf_rev = perf["REVENUE"].sum()
    gap_abs = perf_rev - hist_rev
    gap_pct = (gap_abs / abs(perf_rev) * 100) if perf_rev != 0 else 0.0

    start_dt = str(cleared["START_DATETIME"].min())
    end_dt = str(cleared["START_DATETIME"].max())

    return {
        "historical_revenue": _safe_float(hist_rev),
        "perfect_revenue": _safe_float(perf_rev),
        "gap_abs": _safe_float(gap_abs),
        "gap_pct": _safe_float(gap_pct),
        "interval_count": int(len(hist)),
        "analysis_period": {"start": start_dt, "end": end_dt},
    }


# ---------------------------------------------------------------------------
# Tool 2: Dispatch comparison
# ---------------------------------------------------------------------------

def get_dispatch_comparison(df: pd.DataFrame) -> dict:
    """Compare physical battery usage between Historical and Perfect scenarios.

    Returns per-scenario statistics on charge/discharge cycles, energy
    throughput, and state of charge. Differences reveal whether Perfect used
    the battery more intensively or differently than Historical.

    Returns
    -------
    dict with keys 'historical' and 'perfect', each containing:
        charge_cycles        : int (intervals with CHARGE_ENERGY > 0)
        discharge_cycles     : int (intervals with DISCHARGE_ENERGY > 0)
        idle_cycles          : int (no charge or discharge)
        total_charge_mwh     : float
        total_discharge_mwh  : float
        avg_soc              : float (MWh)
        min_soc              : float (MWh)
        max_soc              : float (MWh)
        net_revenue          : float ($)
    """
    cleared = _cleared(df)
    result = {}

    for scenario in ("historical", "perfect"):
        s = _scenario(cleared, scenario)
        charge_mask = s["CHARGE_ENERGY"] > 0
        discharge_mask = s["DISCHARGE_ENERGY"] > 0
        idle_mask = ~charge_mask & ~discharge_mask

        result[scenario] = {
            "charge_cycles": int(charge_mask.sum()),
            "discharge_cycles": int(discharge_mask.sum()),
            "idle_cycles": int(idle_mask.sum()),
            "total_charge_mwh": _safe_float(s["CHARGE_ENERGY"].sum()),
            "total_discharge_mwh": _safe_float(s["DISCHARGE_ENERGY"].sum()),
            "avg_soc": _safe_float(s["SOC"].mean()),
            "min_soc": _safe_float(s["SOC"].min()),
            "max_soc": _safe_float(s["SOC"].max()),
            "net_revenue": _safe_float(s["REVENUE"].sum()),
        }

    return result


# ---------------------------------------------------------------------------
# Tool 3: High-price intervals
# ---------------------------------------------------------------------------

def identify_high_price_intervals(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    """Return the top-N intervals by cleared price with side-by-side scenario comparison.

    For each high-price interval, shows what Historical earned versus what
    Perfect earned, the revenue gap, and each scenario's dispatch direction.
    A 'missed_opportunity' flag is set when Historical was idle or charging
    while Perfect was discharging.

    Returns
    -------
    List of dicts (sorted by price descending), each with:
        start_datetime       : str
        price                : float ($/MWh)
        hist_revenue         : float ($)
        perfect_revenue      : float ($)
        gap                  : float ($)
        hist_direction       : str ("charge", "discharge", or "idle")
        perfect_direction    : str ("charge", "discharge", or "idle")
        missed_opportunity   : bool
    """
    cleared = _cleared(df)

    hist = _scenario(cleared, "historical").set_index("START_DATETIME")
    perf = _scenario(cleared, "perfect").set_index("START_DATETIME")

    common_idx = hist.index.intersection(perf.index)
    if common_idx.empty:
        return []

    hist = hist.loc[common_idx]
    perf = perf.loc[common_idx]

    combined = pd.DataFrame({
        "price": hist["PRICE_ENERGY"],
        "hist_revenue": hist["REVENUE"],
        "perfect_revenue": perf["REVENUE"],
        "hist_charge": hist["CHARGE_ENERGY"],
        "hist_discharge": hist["DISCHARGE_ENERGY"],
        "perf_charge": perf["CHARGE_ENERGY"],
        "perf_discharge": perf["DISCHARGE_ENERGY"],
    })

    top = combined.nlargest(int(top_n), "price")

    def direction(charge, discharge):
        if discharge > charge and discharge > 0:
            return "discharge"
        elif charge > discharge and charge > 0:
            return "charge"
        elif charge > 0 or discharge > 0:
            return "discharge" if discharge >= charge else "charge"
        return "idle"

    rows = []
    for ts, row in top.iterrows():
        hist_dir = direction(row["hist_charge"], row["hist_discharge"])
        perf_dir = direction(row["perf_charge"], row["perf_discharge"])
        missed = perf_dir == "discharge" and hist_dir in ("idle", "charge")
        rows.append({
            "start_datetime": str(ts),
            "price": _safe_float(row["price"]),
            "hist_revenue": _safe_float(row["hist_revenue"]),
            "perfect_revenue": _safe_float(row["perfect_revenue"]),
            "gap": _safe_float(row["perfect_revenue"] - row["hist_revenue"]),
            "hist_direction": hist_dir,
            "perfect_direction": perf_dir,
            "missed_opportunity": bool(missed),
        })

    return rows


# ---------------------------------------------------------------------------
# Tool 4: SOC patterns
# ---------------------------------------------------------------------------

def analyze_soc_patterns(df: pd.DataFrame) -> dict:
    """Analyse state-of-charge distribution relative to price levels.

    The key diagnostic: was the battery full (unable to charge) during low
    prices, or empty (unable to discharge) during high prices? This surfaces
    the most common hidden driver of battery revenue loss.

    Returns
    -------
    dict with keys 'historical' and 'perfect', each containing:
        soc_quartiles        : dict (Q1, Q2, Q3, Q4 in MWh)
        pct_time_at_min      : float (% of intervals at or near SOC minimum)
        pct_time_at_max      : float (% of intervals at or near SOC maximum)
        avg_soc_top_price_quartile    : float (avg SOC during highest-price intervals)
        avg_soc_bottom_price_quartile : float (avg SOC during lowest-price intervals)
        soc_at_price_spikes  : list (SOC values at the top-10 price intervals)
    """
    cleared = _cleared(df)

    all_prices = cleared["PRICE_ENERGY"]
    price_q75 = all_prices.quantile(0.75)
    price_q25 = all_prices.quantile(0.25)
    top10_times = cleared.nlargest(10, "PRICE_ENERGY")["START_DATETIME"]  # keep as Series to preserve tz

    result = {}
    for scenario in ("historical", "perfect"):
        s = _scenario(cleared, scenario)

        soc = s["SOC"]
        soc_min = soc.min()
        soc_max = soc.max()
        soc_range = soc_max - soc_min
        threshold = soc_range * 0.05 if soc_range > 0 else 0.1

        pct_at_min = _safe_float((soc <= soc_min + threshold).mean() * 100)
        pct_at_max = _safe_float((soc >= soc_max - threshold).mean() * 100)

        high_price_mask = s["PRICE_ENERGY"] >= price_q75
        low_price_mask = s["PRICE_ENERGY"] <= price_q25

        avg_soc_high = _safe_float(s.loc[high_price_mask, "SOC"].mean()) if high_price_mask.any() else 0.0
        avg_soc_low = _safe_float(s.loc[low_price_mask, "SOC"].mean()) if low_price_mask.any() else 0.0

        spike_socs = s[s["START_DATETIME"].isin(top10_times)]["SOC"].tolist()

        result[scenario] = {
            "soc_quartiles": {
                "q1": _safe_float(soc.quantile(0.25)),
                "q2_median": _safe_float(soc.quantile(0.50)),
                "q3": _safe_float(soc.quantile(0.75)),
                "mean": _safe_float(soc.mean()),
            },
            "soc_range_mwh": {"min": _safe_float(soc_min), "max": _safe_float(soc_max)},
            "pct_time_at_min": pct_at_min,
            "pct_time_at_max": pct_at_max,
            "avg_soc_top_price_quartile": avg_soc_high,
            "avg_soc_bottom_price_quartile": avg_soc_low,
            "soc_at_top10_price_spikes": [_safe_float(v) for v in spike_socs],
        }

    return result


# ---------------------------------------------------------------------------
# Tool 5: Dispatch timing / direction conflicts
# ---------------------------------------------------------------------------

def compare_dispatch_timing(df: pd.DataFrame) -> dict:
    """Identify intervals where Historical and Perfect dispatched in opposite directions.

    Direction conflicts are the costliest error type: the battery not only misses
    the revenue it should have earned but also arrives at the wrong SOC for the
    next interval.

    A conflict is defined as:
        - Historical charging while Perfect discharging, OR
        - Historical discharging while Perfect charging.

    Returns
    -------
    dict with:
        total_intervals          : int
        conflict_count           : int
        conflict_pct             : float (% of all intervals)
        conflict_revenue_impact  : float ($ — sum of gaps at conflict intervals)
        hist_charging_perf_discharging : int (count)
        hist_discharging_perf_charging : int (count)
        top_conflicts            : list of dicts (top 10 by gap magnitude)
    """
    cleared = _cleared(df)

    hist = _scenario(cleared, "historical").set_index("START_DATETIME")
    perf = _scenario(cleared, "perfect").set_index("START_DATETIME")

    common_idx = hist.index.intersection(perf.index)
    if common_idx.empty:
        return {
            "total_intervals": 0,
            "conflict_count": 0,
            "conflict_pct": 0.0,
            "conflict_revenue_impact": 0.0,
            "hist_charging_perf_discharging": 0,
            "hist_discharging_perf_charging": 0,
            "top_conflicts": [],
        }

    hist = hist.loc[common_idx]
    perf = perf.loc[common_idx]

    hist_charge = hist["CHARGE_ENERGY"] > hist["DISCHARGE_ENERGY"]
    hist_discharge = hist["DISCHARGE_ENERGY"] > hist["CHARGE_ENERGY"]
    perf_charge = perf["CHARGE_ENERGY"] > perf["DISCHARGE_ENERGY"]
    perf_discharge = perf["DISCHARGE_ENERGY"] > perf["CHARGE_ENERGY"]

    type1 = hist_charge & perf_discharge  # Historical charging, Perfect discharging
    type2 = hist_discharge & perf_charge  # Historical discharging, Perfect charging

    conflict_mask = type1 | type2
    total = len(common_idx)
    conflict_count = int(conflict_mask.sum())

    revenue_gap = perf["REVENUE"] - hist["REVENUE"]
    conflict_impact = _safe_float(revenue_gap[conflict_mask].sum())

    combined = pd.DataFrame({
        "price": hist["PRICE_ENERGY"],
        "hist_revenue": hist["REVENUE"],
        "perfect_revenue": perf["REVENUE"],
        "gap": revenue_gap,
        "conflict_type": pd.Series(
            ["hist_charge/perf_discharge" if t1 else "hist_discharge/perf_charge" if t2 else "none"
             for t1, t2 in zip(type1, type2)],
            index=common_idx,
        ),
    })

    top_conflicts = (
        combined[conflict_mask]
        .nlargest(10, "gap")
        [["price", "hist_revenue", "perfect_revenue", "gap", "conflict_type"]]
        .reset_index()
        .rename(columns={"START_DATETIME": "start_datetime"})
        .assign(
            start_datetime=lambda x: x["start_datetime"].astype(str),
            price=lambda x: x["price"].round(2),
            hist_revenue=lambda x: x["hist_revenue"].round(2),
            perfect_revenue=lambda x: x["perfect_revenue"].round(2),
            gap=lambda x: x["gap"].round(2),
        )
        .to_dict(orient="records")
    )

    return {
        "total_intervals": total,
        "conflict_count": conflict_count,
        "conflict_pct": _safe_float(conflict_count / total * 100 if total > 0 else 0),
        "conflict_revenue_impact": conflict_impact,
        "hist_charging_perf_discharging": int(type1.sum()),
        "hist_discharging_perf_charging": int(type2.sum()),
        "top_conflicts": top_conflicts,
    }


# ---------------------------------------------------------------------------
# Tool 6: Price forecast accuracy
# ---------------------------------------------------------------------------

def get_price_forecast_accuracy(df: pd.DataFrame) -> dict:
    """Measure how well Historical price forecasts tracked actual cleared prices.

    Compares PRICE_ENERGY in Historical "expected" rows (the price the system
    forecast when placing bids) against Historical "cleared" rows (the price
    that actually cleared). Systematic bias here is a structural driver that
    dispatch and SOC improvements alone cannot fix.

    Returns
    -------
    dict with:
        n_intervals           : int
        mape                  : float (mean absolute percentage error, %)
        mean_error            : float ($ bias — positive = forecasts too high)
        rmse                  : float (root mean squared error, $/MWh)
        correlation           : float (Pearson r between forecast and actual)
        pct_underforecast     : float (% of intervals where forecast < actual)
        pct_overforecast      : float (% of intervals where forecast > actual)
        worst_underforecasts  : list (top 5 intervals where actual >> forecast)
        worst_overforecasts   : list (top 5 intervals where forecast >> actual)
    """
    hist = _scenario(df, "historical")
    expected = hist[hist["SCHEDULE_TYPE"] == "expected"].set_index("START_DATETIME")
    cleared = hist[hist["SCHEDULE_TYPE"] == "cleared"].set_index("START_DATETIME")

    common_idx = expected.index.intersection(cleared.index)
    if common_idx.empty:
        return {
            "n_intervals": 0,
            "mape": 0.0,
            "mean_error": 0.0,
            "rmse": 0.0,
            "correlation": 0.0,
            "pct_underforecast": 0.0,
            "pct_overforecast": 0.0,
            "worst_underforecasts": [],
            "worst_overforecasts": [],
        }

    forecast = expected.loc[common_idx, "PRICE_ENERGY"]
    actual = cleared.loc[common_idx, "PRICE_ENERGY"]

    error = forecast - actual
    abs_error = error.abs()

    nonzero = actual != 0
    mape_values = (abs_error[nonzero] / actual[nonzero].abs()) * 100
    mape = _safe_float(mape_values.mean())

    mean_error = _safe_float(error.mean())
    rmse = _safe_float((error**2).mean() ** 0.5)

    if forecast.std() > 0 and actual.std() > 0:
        correlation = _safe_float(forecast.corr(actual), decimals=4)
    else:
        correlation = 0.0

    underforecast = error < 0  # actual > forecast
    overforecast = error > 0   # forecast > actual

    def _top5(mask, sort_col):
        sub = pd.DataFrame(
            {"start_datetime": common_idx[mask].astype(str),
             "forecast_price": forecast[mask].round(2).values,
             "actual_price": actual[mask].round(2).values,
             "error": error[mask].round(2).values}
        )
        return sub.reindex(sub[sort_col].abs().nlargest(5).index).to_dict(orient="records")

    return {
        "n_intervals": len(common_idx),
        "mape": mape,
        "mean_error": mean_error,
        "rmse": rmse,
        "correlation": correlation,
        "pct_underforecast": _safe_float(underforecast.mean() * 100),
        "pct_overforecast": _safe_float(overforecast.mean() * 100),
        "worst_underforecasts": _top5(underforecast.values, "error"),
        "worst_overforecasts": _top5(overforecast.values, "error"),
    }


# ---------------------------------------------------------------------------
# Tool dispatcher (used by agent.py)
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "get_financial_summary": get_financial_summary,
    "get_dispatch_comparison": get_dispatch_comparison,
    "identify_high_price_intervals": identify_high_price_intervals,
    "analyze_soc_patterns": analyze_soc_patterns,
    "compare_dispatch_timing": compare_dispatch_timing,
    "get_price_forecast_accuracy": get_price_forecast_accuracy,
}
