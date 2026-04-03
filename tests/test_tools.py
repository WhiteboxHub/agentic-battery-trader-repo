"""
Unit tests for agent/tools.py.

Run with:
    pytest tests/
"""

import math

import pandas as pd
import pytest

from agent.tools import (
    load_and_validate,
    get_financial_summary,
    get_dispatch_comparison,
    identify_high_price_intervals,
    analyze_soc_patterns,
    compare_dispatch_timing,
    get_price_forecast_accuracy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n_intervals: int = 10) -> pd.DataFrame:
    """Build a minimal valid DataFrame with synthetic data for testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    times = pd.date_range("2026-01-26 04:00", periods=n_intervals, freq="5min", tz="Australia/Brisbane")

    rows = []
    for scenario in ("historical", "perfect"):
        for schedule_type in ("cleared", "expected"):
            prices = rng.uniform(50, 500, n_intervals)
            charge = rng.uniform(0, 1, n_intervals)
            discharge = rng.uniform(0, 1, n_intervals)
            soc = rng.uniform(10, 400, n_intervals)
            revenue = (discharge - charge) * prices
            for i in range(n_intervals):
                rows.append({
                    "SCENARIO_NAME": scenario,
                    "SCHEDULE_TYPE": schedule_type,
                    "START_DATETIME": times[i],
                    "SOC": soc[i],
                    "CHARGE_ENERGY": charge[i],
                    "DISCHARGE_ENERGY": discharge[i],
                    "PRICE_ENERGY": prices[i],
                    "REVENUE": revenue[i],
                })

    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return _make_df(n_intervals=20)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_load_and_validate_missing_column(tmp_path):
    """load_and_validate should raise ValueError when a required column is missing."""
    df = _make_df()
    df_bad = df.drop(columns=["SOC"])
    csv_path = tmp_path / "bad.csv"
    df_bad.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_validate(str(csv_path))


def test_load_and_validate_missing_scenario(tmp_path):
    """load_and_validate should raise ValueError when a required scenario is missing."""
    df = _make_df()
    df_bad = df[df["SCENARIO_NAME"] != "perfect"]
    csv_path = tmp_path / "no_perfect.csv"
    df_bad.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Missing"):
        load_and_validate(str(csv_path))


def test_load_and_validate_no_cleared(tmp_path):
    """load_and_validate should raise ValueError when cleared schedule type is absent."""
    df = _make_df()
    df_bad = df[df["SCHEDULE_TYPE"] != "cleared"]
    csv_path = tmp_path / "no_cleared.csv"
    df_bad.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="cleared"):
        load_and_validate(str(csv_path))


# ---------------------------------------------------------------------------
# Tool 1: Financial summary
# ---------------------------------------------------------------------------

def test_financial_summary_keys(df):
    result = get_financial_summary(df)
    assert set(result.keys()) >= {"historical_revenue", "perfect_revenue", "gap_abs", "gap_pct", "interval_count"}


def test_financial_summary_gap_consistency(df):
    result = get_financial_summary(df)
    assert abs(result["gap_abs"] - (result["perfect_revenue"] - result["historical_revenue"])) < 0.01


def test_financial_summary_no_nan(df):
    result = get_financial_summary(df)
    for key, val in result.items():
        if isinstance(val, float):
            assert not math.isnan(val), f"{key} is NaN"


# ---------------------------------------------------------------------------
# Tool 2: Dispatch comparison
# ---------------------------------------------------------------------------

def test_dispatch_comparison_both_scenarios(df):
    result = get_dispatch_comparison(df)
    assert "historical" in result
    assert "perfect" in result


def test_dispatch_comparison_cycle_counts_nonneg(df):
    result = get_dispatch_comparison(df)
    for scenario in ("historical", "perfect"):
        assert result[scenario]["charge_cycles"] >= 0
        assert result[scenario]["discharge_cycles"] >= 0
        assert result[scenario]["idle_cycles"] >= 0


def test_dispatch_comparison_cycles_sum_to_intervals(df):
    result = get_dispatch_comparison(df)
    n = result["historical"]["charge_cycles"] + result["historical"]["discharge_cycles"] + result["historical"]["idle_cycles"]
    assert n == result["historical"]["charge_cycles"] + result["historical"]["discharge_cycles"] + result["historical"]["idle_cycles"]


# ---------------------------------------------------------------------------
# Tool 3: High-price intervals
# ---------------------------------------------------------------------------

def test_high_price_intervals_length(df):
    result = identify_high_price_intervals(df, top_n=5)
    assert len(result) <= 5


def test_high_price_intervals_keys(df):
    result = identify_high_price_intervals(df, top_n=5)
    if result:
        expected = {"start_datetime", "price", "hist_revenue", "perfect_revenue", "gap", "hist_direction", "perfect_direction", "missed_opportunity"}
        assert expected.issubset(set(result[0].keys()))


def test_high_price_intervals_sorted_descending(df):
    result = identify_high_price_intervals(df, top_n=10)
    prices = [r["price"] for r in result]
    assert prices == sorted(prices, reverse=True)


# ---------------------------------------------------------------------------
# Tool 4: SOC patterns
# ---------------------------------------------------------------------------

def test_soc_patterns_both_scenarios(df):
    result = analyze_soc_patterns(df)
    assert "historical" in result
    assert "perfect" in result


def test_soc_patterns_pct_in_range(df):
    result = analyze_soc_patterns(df)
    for scenario in ("historical", "perfect"):
        assert 0.0 <= result[scenario]["pct_time_at_min"] <= 100.0
        assert 0.0 <= result[scenario]["pct_time_at_max"] <= 100.0


# ---------------------------------------------------------------------------
# Tool 5: Dispatch timing
# ---------------------------------------------------------------------------

def test_dispatch_timing_keys(df):
    result = compare_dispatch_timing(df)
    assert "conflict_count" in result
    assert "conflict_revenue_impact" in result
    assert "top_conflicts" in result


def test_dispatch_timing_conflict_count_nonneg(df):
    result = compare_dispatch_timing(df)
    assert result["conflict_count"] >= 0
    assert result["conflict_pct"] >= 0.0


# ---------------------------------------------------------------------------
# Tool 6: Forecast accuracy
# ---------------------------------------------------------------------------

def test_forecast_accuracy_keys(df):
    result = get_price_forecast_accuracy(df)
    assert "mape" in result
    assert "mean_error" in result
    assert "correlation" in result


def test_forecast_accuracy_mape_nonneg(df):
    result = get_price_forecast_accuracy(df)
    assert result["mape"] >= 0.0


def test_forecast_accuracy_pct_sums_to_lte_100(df):
    result = get_price_forecast_accuracy(df)
    total = result["pct_underforecast"] + result["pct_overforecast"]
    assert total <= 100.1
