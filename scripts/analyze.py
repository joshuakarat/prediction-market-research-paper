#!/usr/bin/env python3
"""Build the empirical results, tables, figures, and machine-readable manifest."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from o2p.core import (  # noqa: E402
    Quote,
    consistent_price_system,
    direct_member_fee,
    fused_distribution,
    maker_fee,
    project_simplex,
    quote_leg_prices,
    non_direct_member_fee,
)


DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SEED = 20260729
EPS = 1e-12
WILD_REPLICATIONS = 4_999
FEE_SCHEDULE_DATE = pd.Timestamp("2026-07-07T00:00:00Z")

NAVY = "#172A46"
BLUE = "#176B87"
TEAL = "#2A9D8F"
GOLD = "#D9A441"
RED = "#C24B40"
GREY = "#6B7280"
LIGHT_GREY = "#D8DEE8"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E8ECF2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "legend.frameon": False,
        }
    )


def _fee_scenarios(row: pd.Series) -> pd.Series:
    """Return leg-level taker costs and resting-order fee sensitivities."""

    direct = Quote(float(row.direct_bid), float(row.direct_ask))
    low = (
        Quote(float(row.low_bid), float(row.low_ask))
        if pd.notna(row.low_bid)
        else None
    )
    high = (
        Quote(float(row.high_bid), float(row.high_ask))
        if pd.notna(row.high_bid)
        else None
    )
    prices = quote_leg_prices(
        str(row.lock_direction), str(row.state_kind), direct, low, high
    )
    direct_member_fees = np.array(
        [direct_member_fee(float(price)) for price in prices], dtype=float
    )
    non_direct_fees = np.array(
        [non_direct_member_fee(float(price)) for price in prices], dtype=float
    )
    maker_fees = np.array(
        [maker_fee(float(price)) for price in prices], dtype=float
    )
    taker_order = np.sort(direct_member_fees)[::-1]
    one_maker_fees = direct_member_fees.sum() - taker_order[:1].sum()
    two_maker_fees = direct_member_fees.sum() - taker_order[:2].sum()
    all_maker_fees = float(maker_fees.sum())
    return pd.Series(
        {
            "member_fee": float(direct_member_fees.sum()),
            "non_direct_fee": float(non_direct_fees.sum()),
            "one_maker_fee": float(one_maker_fees),
            "two_maker_fee": float(two_maker_fees),
            "all_maker_fee": all_maker_fees,
            "legs": int(len(prices)),
        }
    )


def build_panel() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    contracts = pd.read_csv(DATA / "contract_map.csv")
    states = pd.read_csv(DATA / "range_states.csv")
    quotes = pd.read_csv(DATA / "quotes.csv.gz")

    quote_columns = [
        "ticker",
        "end_period_ts",
        "yes_bid",
        "yes_ask",
        "last_price",
        "volume",
        "open_interest",
    ]
    direct_quotes = quotes[quote_columns].rename(
        columns={
            "ticker": "direct_ticker",
            "yes_bid": "direct_bid",
            "yes_ask": "direct_ask",
            "last_price": "direct_last",
            "volume": "direct_candle_volume",
            "open_interest": "direct_candle_open_interest",
        }
    )
    low_quotes = quotes[quote_columns[:4]].rename(
        columns={
            "ticker": "low_ticker",
            "yes_bid": "low_bid",
            "yes_ask": "low_ask",
        }
    )
    high_quotes = quotes[quote_columns[:4]].rename(
        columns={
            "ticker": "high_ticker",
            "yes_bid": "high_bid",
            "yes_ask": "high_ask",
        }
    )

    panel = contracts.merge(direct_quotes, on="direct_ticker", how="inner")
    panel = panel.merge(
        low_quotes, on=["low_ticker", "end_period_ts"], how="left"
    )
    panel = panel.merge(
        high_quotes, on=["high_ticker", "end_period_ts"], how="left"
    )

    panel["timestamp"] = pd.to_datetime(
        panel["end_period_ts"], unit="s", utc=True
    )
    panel["close_timestamp"] = pd.to_datetime(
        panel["direct_close_time"], utc=True
    )
    panel["minutes_to_close"] = (
        panel["close_timestamp"] - panel["timestamp"]
    ).dt.total_seconds() / 60.0
    panel = panel.loc[panel["minutes_to_close"].between(0, 180)].copy()

    required = (
        (panel["state_kind"].eq("left_tail") & panel["high_bid"].notna())
        | (panel["state_kind"].eq("right_tail") & panel["low_bid"].notna())
        | (
            panel["state_kind"].eq("interior")
            & panel["low_bid"].notna()
            & panel["high_bid"].notna()
        )
    )
    availability = panel[
        [
            "index_name",
            "event_ticker",
            "state_index",
            "state_kind",
            "direct_ticker",
            "timestamp",
            "end_period_ts",
            "minutes_to_close",
            "direct_bid",
            "direct_ask",
            "direct_candle_volume",
        ]
    ].copy()
    availability["synchronous"] = required.to_numpy(dtype=bool)
    availability["direct_mid"] = 0.5 * (
        availability["direct_bid"] + availability["direct_ask"]
    )
    availability["direct_spread"] = (
        availability["direct_ask"] - availability["direct_bid"]
    )
    panel = panel.loc[required].copy()

    panel["direct_mid"] = 0.5 * (panel["direct_bid"] + panel["direct_ask"])
    panel["direct_spread"] = panel["direct_ask"] - panel["direct_bid"]
    panel["low_mid"] = 0.5 * (panel["low_bid"] + panel["low_ask"])
    panel["high_mid"] = 0.5 * (panel["high_bid"] + panel["high_ask"])

    left = panel["state_kind"].eq("left_tail")
    right = panel["state_kind"].eq("right_tail")
    interior = panel["state_kind"].eq("interior")
    panel["synthetic_bid"] = np.nan
    panel["synthetic_ask"] = np.nan
    panel.loc[left, "synthetic_bid"] = 1.0 - panel.loc[left, "high_ask"]
    panel.loc[left, "synthetic_ask"] = 1.0 - panel.loc[left, "high_bid"]
    panel.loc[right, "synthetic_bid"] = panel.loc[right, "low_bid"]
    panel.loc[right, "synthetic_ask"] = panel.loc[right, "low_ask"]
    panel.loc[interior, "synthetic_bid"] = (
        panel.loc[interior, "low_bid"] - panel.loc[interior, "high_ask"]
    )
    panel.loc[interior, "synthetic_ask"] = (
        panel.loc[interior, "low_ask"] - panel.loc[interior, "high_bid"]
    )
    panel["synthetic_mid"] = 0.5 * (
        panel["synthetic_bid"] + panel["synthetic_ask"]
    )
    panel["synthetic_spread"] = (
        panel["synthetic_ask"] - panel["synthetic_bid"]
    )
    panel["mid_gap"] = panel["direct_mid"] - panel["synthetic_mid"]
    panel["abs_mid_gap"] = panel["mid_gap"].abs()
    panel["reference_mid"] = (
        0.5
        * (
            panel["direct_mid"]
            + panel["synthetic_mid"].clip(lower=0.0, upper=1.0)
        )
    ).clip(lower=0.0, upper=1.0)
    panel["relative_abs_gap"] = panel["abs_mid_gap"] / panel[
        "reference_mid"
    ].clip(lower=0.01)
    panel["price_band"] = pd.cut(
        panel["reference_mid"],
        bins=[-EPS, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0 + EPS],
        labels=["0-2.5c", "2.5-5c", "5-10c", "10-25c", "25-50c", "50-100c"],
        include_lowest=True,
    )
    panel["combined_spread"] = (
        panel["direct_spread"] + panel["synthetic_spread"]
    )

    direct_cheap = panel["synthetic_bid"] - panel["direct_ask"]
    synthetic_cheap = panel["direct_bid"] - panel["synthetic_ask"]
    panel["gross_lock"] = np.maximum.reduce(
        [direct_cheap.to_numpy(), synthetic_cheap.to_numpy(), np.zeros(len(panel))]
    )
    panel.loc[panel["gross_lock"].le(EPS), "gross_lock"] = 0.0
    panel["lock_direction"] = np.select(
        [
            (direct_cheap >= synthetic_cheap) & (direct_cheap > EPS),
            synthetic_cheap > EPS,
        ],
        ["buy_direct_sell_synthetic", "buy_synthetic_sell_direct"],
        default="none",
    )

    candidates = panel["gross_lock"].gt(0)
    panel["member_fee"] = 0.0
    panel["non_direct_fee"] = 0.0
    panel["one_maker_fee"] = 0.0
    panel["two_maker_fee"] = 0.0
    panel["all_maker_fee"] = 0.0
    panel["legs"] = 0
    if candidates.any():
        fee_scenarios = panel.loc[candidates].apply(_fee_scenarios, axis=1)
        panel.loc[candidates, fee_scenarios.columns] = fee_scenarios.to_numpy()
    panel["member_net_lock"] = panel["gross_lock"] - panel["member_fee"]
    panel["non_direct_net_lock"] = (
        panel["gross_lock"] - panel["non_direct_fee"]
    )
    panel["one_maker_net_lock"] = (
        panel["gross_lock"] - panel["one_maker_fee"]
    )
    panel["two_maker_net_lock"] = (
        panel["gross_lock"] - panel["two_maker_fee"]
    )
    panel["all_maker_net_lock"] = (
        panel["gross_lock"] - panel["all_maker_fee"]
    )
    panel["member_positive"] = panel["member_net_lock"].gt(EPS)
    panel["non_direct_positive"] = panel["non_direct_net_lock"].gt(EPS)
    panel["one_maker_positive"] = panel["one_maker_net_lock"].gt(EPS)
    panel["two_maker_positive"] = panel["two_maker_net_lock"].gt(EPS)
    panel["all_maker_positive"] = panel["all_maker_net_lock"].gt(EPS)
    panel["guaranteed_payout"] = np.select(
        [
            panel["lock_direction"].eq("buy_direct_sell_synthetic"),
            panel["lock_direction"].eq("buy_synthetic_sell_direct"),
        ],
        [1.0, 2.0],
        default=np.nan,
    )
    for label, fee_column, net_column in [
        ("gross", None, "gross_lock"),
        ("member", "member_fee", "member_net_lock"),
        ("one_maker", "one_maker_fee", "one_maker_net_lock"),
        ("two_maker", "two_maker_fee", "two_maker_net_lock"),
        ("all_maker", "all_maker_fee", "all_maker_net_lock"),
        ("non_direct", "non_direct_fee", "non_direct_net_lock"),
    ]:
        fee = 0.0 if fee_column is None else panel[fee_column]
        capital = panel["guaranteed_payout"] - panel["gross_lock"] + fee
        panel[f"{label}_cash_commitment"] = capital
        panel[f"{label}_return_on_cash"] = np.where(
            candidates & capital.gt(EPS), panel[net_column] / capital, np.nan
        )
    panel["date"] = panel["close_timestamp"].dt.date.astype(str)

    panel["time_bucket"] = pd.cut(
        panel["minutes_to_close"],
        bins=[-1e-9, 30, 60, 120, 180],
        labels=["0–30", "30–60", "60–120", "120–180"],
        include_lowest=True,
    )
    invalid = (
        (panel["direct_bid"] > panel["direct_ask"])
        | (panel["synthetic_bid"] > panel["synthetic_ask"])
        | ~panel["direct_bid"].between(0, 1)
        | ~panel["direct_ask"].between(0, 1)
    )
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} invalid quote rows")
    panel = panel.sort_values(
        ["event_ticker", "state_index", "end_period_ts"]
    ).reset_index(drop=True)
    return panel, contracts, states, availability


def describe_panel(
    panel: pd.DataFrame, contracts: pd.DataFrame, states: pd.DataFrame
) -> dict[str, object]:
    event_state_counts = contracts.groupby("event_ticker").size()
    quote_counts = panel.groupby(["event_ticker", "state_index"]).size()
    event_metrics = panel.groupby("event_ticker").agg(
        median_abs_gap=("abs_mid_gap", "median"),
        gross_candidate_rate=("gross_lock", lambda x: float((x > 0).mean())),
        member_positive_rate=("member_positive", "mean"),
        one_maker_positive_rate=("one_maker_positive", "mean"),
        non_direct_positive_rate=("non_direct_positive", "mean"),
    )
    rng = np.random.default_rng(SEED + 11)
    bootstrap_indices = rng.integers(
        0, len(event_metrics), size=(10_000, len(event_metrics))
    )

    def event_mean_interval(column: str) -> tuple[float, float]:
        values = event_metrics[column].to_numpy()
        boot = values[bootstrap_indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        return float(low), float(high)

    gap_ci = event_mean_interval("median_abs_gap")
    gross_ci = event_mean_interval("gross_candidate_rate")
    member_ci = event_mean_interval("member_positive_rate")
    one_maker_ci = event_mean_interval("one_maker_positive_rate")
    non_direct_ci = event_mean_interval("non_direct_positive_rate")
    quote_weights = quote_counts / quote_counts.sum()
    candidate = panel["gross_lock"].gt(0)
    member_candidate = panel["member_positive"]
    non_direct_candidate = panel["non_direct_positive"]
    overview = {
        "sample_start": str(panel["timestamp"].min()),
        "sample_end": str(panel["timestamp"].max()),
        "events": int(panel["event_ticker"].nunique()),
        "trading_dates": int(panel["date"].nunique()),
        "events_sp500": int(
            panel.loc[panel["index_name"].eq("S&P 500"), "event_ticker"].nunique()
        ),
        "events_nasdaq100": int(
            panel.loc[
                panel["index_name"].eq("Nasdaq-100"), "event_ticker"
            ].nunique()
        ),
        "exact_matched_states": int(len(contracts)),
        "full_range_states": int(len(states)),
        "synchronous_state_minutes": int(len(panel)),
        "states_with_synchronous_quotes": int(
            panel[["event_ticker", "state_index"]].drop_duplicates().shape[0]
        ),
        "median_minutes_per_quoted_state": float(quote_counts.median()),
        "mean_minutes_per_quoted_state": float(quote_counts.mean()),
        "p75_minutes_per_quoted_state": float(quote_counts.quantile(0.75)),
        "p90_minutes_per_quoted_state": float(quote_counts.quantile(0.90)),
        "p95_minutes_per_quoted_state": float(quote_counts.quantile(0.95)),
        "maximum_minutes_per_quoted_state": int(quote_counts.max()),
        "top_decile_state_pair_row_share": float(
            quote_counts.nlargest(math.ceil(0.10 * len(quote_counts))).sum()
            / quote_counts.sum()
        ),
        "inverse_hhi_effective_state_pairs": float(
            1.0 / quote_weights.pow(2).sum()
        ),
        "full_partition_events": int((event_state_counts == 30).sum()),
        "direct_synthetic_midpoint_correlation": float(
            panel[["direct_mid", "synthetic_mid"]].corr().iloc[0, 1]
        ),
        "median_absolute_midpoint_gap": float(panel["abs_mid_gap"].median()),
        "p90_absolute_midpoint_gap": float(panel["abs_mid_gap"].quantile(0.90)),
        "p95_absolute_midpoint_gap": float(panel["abs_mid_gap"].quantile(0.95)),
        "midpoint_gap_at_least_1c_rate": float(panel["abs_mid_gap"].ge(0.01).mean()),
        "midpoint_gap_at_least_2c_rate": float(panel["abs_mid_gap"].ge(0.02).mean()),
        "gross_quoted_lock_candidates": int(panel["gross_lock"].gt(0).sum()),
        "gross_quoted_lock_candidate_rate": float(
            panel["gross_lock"].gt(0).mean()
        ),
        "member_fee_positive_candidates": int(panel["member_positive"].sum()),
        "member_fee_positive_rate": float(panel["member_positive"].mean()),
        "one_maker_positive_candidates": int(panel["one_maker_positive"].sum()),
        "one_maker_positive_rate": float(panel["one_maker_positive"].mean()),
        "two_maker_positive_candidates": int(panel["two_maker_positive"].sum()),
        "two_maker_positive_rate": float(panel["two_maker_positive"].mean()),
        "all_maker_positive_candidates": int(panel["all_maker_positive"].sum()),
        "all_maker_positive_rate": float(panel["all_maker_positive"].mean()),
        "non_direct_positive_candidates": int(
            panel["non_direct_positive"].sum()
        ),
        "non_direct_positive_rate": float(panel["non_direct_positive"].mean()),
        "gross_candidates_below_5c_share": float(
            panel.loc[candidate, "reference_mid"].le(0.05).mean()
        ),
        "member_candidates_below_5c_share": float(
            panel.loc[member_candidate, "reference_mid"].le(0.05).mean()
        ),
        "non_direct_candidates_below_5c_share": float(
            panel.loc[non_direct_candidate, "reference_mid"].le(0.05).mean()
        ),
        "events_with_gross_candidate": int(
            panel.loc[panel["gross_lock"].gt(0), "event_ticker"].nunique()
        ),
        "events_with_member_positive_candidate": int(
            panel.loc[panel["member_positive"], "event_ticker"].nunique()
        ),
        "events_with_one_maker_positive_candidate": int(
            panel.loc[panel["one_maker_positive"], "event_ticker"].nunique()
        ),
        "events_with_non_direct_positive_candidate": int(
            panel.loc[panel["non_direct_positive"], "event_ticker"].nunique()
        ),
        "event_balanced_median_abs_gap_mean": float(
            event_metrics["median_abs_gap"].mean()
        ),
        "event_balanced_median_abs_gap_ci_low": gap_ci[0],
        "event_balanced_median_abs_gap_ci_high": gap_ci[1],
        "event_balanced_gross_candidate_rate_mean": float(
            event_metrics["gross_candidate_rate"].mean()
        ),
        "event_balanced_gross_candidate_rate_ci_low": gross_ci[0],
        "event_balanced_gross_candidate_rate_ci_high": gross_ci[1],
        "event_balanced_member_positive_rate_mean": float(
            event_metrics["member_positive_rate"].mean()
        ),
        "event_balanced_member_positive_rate_ci_low": member_ci[0],
        "event_balanced_member_positive_rate_ci_high": member_ci[1],
        "event_balanced_one_maker_positive_rate_mean": float(
            event_metrics["one_maker_positive_rate"].mean()
        ),
        "event_balanced_one_maker_positive_rate_ci_low": one_maker_ci[0],
        "event_balanced_one_maker_positive_rate_ci_high": one_maker_ci[1],
        "event_balanced_non_direct_positive_rate_mean": float(
            event_metrics["non_direct_positive_rate"].mean()
        ),
        "event_balanced_non_direct_positive_rate_ci_low": non_direct_ci[0],
        "event_balanced_non_direct_positive_rate_ci_high": non_direct_ci[1],
    }
    return overview


def grouped_summary(panel: pd.DataFrame) -> pd.DataFrame:
    def summarize(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "observations": len(frame),
                "events": frame["event_ticker"].nunique(),
                "states": frame[["event_ticker", "state_index"]]
                .drop_duplicates()
                .shape[0],
                "median_abs_gap": frame["abs_mid_gap"].median(),
                "p90_abs_gap": frame["abs_mid_gap"].quantile(0.90),
                "mean_direct_spread": frame["direct_spread"].mean(),
                "mean_synthetic_spread": frame["synthetic_spread"].mean(),
                "gross_candidate_rate": frame["gross_lock"].gt(0).mean(),
                "member_positive_rate": frame["member_positive"].mean(),
                "one_maker_positive_rate": frame["one_maker_positive"].mean(),
                "two_maker_positive_rate": frame["two_maker_positive"].mean(),
                "non_direct_positive_rate": frame["non_direct_positive"].mean(),
                "median_gross_lock_if_positive": frame.loc[
                    frame["gross_lock"].gt(0), "gross_lock"
                ].median(),
            }
        )

    rows: list[pd.Series] = []
    overall = summarize(panel)
    overall["index_name"] = "All"
    overall["time_bucket"] = "All"
    rows.append(overall)
    for index_name, frame in panel.groupby("index_name", observed=True):
        row = summarize(frame)
        row["index_name"] = index_name
        row["time_bucket"] = "All"
        rows.append(row)
    for (index_name, bucket), frame in panel.groupby(
        ["index_name", "time_bucket"], observed=True
    ):
        row = summarize(frame)
        row["index_name"] = index_name
        row["time_bucket"] = str(bucket)
        rows.append(row)
    result = pd.DataFrame(rows)
    columns = ["index_name", "time_bucket"] + [
        column
        for column in result.columns
        if column not in {"index_name", "time_bucket"}
    ]
    return result[columns]


@dataclass
class RegressionResult:
    outcome: str
    specification: str
    n: int
    clusters: int
    coefficient: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float
    r_squared: float
    cluster_level: str = "date"
    wild_cluster_p_value: float = float("nan")


def cluster_ols(
    frame: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    cluster: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    columns = [outcome, cluster] + regressors
    data = frame[columns].dropna()
    y = data[outcome].to_numpy(dtype=float)
    x = np.column_stack(
        [np.ones(len(data))]
        + [data[column].to_numpy(dtype=float) for column in regressors]
    )
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    groups = data[cluster].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k = x.shape
    g = len(unique_groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 else 1.0
    covariance = correction * xtx_inv @ meat @ xtx_inv
    fitted = x @ beta
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / total if total > 0 else 0.0
    return beta, covariance, r_squared, g


def wild_cluster_p_value(
    frame: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    cluster: str,
    tested_regressor: str,
    *,
    null_value: float = 0.0,
    replications: int = WILD_REPLICATIONS,
    seed_offset: int = 0,
) -> float:
    """Restricted Rademacher wild-cluster bootstrap-t p-value for OLS."""

    columns = [outcome, cluster] + regressors
    data = frame[columns].dropna().reset_index(drop=True)
    y = data[outcome].to_numpy(dtype=float)
    x = np.column_stack(
        [np.ones(len(data))]
        + [data[column].to_numpy(dtype=float) for column in regressors]
    )
    tested_position = 1 + regressors.index(tested_regressor)
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    groups = data[cluster].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    group_xtx = []
    group_scores = []
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in unique_groups:
        mask = groups == group
        x_group = x[mask]
        score = x_group.T @ residual[mask]
        group_xtx.append(x_group.T @ x_group)
        group_scores.append(score)
        meat += np.outer(score, score)
    n, k = x.shape
    group_count = len(unique_groups)
    correction = (
        (group_count / (group_count - 1)) * ((n - 1) / (n - k))
        if group_count > 1
        else 1.0
    )
    covariance = correction * xtx_inverse @ meat @ xtx_inverse
    standard_error = math.sqrt(
        max(float(covariance[tested_position, tested_position]), 0.0)
    )
    if standard_error <= 0:
        return float("nan")
    observed_t = (float(beta[tested_position]) - null_value) / standard_error

    remaining = [position for position in range(k) if position != tested_position]
    x_restricted = x[:, remaining]
    y_restricted = y - null_value * x[:, tested_position]
    nuisance = np.linalg.pinv(x_restricted.T @ x_restricted) @ x_restricted.T @ y_restricted
    restricted_beta = np.empty(k)
    restricted_beta[tested_position] = null_value
    restricted_beta[remaining] = nuisance
    restricted_residual = y - x @ restricted_beta

    group_xtx_array = np.stack(group_xtx)
    restricted_scores = np.stack(
        [
            x[groups == group].T @ restricted_residual[groups == group]
            for group in unique_groups
        ]
    )
    rng = np.random.default_rng(SEED + 10_000 + seed_offset)
    exceedances = 0
    valid = 0
    for _ in range(replications):
        weights = rng.choice(np.array([-1.0, 1.0]), size=group_count)
        delta = xtx_inverse @ np.einsum(
            "g,gk->k", weights, restricted_scores
        )
        bootstrap_scores = (
            weights[:, None] * restricted_scores
            - np.einsum("gij,j->gi", group_xtx_array, delta)
        )
        bootstrap_meat = bootstrap_scores.T @ bootstrap_scores
        bootstrap_covariance = (
            correction * xtx_inverse @ bootstrap_meat @ xtx_inverse
        )
        bootstrap_se = math.sqrt(
            max(
                float(
                    bootstrap_covariance[
                        tested_position, tested_position
                    ]
                ),
                0.0,
            )
        )
        if bootstrap_se <= 0:
            continue
        bootstrap_t = delta[tested_position] / bootstrap_se
        exceedances += int(abs(bootstrap_t) >= abs(observed_t))
        valid += 1
    return float((exceedances + 1) / (valid + 1))


def wild_cluster_joint_p_value(
    frame: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    cluster: str,
    tested_regressors: list[str],
    *,
    replications: int = WILD_REPLICATIONS,
    seed_offset: int = 0,
) -> float:
    """Restricted Rademacher wild-cluster p-value for a joint OLS test."""

    columns = [outcome, cluster] + regressors
    data = frame[columns].dropna().reset_index(drop=True)
    y = data[outcome].to_numpy(dtype=float)
    x = np.column_stack(
        [np.ones(len(data))]
        + [data[column].to_numpy(dtype=float) for column in regressors]
    )
    tested_positions = np.array(
        [1 + regressors.index(regressor) for regressor in tested_regressors]
    )
    remaining = np.array(
        [position for position in range(x.shape[1]) if position not in tested_positions]
    )
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    groups = data[cluster].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    group_xtx = []
    full_scores = []
    for group in unique_groups:
        mask = groups == group
        x_group = x[mask]
        group_xtx.append(x_group.T @ x_group)
        full_scores.append(x_group.T @ residual[mask])
    group_xtx_array = np.stack(group_xtx)
    meat = np.stack(full_scores).T @ np.stack(full_scores)
    n, k = x.shape
    group_count = len(unique_groups)
    correction = (
        (group_count / (group_count - 1)) * ((n - 1) / (n - k))
        if group_count > 1
        else 1.0
    )
    covariance = correction * xtx_inverse @ meat @ xtx_inverse
    tested_covariance = covariance[np.ix_(tested_positions, tested_positions)]
    observed = float(
        beta[tested_positions]
        @ np.linalg.pinv(tested_covariance)
        @ beta[tested_positions]
        / len(tested_positions)
    )

    x_restricted = x[:, remaining]
    nuisance = (
        np.linalg.pinv(x_restricted.T @ x_restricted)
        @ x_restricted.T
        @ y
    )
    restricted_beta = np.zeros(k)
    restricted_beta[remaining] = nuisance
    restricted_residual = y - x @ restricted_beta
    restricted_scores = np.stack(
        [
            x[groups == group].T @ restricted_residual[groups == group]
            for group in unique_groups
        ]
    )

    rng = np.random.default_rng(SEED + 20_000 + seed_offset)
    exceedances = 0
    valid = 0
    for _ in range(replications):
        weights = rng.choice(np.array([-1.0, 1.0]), size=group_count)
        delta = xtx_inverse @ np.einsum("g,gk->k", weights, restricted_scores)
        bootstrap_scores = (
            weights[:, None] * restricted_scores
            - np.einsum("gij,j->gi", group_xtx_array, delta)
        )
        bootstrap_meat = bootstrap_scores.T @ bootstrap_scores
        bootstrap_covariance = (
            correction * xtx_inverse @ bootstrap_meat @ xtx_inverse
        )
        tested_bootstrap_covariance = bootstrap_covariance[
            np.ix_(tested_positions, tested_positions)
        ]
        bootstrap_statistic = float(
            delta[tested_positions]
            @ np.linalg.pinv(tested_bootstrap_covariance)
            @ delta[tested_positions]
            / len(tested_positions)
        )
        if not np.isfinite(bootstrap_statistic):
            continue
        exceedances += int(bootstrap_statistic >= observed)
        valid += 1
    return float((exceedances + 1) / (valid + 1))


def prepare_error_correction(
    panel: pd.DataFrame, interval_seconds: int = 60
) -> pd.DataFrame:
    keys = ["event_ticker", "state_index"]
    data = panel.sort_values(keys + ["end_period_ts"]).copy()
    grouped = data.groupby(keys, sort=False)
    data["previous_ts"] = grouped["end_period_ts"].shift(1)
    data["previous_previous_ts"] = grouped["end_period_ts"].shift(2)
    data["previous_previous_previous_ts"] = grouped["end_period_ts"].shift(3)
    data["lag_gap"] = grouped["mid_gap"].shift(1)
    data["lag2_gap"] = grouped["mid_gap"].shift(2)
    data["lag3_gap"] = grouped["mid_gap"].shift(3)
    data["delta_direct"] = grouped["direct_mid"].diff()
    data["delta_synthetic"] = grouped["synthetic_mid"].diff()
    data["lag_delta_direct"] = grouped["delta_direct"].shift(1)
    data["lag_delta_synthetic"] = grouped["delta_synthetic"].shift(1)
    data["common_mid"] = (
        0.5
        * (
            data["direct_mid"]
            + data["synthetic_mid"].clip(lower=0.0, upper=1.0)
        )
    ).clip(lower=0.0, upper=1.0)
    data["lag_common_mid"] = grouped["common_mid"].shift(1)
    data["lag_common_mid_sq"] = data["lag_common_mid"] ** 2
    current_consecutive = (
        data["end_period_ts"] - data["previous_ts"]
    ).eq(interval_seconds)
    previous_consecutive = (
        data["previous_ts"] - data["previous_previous_ts"]
    ).eq(interval_seconds)
    third_consecutive = (
        data["previous_previous_ts"] - data["previous_previous_previous_ts"]
    ).eq(interval_seconds)
    data["lag_pair_valid"] = current_consecutive & previous_consecutive
    data["lag_triplet_valid"] = (
        current_consecutive & previous_consecutive & third_consecutive
    )
    data["nasdaq"] = data["index_name"].eq("Nasdaq-100").astype(float)
    data["horizon_scaled"] = data["minutes_to_close"] / 180.0
    return data.loc[current_consecutive].copy()


def estimate_error_correction(
    frame: pd.DataFrame, specification: str, *, wild: bool = False
) -> tuple[list[RegressionResult], dict[str, float]]:
    regressors = [
        "lag_gap",
        "nasdaq",
        "horizon_scaled",
        "lag_common_mid",
        "lag_common_mid_sq",
    ]
    results: list[RegressionResult] = []
    for outcome in ["delta_direct", "delta_synthetic"]:
        beta, covariance, r_squared, clusters = cluster_ols(
            frame, outcome, regressors, "date"
        )
        standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
        df = max(clusters - 1, 1)
        critical = float(stats.t.ppf(0.975, df=df))
        coefficient = float(beta[1])
        p_value = float(
            2
            * stats.t.sf(
                abs(coefficient / standard_error), df=df
            )
        ) if standard_error > 0 else float("nan")
        results.append(
            RegressionResult(
                outcome=outcome,
                specification=specification,
                n=int(frame[[outcome] + regressors].dropna().shape[0]),
                clusters=clusters,
                coefficient=coefficient,
                standard_error=standard_error,
                ci_low=coefficient - critical * standard_error,
                ci_high=coefficient + critical * standard_error,
                p_value=p_value,
                r_squared=float(r_squared),
                cluster_level="date",
                wild_cluster_p_value=(
                    wild_cluster_p_value(
                        frame,
                        outcome,
                        regressors,
                        "date",
                        "lag_gap",
                        seed_offset=31 if outcome == "delta_direct" else 32,
                    )
                    if wild
                    else float("nan")
                ),
            )
        )

    direct = results[0].coefficient
    synthetic = results[1].coefficient
    denominator = synthetic - direct
    direct_share = synthetic / denominator if denominator != 0 else float("nan")
    gap_speed = direct - synthetic
    persistence = 1.0 + gap_speed
    half_life = (
        math.log(0.5) / math.log(persistence)
        if 0 < persistence < 1
        else float("nan")
    )
    derived = {
        "direct_price_discovery_share": float(direct_share),
        "gap_adjustment_coefficient": float(gap_speed),
        "gap_persistence": float(persistence),
        "gap_half_life_minutes": float(half_life),
    }
    return results, derived


def bootstrap_price_discovery(
    frame: pd.DataFrame,
    replications: int = 10_000,
) -> dict[str, float]:
    """Date-cluster bootstrap for the OLS two-equation component share."""

    regressors = [
        "lag_gap",
        "nasdaq",
        "horizon_scaled",
        "lag_common_mid",
        "lag_common_mid_sq",
    ]
    columns = [
        "date",
        "delta_direct",
        "delta_synthetic",
        *regressors,
    ]
    data = frame[columns].dropna().copy()
    x = np.column_stack(
        [np.ones(len(data))]
        + [data[column].to_numpy(dtype=float) for column in regressors]
    )
    y_direct = data["delta_direct"].to_numpy(dtype=float)
    y_synthetic = data["delta_synthetic"].to_numpy(dtype=float)
    groups = data["date"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    xtx_by_group = []
    xty_direct_by_group = []
    xty_synthetic_by_group = []
    for group in unique_groups:
        mask = groups == group
        x_group = x[mask]
        xtx_by_group.append(x_group.T @ x_group)
        xty_direct_by_group.append(x_group.T @ y_direct[mask])
        xty_synthetic_by_group.append(x_group.T @ y_synthetic[mask])
    xtx_array = np.stack(xtx_by_group)
    xty_direct_array = np.stack(xty_direct_by_group)
    xty_synthetic_array = np.stack(xty_synthetic_by_group)
    rng = np.random.default_rng(SEED + 404)
    draws = rng.integers(
        0, len(unique_groups), size=(replications, len(unique_groups))
    )
    direct_shares = np.empty(replications)
    half_lives = np.full(replications, np.nan)
    direct_adjustments = np.empty(replications)
    synthetic_adjustments = np.empty(replications)
    for replication, indices in enumerate(draws):
        xtx = xtx_array[indices].sum(axis=0)
        direct_beta = np.linalg.pinv(xtx) @ xty_direct_array[indices].sum(axis=0)
        synthetic_beta = (
            np.linalg.pinv(xtx) @ xty_synthetic_array[indices].sum(axis=0)
        )
        direct = float(direct_beta[1])
        synthetic = float(synthetic_beta[1])
        direct_adjustments[replication] = direct
        synthetic_adjustments[replication] = synthetic
        denominator = synthetic - direct
        direct_shares[replication] = (
            synthetic / denominator if denominator != 0 else np.nan
        )
        persistence = 1.0 + direct - synthetic
        if 0 < persistence < 1:
            half_lives[replication] = math.log(0.5) / math.log(persistence)
    return {
        "bootstrap_replications": replications,
        "direct_share_ci_low": float(
            np.nanquantile(direct_shares, 0.025)
        ),
        "direct_share_ci_high": float(
            np.nanquantile(direct_shares, 0.975)
        ),
        "half_life_ci_low": float(np.nanquantile(half_lives, 0.025)),
        "half_life_ci_high": float(np.nanquantile(half_lives, 0.975)),
        "direct_adjustment_bootstrap_ci_low": float(
            np.nanquantile(direct_adjustments, 0.025)
        ),
        "direct_adjustment_bootstrap_ci_high": float(
            np.nanquantile(direct_adjustments, 0.975)
        ),
        "synthetic_adjustment_bootstrap_ci_low": float(
            np.nanquantile(synthetic_adjustments, 0.025)
        ),
        "synthetic_adjustment_bootstrap_ci_high": float(
            np.nanquantile(synthetic_adjustments, 0.975)
        ),
    }


def _cluster_hansen_j(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, int, float]:
    """Two-step cluster-robust Hansen J test for overidentifying restrictions."""

    ztz_inverse = np.linalg.pinv(z.T @ z)
    initial_influence = (
        np.linalg.pinv(x.T @ z @ ztz_inverse @ z.T @ x)
        @ x.T
        @ z
        @ ztz_inverse
    )
    initial_beta = initial_influence @ z.T @ y
    initial_residual = y - x @ initial_beta
    unique_groups = np.unique(groups)
    initial_scores = np.stack(
        [z[groups == group].T @ initial_residual[groups == group]
         for group in unique_groups]
    )
    moment_covariance = initial_scores.T @ initial_scores
    weight = np.linalg.pinv(moment_covariance)
    gmm_bread = np.linalg.pinv(x.T @ z @ weight @ z.T @ x)
    gmm_beta = gmm_bread @ x.T @ z @ weight @ z.T @ y
    gmm_residual = y - x @ gmm_beta
    moment_sum = z.T @ gmm_residual
    statistic = float(max(moment_sum @ weight @ moment_sum, 0.0))
    degrees_freedom = int(z.shape[1] - x.shape[1])
    p_value = (
        float(stats.chi2.sf(statistic, degrees_freedom))
        if degrees_freedom > 0
        else float("nan")
    )
    return statistic, degrees_freedom, p_value


def _iv_analysis_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the single common sample used by every overidentified-IV audit."""

    controls = [
        "nasdaq",
        "horizon_scaled",
        "lag_common_mid",
        "lag_common_mid_sq",
    ]
    columns = [
        "date",
        "lag_gap",
        "lag2_gap",
        "lag3_gap",
        "delta_direct",
        "delta_synthetic",
        *controls,
    ]
    return frame.loc[frame["lag_triplet_valid"], columns].dropna().copy()


def _fieller_ratio_interval(
    numerator: float,
    denominator: float,
    numerator_variance: float,
    denominator_variance: float,
    numerator_denominator_covariance: float,
    critical: float,
) -> tuple[str, float, float]:
    """Fieller confidence set for a ratio of jointly estimated coefficients."""

    critical_squared = critical**2
    quadratic = denominator**2 - critical_squared * denominator_variance
    linear = (
        -2.0 * numerator * denominator
        + 2.0 * critical_squared * numerator_denominator_covariance
    )
    constant = numerator**2 - critical_squared * numerator_variance
    if abs(quadratic) <= EPS:
        if abs(linear) <= EPS:
            return (
                ("all_real", float("-inf"), float("inf"))
                if constant <= 0
                else ("empty", float("nan"), float("nan"))
            )
        boundary = -constant / linear
        return (
            ("upper_bounded", float("-inf"), float(boundary))
            if linear > 0
            else ("lower_bounded", float(boundary), float("inf"))
        )
    discriminant = linear**2 - 4.0 * quadratic * constant
    if discriminant < 0:
        return (
            ("all_real", float("-inf"), float("inf"))
            if quadratic < 0
            else ("empty", float("nan"), float("nan"))
        )
    root_distance = math.sqrt(max(discriminant, 0.0))
    roots = sorted(
        [
            (-linear - root_distance) / (2.0 * quadratic),
            (-linear + root_distance) / (2.0 * quadratic),
        ]
    )
    if quadratic > 0:
        return "bounded", float(roots[0]), float(roots[1])
    return "disjoint", float(roots[0]), float(roots[1])


def cluster_iv_pair(
    frame: pd.DataFrame,
    specification: str,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Overidentified 2SLS for both ECM equations, clustered by date."""

    controls = [
        "nasdaq",
        "horizon_scaled",
        "lag_common_mid",
        "lag_common_mid_sq",
    ]
    data = _iv_analysis_sample(frame)
    x = np.column_stack(
        [np.ones(len(data)), data["lag_gap"].to_numpy(dtype=float)]
        + [data[column].to_numpy(dtype=float) for column in controls]
    )
    z = np.column_stack(
        [
            np.ones(len(data)),
            data["lag2_gap"].to_numpy(dtype=float),
            data["lag3_gap"].to_numpy(dtype=float),
        ]
        + [data[column].to_numpy(dtype=float) for column in controls]
    )
    ztz_inverse = np.linalg.pinv(z.T @ z)
    xzpz = x.T @ z @ ztz_inverse
    bread_left = np.linalg.pinv(xzpz @ z.T @ x)
    influence = bread_left @ xzpz
    groups = data["date"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n, k = x.shape
    group_count = len(unique_groups)
    correction = (
        (group_count / (group_count - 1)) * ((n - 1) / (n - k))
        if group_count > 1
        else 1.0
    )

    betas: dict[str, np.ndarray] = {}
    covariances: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    critical = float(stats.t.ppf(0.975, df=max(group_count - 1, 1)))
    for outcome in ["delta_direct", "delta_synthetic"]:
        y = data[outcome].to_numpy(dtype=float)
        beta = influence @ z.T @ y
        residual = y - x @ beta
        scores = []
        for group in unique_groups:
            mask = groups == group
            scores.append(z[mask].T @ residual[mask])
        score_array = np.stack(scores)
        meat = score_array.T @ score_array
        covariance = correction * influence @ meat @ influence.T
        standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
        coefficient = float(beta[1])
        p_value = (
            float(
                2
                * stats.t.sf(
                    abs(coefficient / standard_error),
                    df=max(group_count - 1, 1),
                )
            )
            if standard_error > 0
            else float("nan")
        )
        fitted = x @ beta
        total = float(np.sum((y - y.mean()) ** 2))
        r_squared = (
            1.0 - float(np.sum(np.square(residual))) / total
            if total > 0
            else float("nan")
        )
        ar_wild_p = wild_cluster_joint_p_value(
            data,
            outcome,
            ["lag2_gap", "lag3_gap", *controls],
            "date",
            ["lag2_gap", "lag3_gap"],
            seed_offset=81 if outcome == "delta_direct" else 82,
        )
        hansen_j, hansen_df, hansen_p = _cluster_hansen_j(
            x, z, y, groups
        )
        row = asdict(
            RegressionResult(
                outcome=outcome,
                specification=specification,
                n=int(len(data)),
                clusters=group_count,
                coefficient=coefficient,
                standard_error=standard_error,
                ci_low=coefficient - critical * standard_error,
                ci_high=coefficient + critical * standard_error,
                p_value=p_value,
                r_squared=r_squared,
                cluster_level="date",
                wild_cluster_p_value=ar_wild_p,
            )
        )
        row.update(
            {
                "hansen_j": hansen_j,
                "hansen_degrees_freedom": hansen_df,
                "hansen_p_value": hansen_p,
            }
        )
        rows.append(row)
        betas[outcome] = beta
        covariances[outcome] = covariance
        residuals[outcome] = residual

    cross_meat = np.zeros((z.shape[1], z.shape[1]))
    for group in unique_groups:
        mask = groups == group
        direct_score = z[mask].T @ residuals["delta_direct"][mask]
        synthetic_score = z[mask].T @ residuals["delta_synthetic"][mask]
        cross_meat += np.outer(direct_score, synthetic_score)
    cross_covariance = correction * influence @ cross_meat @ influence.T

    direct = float(betas["delta_direct"][1])
    synthetic = float(betas["delta_synthetic"][1])
    direct_variance = float(covariances["delta_direct"][1, 1])
    synthetic_variance = float(covariances["delta_synthetic"][1, 1])
    cross_variance = float(cross_covariance[1, 1])
    denominator = synthetic - direct
    direct_share = synthetic / denominator
    persistence = 1.0 + direct - synthetic
    persistence_variance = (
        direct_variance + synthetic_variance - 2.0 * cross_variance
    )
    persistence_se = math.sqrt(max(persistence_variance, 0.0))
    half_life = (
        math.log(0.5) / math.log(persistence)
        if 0 < persistence < 1
        else float("nan")
    )
    half_life_gradient = (
        -math.log(0.5) / (persistence * math.log(persistence) ** 2)
        if 0 < persistence < 1
        else float("nan")
    )
    half_life_se = abs(half_life_gradient) * persistence_se
    derived_critical = critical
    share_numerator_variance = synthetic_variance
    share_denominator_variance = persistence_variance
    share_numerator_denominator_covariance = synthetic_variance - cross_variance
    fieller_type, fieller_low, fieller_high = _fieller_ratio_interval(
        synthetic,
        denominator,
        share_numerator_variance,
        share_denominator_variance,
        share_numerator_denominator_covariance,
        derived_critical,
    )
    derived = {
        "iv_observations": int(len(data)),
        "iv_date_clusters": int(group_count),
        "direct_price_discovery_share": float(direct_share),
        "direct_share_fieller_type": fieller_type,
        "direct_share_fieller_ci_low": fieller_low,
        "direct_share_fieller_ci_high": fieller_high,
        "direct_share_fieller_admissible_ci_low": (
            max(0.0, fieller_low) if fieller_type == "bounded" else float("nan")
        ),
        "direct_share_fieller_admissible_ci_high": (
            min(1.0, fieller_high) if fieller_type == "bounded" else float("nan")
        ),
        "gap_adjustment_coefficient": float(direct - synthetic),
        "gap_persistence": float(persistence),
        "gap_persistence_se": float(persistence_se),
        "gap_half_life_minutes": float(half_life),
        "gap_half_life_se": float(half_life_se),
        "restricted_synthetic_response": 0.0,
        "restricted_gap_persistence": float(1.0 + direct),
        "restricted_gap_half_life_minutes": float(
            math.log(0.5) / math.log(1.0 + direct)
            if 0 < 1.0 + direct < 1
            else float("nan")
        ),
        "direct_hansen_j": float(rows[0]["hansen_j"]),
        "direct_hansen_j_p_value": float(rows[0]["hansen_p_value"]),
        "synthetic_hansen_j": float(rows[1]["hansen_j"]),
        "synthetic_hansen_j_p_value": float(rows[1]["hansen_p_value"]),
        "hansen_degrees_freedom": int(rows[0]["hansen_degrees_freedom"]),
    }

    first_beta, first_covariance, first_r_squared, first_clusters = cluster_ols(
        data,
        "lag_gap",
        ["lag2_gap", "lag3_gap", *controls],
        "date",
    )
    instrument_beta = first_beta[1:3]
    instrument_covariance = first_covariance[1:3, 1:3]
    first_f = float(
        instrument_beta
        @ np.linalg.pinv(instrument_covariance)
        @ instrument_beta
        / len(instrument_beta)
    )
    first_joint_p = float(
        stats.f.sf(first_f, len(instrument_beta), max(first_clusters - 1, 1))
    )
    first_wild_p = wild_cluster_joint_p_value(
        data,
        "lag_gap",
        ["lag2_gap", "lag3_gap", *controls],
        "date",
        ["lag2_gap", "lag3_gap"],
        seed_offset=80,
    )
    first_stage_rows: list[dict[str, object]] = []
    for position, instrument in enumerate(["lag2_gap", "lag3_gap"], start=1):
        first_se = math.sqrt(max(float(first_covariance[position, position]), 0.0))
        first_coefficient = float(first_beta[position])
        first_p = (
            float(
                2
                * stats.t.sf(
                    abs(first_coefficient / first_se),
                    df=max(first_clusters - 1, 1),
                )
            )
            if first_se > 0
            else float("nan")
        )
        first_stage_rows.append(
            {
                "specification": specification,
                "n": int(len(data)),
                "clusters": int(first_clusters),
                "cluster_level": "date",
                "instrument": instrument,
                "coefficient": first_coefficient,
                "standard_error": first_se,
                "p_value": first_p,
                "joint_partial_f": first_f,
                "joint_p_value": first_joint_p,
                "wild_cluster_joint_p_value": first_wild_p,
                "r_squared": float(first_r_squared),
            }
        )
    first_stage = pd.DataFrame(first_stage_rows)
    derived.update(
        {
            "first_stage_lag2_coefficient": float(first_beta[1]),
            "first_stage_lag2_standard_error": float(
                math.sqrt(max(float(first_covariance[1, 1]), 0.0))
            ),
            "first_stage_lag3_coefficient": float(first_beta[2]),
            "first_stage_lag3_standard_error": float(
                math.sqrt(max(float(first_covariance[2, 2]), 0.0))
            ),
            "first_stage_partial_f": first_f,
            "first_stage_joint_p_value": first_joint_p,
            "first_stage_wild_cluster_p_value": first_wild_p,
        }
    )
    return pd.DataFrame(rows), derived, first_stage


def bootstrap_iv_price_discovery(
    frame: pd.DataFrame,
    replications: int = 10_000,
) -> dict[str, float]:
    """Pairs cluster-resampled IV coefficients into nonlinear ECM quantities."""

    controls = [
        "nasdaq",
        "horizon_scaled",
        "lag_common_mid",
        "lag_common_mid_sq",
    ]
    data = _iv_analysis_sample(frame)
    x = np.column_stack(
        [np.ones(len(data)), data["lag_gap"].to_numpy(dtype=float)]
        + [data[column].to_numpy(dtype=float) for column in controls]
    )
    z = np.column_stack(
        [
            np.ones(len(data)),
            data["lag2_gap"].to_numpy(dtype=float),
            data["lag3_gap"].to_numpy(dtype=float),
        ]
        + [data[column].to_numpy(dtype=float) for column in controls]
    )
    y_direct = data["delta_direct"].to_numpy(dtype=float)
    y_synthetic = data["delta_synthetic"].to_numpy(dtype=float)
    groups = data["date"].astype(str).to_numpy()
    unique_groups = np.unique(groups)

    ztz_by_group = []
    xtz_by_group = []
    zty_direct_by_group = []
    zty_synthetic_by_group = []
    for group in unique_groups:
        mask = groups == group
        x_group = x[mask]
        z_group = z[mask]
        ztz_by_group.append(z_group.T @ z_group)
        xtz_by_group.append(x_group.T @ z_group)
        zty_direct_by_group.append(z_group.T @ y_direct[mask])
        zty_synthetic_by_group.append(z_group.T @ y_synthetic[mask])
    ztz_array = np.stack(ztz_by_group)
    xtz_array = np.stack(xtz_by_group)
    zty_direct_array = np.stack(zty_direct_by_group)
    zty_synthetic_array = np.stack(zty_synthetic_by_group)

    rng = np.random.default_rng(SEED + 405)
    draws = rng.integers(
        0, len(unique_groups), size=(replications, len(unique_groups))
    )
    direct_adjustments = np.empty(replications)
    synthetic_adjustments = np.empty(replications)
    persistences = np.empty(replications)
    half_lives = np.full(replications, np.nan)
    restricted_half_lives = np.full(replications, np.nan)
    direct_shares = np.full(replications, np.nan)
    for replication, indices in enumerate(draws):
        ztz = ztz_array[indices].sum(axis=0)
        xtz = xtz_array[indices].sum(axis=0)
        influence = (
            np.linalg.pinv(xtz @ np.linalg.pinv(ztz) @ xtz.T)
            @ xtz
            @ np.linalg.pinv(ztz)
        )
        direct_beta = influence @ zty_direct_array[indices].sum(axis=0)
        synthetic_beta = influence @ zty_synthetic_array[indices].sum(axis=0)
        direct = float(direct_beta[1])
        synthetic = float(synthetic_beta[1])
        direct_adjustments[replication] = direct
        synthetic_adjustments[replication] = synthetic
        persistence = 1.0 + direct - synthetic
        persistences[replication] = persistence
        if 0 < persistence < 1:
            half_lives[replication] = math.log(0.5) / math.log(persistence)
        restricted_persistence = 1.0 + direct
        if 0 < restricted_persistence < 1:
            restricted_half_lives[replication] = (
                math.log(0.5) / math.log(restricted_persistence)
            )
        denominator = synthetic - direct
        if abs(denominator) > EPS:
            direct_shares[replication] = synthetic / denominator

    return {
        "bootstrap_replications": int(replications),
        "bootstrap_valid_half_life_share": float(np.isfinite(half_lives).mean()),
        "bootstrap_valid_restricted_half_life_share": float(
            np.isfinite(restricted_half_lives).mean()
        ),
        "gap_persistence_bootstrap_ci_low": float(
            np.quantile(persistences, 0.025)
        ),
        "gap_persistence_bootstrap_ci_high": float(
            np.quantile(persistences, 0.975)
        ),
        "gap_half_life_bootstrap_ci_low": float(
            np.nanquantile(half_lives, 0.025)
        ),
        "gap_half_life_bootstrap_ci_high": float(
            np.nanquantile(half_lives, 0.975)
        ),
        "restricted_half_life_bootstrap_ci_low": float(
            np.nanquantile(restricted_half_lives, 0.025)
        ),
        "restricted_half_life_bootstrap_ci_high": float(
            np.nanquantile(restricted_half_lives, 0.975)
        ),
        "direct_share_bootstrap_ci_low": float(
            np.nanquantile(direct_shares, 0.025)
        ),
        "direct_share_bootstrap_ci_high": float(
            np.nanquantile(direct_shares, 0.975)
        ),
        "direct_share_bootstrap_admissible_fraction": float(
            np.nanmean((direct_shares >= 0.0) & (direct_shares <= 1.0))
        ),
        "direct_share_bootstrap_below_zero_fraction": float(
            np.nanmean(direct_shares < 0.0)
        ),
        "direct_share_bootstrap_above_one_fraction": float(
            np.nanmean(direct_shares > 1.0)
        ),
        "direct_adjustment_bootstrap_ci_low": float(
            np.quantile(direct_adjustments, 0.025)
        ),
        "direct_adjustment_bootstrap_ci_high": float(
            np.quantile(direct_adjustments, 0.975)
        ),
        "synthetic_adjustment_bootstrap_ci_low": float(
            np.quantile(synthetic_adjustments, 0.025)
        ),
        "synthetic_adjustment_bootstrap_ci_high": float(
            np.quantile(synthetic_adjustments, 0.975)
        ),
    }


def noise_null_simulation(
    frame: pd.DataFrame,
    observed_first_stage_f: float,
    *,
    replications: int = 999,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate independent quote noise under a constant latent price."""

    columns = [
        "date",
        "nasdaq",
        "horizon_scaled",
        "lag_common_mid",
        "lag_common_mid_sq",
    ]
    data = _iv_analysis_sample(frame)[columns].copy()
    controls = np.column_stack(
        [
            np.ones(len(data)),
            data["nasdaq"].to_numpy(dtype=float),
            data["horizon_scaled"].to_numpy(dtype=float),
            data["lag_common_mid"].to_numpy(dtype=float),
            data["lag_common_mid_sq"].to_numpy(dtype=float),
        ]
    )
    controls_inverse = np.linalg.pinv(controls.T @ controls)

    def residualise(values: np.ndarray) -> np.ndarray:
        return values - controls @ (controls_inverse @ controls.T @ values)

    group_codes, unique_groups = pd.factorize(data["date"].astype(str))
    group_count = len(unique_groups)
    n = len(data)
    correction = (group_count / (group_count - 1)) * (
        (n - 1) / (n - controls.shape[1] - 2)
    )
    rng = np.random.default_rng(SEED + 12_000)
    draws: list[dict[str, float]] = []
    for replication in range(replications):
        errors = rng.normal(0.0, 0.01, size=(n, 8))
        current_direct, current_synthetic = errors[:, 0], errors[:, 1]
        lag_direct, lag_synthetic = errors[:, 2], errors[:, 3]
        lag2_direct, lag2_synthetic = errors[:, 4], errors[:, 5]
        lag3_direct, lag3_synthetic = errors[:, 6], errors[:, 7]
        gap = lag_direct - lag_synthetic
        lag2_gap = lag2_direct - lag2_synthetic
        lag3_gap = lag3_direct - lag3_synthetic
        delta_direct = current_direct - lag_direct
        delta_synthetic = current_synthetic - lag_synthetic
        gap_residual = residualise(gap)
        instruments_residual = residualise(
            np.column_stack([lag2_gap, lag3_gap])
        )
        direct_residual = residualise(delta_direct)
        synthetic_residual = residualise(delta_synthetic)
        denominator = float(gap_residual @ gap_residual)
        direct_ols = float(gap_residual @ direct_residual / denominator)
        synthetic_ols = float(gap_residual @ synthetic_residual / denominator)

        instrument_cross_product = instruments_residual.T @ instruments_residual
        instrument_cross_product_inverse = np.linalg.pinv(
            instrument_cross_product
        )
        first_coefficients = (
            instrument_cross_product_inverse
            @ instruments_residual.T
            @ gap_residual
        )
        first_residual = gap_residual - instruments_residual @ first_coefficients
        scores = np.zeros((group_count, 2))
        np.add.at(
            scores,
            group_codes,
            instruments_residual * first_residual[:, None],
        )
        first_covariance = (
            correction
            * instrument_cross_product_inverse
            @ (scores.T @ scores)
            @ instrument_cross_product_inverse
        )
        first_f = float(
            first_coefficients
            @ np.linalg.pinv(first_covariance)
            @ first_coefficients
            / 2
        )
        draws.append(
            {
                "replication": replication + 1,
                "direct_ols": direct_ols,
                "synthetic_ols": synthetic_ols,
                "first_stage_f": first_f,
            }
        )
    simulations = pd.DataFrame(draws)
    summary = {
        "replications": int(replications),
        "observations_per_replication": int(n),
        "direct_ols_median": float(simulations["direct_ols"].median()),
        "direct_ols_p025": float(simulations["direct_ols"].quantile(0.025)),
        "direct_ols_p975": float(simulations["direct_ols"].quantile(0.975)),
        "synthetic_ols_median": float(simulations["synthetic_ols"].median()),
        "synthetic_ols_p025": float(
            simulations["synthetic_ols"].quantile(0.025)
        ),
        "synthetic_ols_p975": float(
            simulations["synthetic_ols"].quantile(0.975)
        ),
        "first_stage_f_median": float(simulations["first_stage_f"].median()),
        "first_stage_f_p95": float(simulations["first_stage_f"].quantile(0.95)),
        "observed_first_stage_f": float(observed_first_stage_f),
        "null_exceedance_rate": float(
            simulations["first_stage_f"].ge(observed_first_stage_f).mean()
        ),
    }
    return simulations, summary


def error_correction_analysis(
    panel: pd.DataFrame,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, dict[str, float]],
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    minute = prepare_error_correction(panel, 60)
    primary = minute.loc[minute["combined_spread"].le(0.10)].copy()
    specifications: list[tuple[str, pd.DataFrame]] = [
        ("Primary: combined spread ≤ 10¢", primary),
        ("All exact consecutive minutes", minute),
        ("Interior states only", minute.loc[minute["state_kind"].eq("interior")]),
        ("Exclude final 10 minutes", minute.loc[minute["minutes_to_close"].gt(10)]),
        ("S&P 500", minute.loc[minute["index_name"].eq("S&P 500")]),
        ("Nasdaq-100", minute.loc[minute["index_name"].eq("Nasdaq-100")]),
    ]
    five_panel = panel.loc[panel["end_period_ts"].mod(300).eq(0)].copy()
    specifications.append(
        ("Five-minute sampling", prepare_error_correction(five_panel, 300))
    )
    regression_rows: list[dict[str, object]] = []
    derived_rows: list[dict[str, object]] = []
    baseline_derived: dict[str, float] = {}
    for name, frame in specifications:
        estimates, derived = estimate_error_correction(
            frame, name, wild=name.startswith("Primary")
        )
        regression_rows.extend(asdict(row) for row in estimates)
        derived_rows.append({"specification": name, **derived})
        if name.startswith("Primary"):
            baseline_derived = derived

    primary_bootstrap = bootstrap_price_discovery(primary)
    baseline_derived.update(primary_bootstrap)

    iv_results, iv_derived, first_stage = cluster_iv_pair(
        primary,
        "IV: lag-2 and lag-3 gap instruments",
    )
    iv_derived.update(bootstrap_iv_price_discovery(primary))
    common_sample = _iv_analysis_sample(primary)
    common_ols_results, common_ols_derived = estimate_error_correction(
        common_sample,
        "OLS on overidentified-IV sample",
        wild=True,
    )
    regression_rows.extend(asdict(row) for row in common_ols_results)
    derived_rows.append(
        {
            "specification": "OLS on overidentified-IV sample",
            **common_ols_derived,
        }
    )
    iv_derived["common_sample_ols"] = {
        row.outcome: {
            "n": int(row.n),
            "coefficient": float(row.coefficient),
            "standard_error": float(row.standard_error),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
            "wild_cluster_p_value": float(row.wild_cluster_p_value),
        }
        for row in common_ols_results
    }
    simulations, simulation_summary = noise_null_simulation(
        primary,
        iv_derived["first_stage_partial_f"],
    )

    lagged = primary.loc[primary["lag_pair_valid"]].copy()
    lead_lag_rows: list[dict[str, object]] = []
    regressors = [
        "lag_gap",
        "lag_delta_direct",
        "lag_delta_synthetic",
        "nasdaq",
        "horizon_scaled",
        "lag_common_mid",
        "lag_common_mid_sq",
    ]
    for outcome in ["delta_direct", "delta_synthetic"]:
        beta, covariance, r_squared, clusters = cluster_ols(
            lagged, outcome, regressors, "date"
        )
        df = max(clusters - 1, 1)
        critical = float(stats.t.ppf(0.975, df=df))
        for position, regressor in enumerate(regressors, start=1):
            se = math.sqrt(max(float(covariance[position, position]), 0.0))
            coefficient = float(beta[position])
            lead_lag_rows.append(
                {
                    "outcome": outcome,
                    "regressor": regressor,
                    "n": len(lagged),
                    "clusters": clusters,
                    "cluster_level": "date",
                    "coefficient": coefficient,
                    "standard_error": se,
                    "ci_low": coefficient - critical * se,
                    "ci_high": coefficient + critical * se,
                    "p_value": float(
                        2 * stats.t.sf(abs(coefficient / se), df=df)
                    )
                    if se > 0
                    else np.nan,
                    "r_squared": r_squared,
                }
            )
    diagnostics: dict[str, object] = {
        **baseline_derived,
        "lead_lag_rows": int(len(lagged)),
        "iv": iv_derived,
        "noise_null": simulation_summary,
    }
    return (
        pd.DataFrame(regression_rows),
        pd.DataFrame(derived_rows),
        diagnostics,
    ), pd.DataFrame(lead_lag_rows), diagnostics, iv_results, first_stage, simulations


def _fixed_horizon_sample(
    panel: pd.DataFrame,
    full_events: set[str],
    horizon: int,
    tolerance: int = 15,
) -> pd.DataFrame:
    frame = panel.loc[
        panel["event_ticker"].isin(full_events)
        & panel["minutes_to_close"].between(horizon, horizon + tolerance)
    ].copy()
    frame["distance_to_horizon"] = frame["minutes_to_close"] - horizon
    frame = (
        frame.sort_values("distance_to_horizon")
        .groupby(["event_ticker", "state_index"], as_index=False)
        .first()
    )
    direct = frame["direct_mid"].clip(0, 1)
    synthetic = frame["synthetic_mid"].clip(0, 1)
    wd = 1.0 / np.square(frame["direct_spread"].clip(0.01, 0.25))
    ws = 1.0 / np.square(frame["synthetic_spread"].clip(0.01, 0.25))
    fused = ((wd * direct + ws * synthetic) / (wd + ws)).clip(0, 1)
    frame["forecast_direct"] = direct
    frame["forecast_synthetic"] = synthetic
    frame["forecast_fused"] = fused
    frame["synthetic_was_clipped"] = frame["synthetic_mid"].ne(synthetic)
    frame["horizon"] = horizon
    return frame


def _metric_values(
    probability: pd.Series, outcome: pd.Series
) -> dict[str, pd.Series]:
    clipped = probability.clip(0.005, 0.995)
    return {
        "brier": (probability - outcome) ** 2,
        "log_loss": -(
            outcome * np.log(clipped) + (1 - outcome) * np.log(1 - clipped)
        ),
        "absolute_error": (probability - outcome).abs(),
    }


def _event_bootstrap_difference(
    event_values: pd.DataFrame,
    method: str,
    benchmark: str,
    replications: int = 10_000,
) -> tuple[float, float, float, float]:
    wide = event_values.pivot(
        index="event_ticker", columns="method", values="value"
    ).dropna(subset=[method, benchmark])
    differences = (wide[method] - wide[benchmark]).to_numpy()
    estimate = float(differences.mean())
    rng = np.random.default_rng(SEED + len(method) + len(benchmark))
    indices = rng.integers(0, len(differences), size=(replications, len(differences)))
    boot = differences[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    p_value = float(
        2
        * min(
            (np.count_nonzero(boot >= 0) + 1) / (replications + 1),
            (np.count_nonzero(boot <= 0) + 1) / (replications + 1),
        )
    )
    return estimate, float(low), float(high), min(p_value, 1.0)


def forecast_analysis(
    panel: pd.DataFrame, contracts: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    state_counts = contracts.groupby("event_ticker").size()
    full_events = set(state_counts.loc[state_counts.eq(30)].index)
    candidate_samples = pd.concat(
        [
            _fixed_horizon_sample(panel, full_events, horizon)
            for horizon in [120, 60, 30, 10]
        ],
        ignore_index=True,
    )
    pair_coverage = candidate_samples.groupby(
        ["event_ticker", "state_index"]
    )["horizon"].nunique()
    common_pairs = pair_coverage.loc[pair_coverage.eq(4)].reset_index()[
        ["event_ticker", "state_index"]
    ]
    samples = candidate_samples.merge(
        common_pairs, on=["event_ticker", "state_index"], how="inner"
    )
    winner_events = set(
        samples.loc[samples["outcome"].eq(1)]
        .groupby("event_ticker")["horizon"]
        .nunique()
        .loc[lambda x: x.eq(4)]
        .index
    )
    samples = samples.loc[samples["event_ticker"].isin(winner_events)].copy()
    long_rows: list[pd.DataFrame] = []
    for method in ["direct", "synthetic", "fused"]:
        values = _metric_values(
            samples[f"forecast_{method}"], samples["outcome"]
        )
        for metric, metric_values in values.items():
            piece = samples[
                ["event_ticker", "index_name", "horizon", "outcome"]
            ].copy()
            piece["method"] = method
            piece["metric"] = metric
            piece["value"] = metric_values
            long_rows.append(piece)
    long = pd.concat(long_rows, ignore_index=True)
    event_values = (
        long.groupby(
            ["event_ticker", "index_name", "horizon", "method", "metric"],
            as_index=False,
        )["value"]
        .mean()
    )
    scores = (
        event_values.groupby(["horizon", "method", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            median=("value", "median"),
            events=("event_ticker", "nunique"),
            standard_error=("value", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
        )
    )
    bootstrap_rows: list[dict[str, object]] = []
    for (horizon, metric), frame in event_values.groupby(["horizon", "metric"]):
        for method in ["synthetic", "fused"]:
            estimate, low, high, p_value = _event_bootstrap_difference(
                frame, method, "direct"
            )
            bootstrap_rows.append(
                {
                    "horizon": horizon,
                    "metric": metric,
                    "method": method,
                    "benchmark": "direct",
                    "difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                }
            )
    by_class = (
        long.groupby(
            ["horizon", "method", "metric", "outcome"], as_index=False
        )
        .agg(mean=("value", "mean"), observations=("value", "size"))
    )
    pairs_per_event = (
        samples[["event_ticker", "state_index"]]
        .drop_duplicates()
        .groupby("event_ticker")
        .size()
    )
    horizon_120 = samples.loc[samples["horizon"].eq(120)]
    diagnostics = {
        "full_partition_events": len(full_events),
        "horizon_tolerance_minutes": 15,
        "candidate_forecast_rows_before_balancing": int(len(candidate_samples)),
        "forecast_rows": int(len(samples)),
        "unique_events": int(samples["event_ticker"].nunique()),
        "common_state_pairs": int(
            samples[["event_ticker", "state_index"]].drop_duplicates().shape[0]
        ),
        "positive_outcomes": int(samples["outcome"].sum()),
        "mean_state_pairs_per_event": float(pairs_per_event.mean()),
        "median_state_pairs_per_event": float(pairs_per_event.median()),
        "p25_state_pairs_per_event": float(pairs_per_event.quantile(0.25)),
        "p75_state_pairs_per_event": float(pairs_per_event.quantile(0.75)),
        "p90_state_pairs_per_event": float(pairs_per_event.quantile(0.90)),
        "maximum_state_pairs_per_event": int(pairs_per_event.max()),
        "horizon_120_direct_mid_p25": float(
            horizon_120["forecast_direct"].quantile(0.25)
        ),
        "horizon_120_direct_mid_median": float(
            horizon_120["forecast_direct"].median()
        ),
        "horizon_120_direct_mid_p75": float(
            horizon_120["forecast_direct"].quantile(0.75)
        ),
        "horizon_120_winner_rate": float(horizon_120["outcome"].mean()),
        "synthetic_clip_rate": float(samples["synthetic_was_clipped"].mean()),
        "rows_by_horizon": {
            str(int(k)): int(v)
            for k, v in samples.groupby("horizon").size().items()
        },
        "events_with_realized_state_by_horizon": {
            str(int(h)): int(
                frame.groupby("event_ticker")["outcome"].sum().eq(1).sum()
            )
            for h, frame in samples.groupby("horizon")
        },
    }
    return scores, pd.DataFrame(bootstrap_rows), by_class, diagnostics, samples


def candidate_episode_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse overlapping positive state-minutes into consecutive episodes."""

    screens = {
        "Before fees": ("gross_lock", "gross_lock", "gross_return_on_cash"),
        "Direct member, all taker": (
            "member_positive",
            "member_net_lock",
            "member_return_on_cash",
        ),
        "Direct member, one maker leg": (
            "one_maker_positive",
            "one_maker_net_lock",
            "one_maker_return_on_cash",
        ),
        "Direct member, two maker legs": (
            "two_maker_positive",
            "two_maker_net_lock",
            "two_maker_return_on_cash",
        ),
        "Direct member, all maker": (
            "all_maker_positive",
            "all_maker_net_lock",
            "all_maker_return_on_cash",
        ),
        "Non-direct, all taker": (
            "non_direct_positive",
            "non_direct_net_lock",
            "non_direct_return_on_cash",
        ),
    }
    rows: list[dict[str, object]] = []
    for scope, scoped in [
        ("All candle ends", panel),
        ("Pre-close only", panel.loc[panel["minutes_to_close"].gt(0)]),
    ]:
        for screen, (column, net_column, return_column) in screens.items():
            frame = scoped.sort_values(
                ["event_ticker", "state_index", "timestamp"]
            ).copy()
            if column == "gross_lock":
                frame["positive"] = frame[column].gt(EPS)
            else:
                frame["positive"] = frame[column].astype(bool)
            grouped = frame.groupby(["event_ticker", "state_index"], sort=False)
            previous_timestamp = grouped["timestamp"].shift(1)
            previous_positive = grouped["positive"].shift(fill_value=False)
            starts = frame["positive"] & (
                ~previous_positive
                | (frame["timestamp"] - previous_timestamp)
                .dt.total_seconds()
                .ne(60)
            )
            frame["episode"] = starts.groupby(
                [frame["event_ticker"], frame["state_index"]]
            ).cumsum()
            positive = frame.loc[frame["positive"]].copy()
            episode_keys = ["event_ticker", "state_index", "episode"]
            durations = positive.groupby(episode_keys).agg(
                duration_minutes=("timestamp", "size")
            )
            best = (
                positive.sort_values(net_column, ascending=False)
                .groupby(episode_keys)
                .first()[
                    [
                        net_column,
                        return_column,
                        "minutes_to_close",
                    ]
                ]
            )
            episodes = durations.join(best)
            durations = episodes["duration_minutes"]
            if len(episodes):
                median_profit = 100 * episodes[net_column].median()
                maximum_profit = 100 * episodes[net_column].max()
                aggregate_profit = 100 * episodes[net_column].sum()
                median_return = 100 * episodes[return_column].median()
                median_holding = episodes["minutes_to_close"].median()
            else:
                median_profit = maximum_profit = aggregate_profit = float("nan")
                median_return = median_holding = float("nan")
            rows.append(
                {
                    "scope": scope,
                    "screen": screen,
                    "eligible_state_minutes": int(len(frame)),
                    "positive_state_minutes": int(frame["positive"].sum()),
                    "positive_rate": float(frame["positive"].mean()),
                    "events": int(
                        frame.loc[frame["positive"], "event_ticker"].nunique()
                    ),
                    "event_state_pairs": int(
                        frame.loc[frame["positive"],
                                  ["event_ticker", "state_index"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "episodes": int(len(episodes)),
                    "median_episode_minutes": float(durations.median())
                    if len(durations)
                    else float("nan"),
                    "p90_episode_minutes": float(durations.quantile(0.90))
                    if len(durations)
                    else float("nan"),
                    "maximum_episode_minutes": int(durations.max())
                    if len(durations)
                    else 0,
                    "median_best_profit_cents": float(median_profit),
                    "maximum_best_profit_cents": float(maximum_profit),
                    "aggregate_best_profit_cents": float(aggregate_profit),
                    "median_return_on_cash_percent": float(median_return),
                    "median_holding_minutes_at_best": float(median_holding),
                }
            )
    return pd.DataFrame(rows)


def sample_concentration_analysis(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Describe how the nominal state-minute count is concentrated."""

    counts = panel.groupby(["event_ticker", "state_index"]).size()
    quantiles = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    distribution = pd.DataFrame(
        {
            "quantile": quantiles,
            "minutes_per_event_state": [
                float(counts.quantile(value)) for value in quantiles
            ],
        }
    )
    ordered = counts.sort_values(ascending=False)
    weights = counts / counts.sum()
    summary = {
        "event_state_pairs": int(len(counts)),
        "state_minutes": int(counts.sum()),
        "mean_minutes_per_pair": float(counts.mean()),
        "median_minutes_per_pair": float(counts.median()),
        "top_10_percent_row_share": float(
            ordered.iloc[: math.ceil(0.10 * len(ordered))].sum() / counts.sum()
        ),
        "top_20_percent_row_share": float(
            ordered.iloc[: math.ceil(0.20 * len(ordered))].sum() / counts.sum()
        ),
        "inverse_hhi_effective_pairs": float(1.0 / weights.pow(2).sum()),
    }
    return distribution, summary


def price_conditioning_analysis(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condition gap and candidate rates on the claim's price level."""

    frame = panel.copy()
    frame["non_direct_positive_combined_spread"] = frame["combined_spread"].where(
        frame["non_direct_positive"]
    )
    ranked = frame["reference_mid"].rank(method="first")
    frame["price_decile"] = pd.qcut(
        ranked,
        q=10,
        labels=[f"D{index}" for index in range(1, 11)],
    )

    def aggregate(grouped) -> pd.DataFrame:
        return grouped.agg(
            observations=("reference_mid", "size"),
            events=("event_ticker", "nunique"),
            price_min=("reference_mid", "min"),
            price_median=("reference_mid", "median"),
            price_max=("reference_mid", "max"),
            median_abs_gap=("abs_mid_gap", "median"),
            median_relative_abs_gap=("relative_abs_gap", "median"),
            median_combined_spread=("combined_spread", "median"),
            non_direct_positive_median_combined_spread=(
                "non_direct_positive_combined_spread",
                "median",
            ),
            gross_candidates=("gross_lock", lambda x: int((x > EPS).sum())),
            gross_candidate_rate=("gross_lock", lambda x: float((x > EPS).mean())),
            member_positive=("member_positive", "sum"),
            member_positive_rate=("member_positive", "mean"),
            non_direct_positive=("non_direct_positive", "sum"),
            non_direct_positive_rate=("non_direct_positive", "mean"),
        ).reset_index()

    deciles = aggregate(frame.groupby("price_decile", observed=True))
    bands = aggregate(frame.groupby("price_band", observed=True))
    return deciles, bands


def dynamic_mapping_placebo(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Shift the synthetic representation two states within each event."""

    keys = ["event_ticker", "end_period_ts", "state_index"]
    shifted = panel[
        keys + ["synthetic_mid", "synthetic_spread"]
    ].copy()
    shifted["state_index"] = shifted["state_index"] - 2
    shifted = shifted.rename(
        columns={
            "synthetic_mid": "placebo_synthetic_mid",
            "synthetic_spread": "placebo_synthetic_spread",
        }
    )
    placebo = panel.merge(shifted, on=keys, how="inner")
    placebo["exact_synthetic_mid"] = placebo["synthetic_mid"]
    placebo["exact_abs_gap_overlap"] = (
        placebo["direct_mid"] - placebo["synthetic_mid"]
    ).abs()
    placebo["synthetic_mid"] = placebo["placebo_synthetic_mid"]
    placebo["synthetic_spread"] = placebo["placebo_synthetic_spread"]
    placebo["mid_gap"] = placebo["direct_mid"] - placebo["synthetic_mid"]
    placebo["abs_mid_gap"] = placebo["mid_gap"].abs()
    placebo["combined_spread"] = (
        placebo["direct_spread"] + placebo["synthetic_spread"]
    )
    prepared = prepare_error_correction(placebo, 60)
    primary = prepared.loc[prepared["combined_spread"].le(0.10)].copy()
    estimates, derived = estimate_error_correction(
        primary,
        "Placebo: synthetic state shifted by two bins",
        wild=True,
    )
    summary = {
        "overlap_state_minutes": int(len(placebo)),
        "dynamic_observations": int(estimates[0].n),
        "date_clusters": int(estimates[0].clusters),
        "exact_overlap_correlation": float(
            placebo[["direct_mid", "exact_synthetic_mid"]]
            .corr()
            .iloc[0, 1]
        ),
        "placebo_correlation": float(
            placebo[["direct_mid", "synthetic_mid"]].corr().iloc[0, 1]
        ),
        "exact_overlap_mae": float(placebo["exact_abs_gap_overlap"].mean()),
        "placebo_mae": float(placebo["abs_mid_gap"].mean()),
        "placebo_gap_persistence": float(derived["gap_persistence"]),
    }
    return pd.DataFrame(asdict(row) for row in estimates), summary


def selection_analysis(
    panel: pd.DataFrame,
    availability: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare synchronous rows with direct-quoted rows missing a threshold leg."""

    frame = availability.sort_values(
        ["event_ticker", "state_index", "end_period_ts"]
    ).copy()
    grouped = frame.groupby(["event_ticker", "state_index"], sort=False)
    previous_ts = grouped["end_period_ts"].shift(1)
    next_ts = grouped["end_period_ts"].shift(-1)
    previous_sync = grouped["synchronous"].shift(1)
    next_sync = grouped["synchronous"].shift(-1)
    previous_missing_minute = (
        (frame["end_period_ts"] - previous_ts).gt(60)
        | (previous_ts.isna() & frame["minutes_to_close"].lt(180))
    )
    next_missing_minute = (
        (next_ts - frame["end_period_ts"]).gt(60)
        | (next_ts.isna() & frame["minutes_to_close"].gt(0))
    )
    frame["adjacent_to_missing"] = frame["synchronous"] & (
        ((frame["end_period_ts"] - previous_ts).eq(60) & previous_sync.eq(False))
        | ((next_ts - frame["end_period_ts"]).eq(60) & next_sync.eq(False))
        | previous_missing_minute
        | next_missing_minute
    )
    metrics = panel[
        [
            "event_ticker",
            "state_index",
            "end_period_ts",
            "abs_mid_gap",
            "gross_lock",
        ]
    ]
    frame = frame.merge(
        metrics,
        on=["event_ticker", "state_index", "end_period_ts"],
        how="left",
    )
    groups = [
        ("Threshold leg missing", ~frame["synchronous"]),
        ("All synchronous rows", frame["synchronous"]),
        ("Synchronous, adjacent to missing", frame["adjacent_to_missing"]),
        (
            "Synchronous, not adjacent to missing",
            frame["synchronous"] & ~frame["adjacent_to_missing"],
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, mask in groups:
        subset = frame.loc[mask]
        rows.append(
            {
                "sample": label,
                "direct_quoted_minutes": int(len(subset)),
                "event_state_pairs": int(
                    subset[["event_ticker", "state_index"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "median_direct_mid": float(subset["direct_mid"].median()),
                "median_direct_spread": float(
                    subset["direct_spread"].median()
                ),
                "mean_candle_volume": float(
                    subset["direct_candle_volume"].mean()
                ),
                "median_abs_gap": float(subset["abs_mid_gap"].median())
                if subset["abs_mid_gap"].notna().any()
                else float("nan"),
                "mean_abs_gap": float(subset["abs_mid_gap"].mean())
                if subset["abs_mid_gap"].notna().any()
                else float("nan"),
                "gross_candidate_rate": float(
                    subset["gross_lock"].gt(EPS).mean()
                )
                if subset["gross_lock"].notna().any()
                else float("nan"),
            }
        )
    summary = {
        "direct_quoted_minutes": int(len(frame)),
        "synchronous_minutes": int(frame["synchronous"].sum()),
        "missing_threshold_minutes": int((~frame["synchronous"]).sum()),
        "synchronous_selection_rate": float(frame["synchronous"].mean()),
        "adjacent_synchronous_minutes": int(frame["adjacent_to_missing"].sum()),
    }
    return pd.DataFrame(rows), summary


def fee_schedule_break_analysis(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Interrupted-time-series diagnostic around the 7 July fee schedule."""

    event = panel.groupby(
        ["event_ticker", "date", "index_name"], as_index=False
    ).agg(
        median_abs_gap=("abs_mid_gap", "median"),
        mean_abs_gap=("abs_mid_gap", "mean"),
        gross_candidate_rate=("gross_lock", lambda x: float((x > EPS).mean())),
        member_positive_rate=("member_positive", "mean"),
        mean_combined_spread=("combined_spread", "mean"),
        median_reference_mid=("reference_mid", "median"),
        quoted_rows=("mid_gap", "size"),
    )
    event["date_timestamp"] = pd.to_datetime(event["date"], utc=True)
    event["post"] = event["date_timestamp"].ge(FEE_SCHEDULE_DATE).astype(float)
    event["running_day"] = (
        event["date_timestamp"] - FEE_SCHEDULE_DATE
    ).dt.days.astype(float)
    event["post_trend"] = event["post"] * event["running_day"]
    event["nasdaq"] = event["index_name"].eq("Nasdaq-100").astype(float)
    event["log_quoted_rows"] = np.log1p(event["quoted_rows"])
    comparison = (
        event.assign(period=np.where(event["post"].eq(1), "On/after 7 July", "Before 7 July"))
        .groupby("period", as_index=False)
        .agg(
            dates=("date", "nunique"),
            events=("event_ticker", "nunique"),
            median_abs_gap=("median_abs_gap", "mean"),
            gross_candidate_rate=("gross_candidate_rate", "mean"),
            member_positive_rate=("member_positive_rate", "mean"),
            mean_combined_spread=("mean_combined_spread", "mean"),
        )
    )
    controls = [
        "post",
        "running_day",
        "post_trend",
        "nasdaq",
        "median_reference_mid",
        "log_quoted_rows",
    ]
    rows: list[dict[str, object]] = []
    for offset, outcome in enumerate(
        [
            "median_abs_gap",
            "gross_candidate_rate",
            "member_positive_rate",
            "mean_combined_spread",
        ]
    ):
        beta, covariance, r_squared, clusters = cluster_ols(
            event, outcome, controls, "date"
        )
        standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
        coefficient = float(beta[1])
        critical = float(stats.t.ppf(0.975, df=max(clusters - 1, 1)))
        rows.append(
            {
                "outcome": outcome,
                "events": int(len(event)),
                "date_clusters": int(clusters),
                "post_coefficient": coefficient,
                "standard_error": standard_error,
                "ci_low": coefficient - critical * standard_error,
                "ci_high": coefficient + critical * standard_error,
                "wild_cluster_p_value": wild_cluster_p_value(
                    event,
                    outcome,
                    controls,
                    "date",
                    "post",
                    seed_offset=170 + offset,
                ),
                "r_squared": float(r_squared),
            }
        )
    diagnostics: dict[str, object] = {
        "schedule_effective_date": "2026-07-07",
        "dates": int(event["date"].nunique()),
        "events": int(len(event)),
        "interpretation": (
            "non-causal interrupted-time-series diagnostic; account type and "
            "contemporaneous market conditions are not identified"
        ),
    }
    return comparison, pd.DataFrame(rows), diagnostics


def complete_surface_restrictions(
    panel: pd.DataFrame,
    contracts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Test package and monotonicity restrictions at every complete snapshot."""

    state_counts = contracts.groupby("event_ticker").size()
    full_events = set(state_counts.loc[state_counts.eq(30)].index)
    candidates = panel.loc[panel["event_ticker"].isin(full_events)].copy()
    counts = candidates.groupby(["event_ticker", "end_period_ts"])[
        "state_index"
    ].nunique()
    complete_keys = counts.loc[counts.eq(30)].index
    rows: list[dict[str, object]] = []
    for event_ticker, end_period_ts in complete_keys:
        frame = candidates.loc[
            candidates["event_ticker"].eq(event_ticker)
            & candidates["end_period_ts"].eq(end_period_ts)
        ].sort_values("state_index")
        direct_bid_mass = float(frame["direct_bid"].sum())
        direct_ask_mass = float(frame["direct_ask"].sum())
        threshold_bids = frame.iloc[:-1]["high_bid"].to_numpy(dtype=float)
        threshold_asks = frame.iloc[:-1]["high_ask"].to_numpy(dtype=float)
        monotonic_lock = threshold_bids[1:] - threshold_asks[:-1]
        rows.append(
            {
                "event_ticker": event_ticker,
                "date": str(frame["date"].iloc[0]),
                "index_name": str(frame["index_name"].iloc[0]),
                "timestamp": str(frame["timestamp"].iloc[0]),
                "minutes_to_close": float(frame["minutes_to_close"].iloc[0]),
                "direct_bid_mass": direct_bid_mass,
                "direct_ask_mass": direct_ask_mass,
                "direct_package_lock": float(
                    max(direct_bid_mass - 1.0, 1.0 - direct_ask_mass, 0.0)
                ),
                "threshold_monotonicity_violations": int(
                    (monotonic_lock > EPS).sum()
                ),
                "maximum_threshold_monotonicity_lock": float(
                    max(float(monotonic_lock.max(initial=0.0)), 0.0)
                ),
                "pairwise_gross_candidates": int(frame["gross_lock"].gt(EPS).sum()),
            }
        )
    snapshots = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "complete_snapshots": int(len(snapshots)),
        "events": int(snapshots["event_ticker"].nunique()),
        "dates": int(snapshots["date"].nunique()),
        "direct_package_violation_snapshots": int(
            snapshots["direct_package_lock"].gt(EPS).sum()
        ),
        "threshold_monotonicity_violation_snapshots": int(
            snapshots["threshold_monotonicity_violations"].gt(0).sum()
        ),
        "pairwise_candidate_snapshots": int(
            snapshots["pairwise_gross_candidates"].gt(0).sum()
        ),
        "maximum_direct_bid_mass": float(snapshots["direct_bid_mass"].max()),
        "minimum_direct_ask_mass": float(snapshots["direct_ask_mass"].min()),
    }
    return snapshots, summary


def calibration_analysis(
    panel: pd.DataFrame,
    contracts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reliability curve on complete 30-state surfaces at 180 minutes."""

    state_counts = contracts.groupby("event_ticker").size()
    full_events = set(state_counts.loc[state_counts.eq(30)].index)
    frame = panel.loc[
        panel["event_ticker"].isin(full_events)
        & panel["minutes_to_close"].eq(180)
    ].copy()
    counts = frame.groupby(["event_ticker", "end_period_ts"])["state_index"].nunique()
    complete_keys = counts.loc[counts.eq(30)].index
    key_frame = pd.DataFrame(
        complete_keys.tolist(), columns=["event_ticker", "end_period_ts"]
    )
    frame = frame.merge(key_frame, on=["event_ticker", "end_period_ts"], how="inner")
    bins = [-EPS, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 1.0 + EPS]
    labels = ["0-1%", "1-2.5%", "2.5-5%", "5-10%", "10-20%", "20-40%", "40-100%"]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED + 14_000)
    for method, column in [
        ("Direct range", "direct_mid"),
        ("Threshold-implied", "synthetic_mid"),
    ]:
        method_frame = frame[
            ["event_ticker", "date", "outcome", column]
        ].copy()
        method_frame["forecast"] = method_frame[column].clip(0.0, 1.0)
        method_frame["bin"] = pd.cut(
            method_frame["forecast"], bins=bins, labels=labels, include_lowest=True
        )
        for label, subset in method_frame.groupby("bin", observed=True):
            event_sums = subset.groupby("event_ticker").agg(
                outcomes=("outcome", "sum"), observations=("outcome", "size")
            )
            event_names = event_sums.index.to_numpy()
            bootstrap = np.empty(4_999)
            for replication in range(len(bootstrap)):
                draw = rng.choice(event_names, size=len(event_names), replace=True)
                sampled = event_sums.loc[draw]
                bootstrap[replication] = (
                    sampled["outcomes"].sum() / sampled["observations"].sum()
                )
            rows.append(
                {
                    "method": method,
                    "probability_bin": str(label),
                    "observations": int(len(subset)),
                    "events": int(subset["event_ticker"].nunique()),
                    "mean_forecast": float(subset["forecast"].mean()),
                    "realised_frequency": float(subset["outcome"].mean()),
                    "calibration_gap": float(
                        subset["outcome"].mean() - subset["forecast"].mean()
                    ),
                    "realised_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "realised_ci_high": float(np.quantile(bootstrap, 0.975)),
                }
            )
    calibration = pd.DataFrame(rows)
    diagnostics: dict[str, object] = {
        "horizon_minutes": 180,
        "complete_surface_events": int(frame["event_ticker"].nunique()),
        "state_rows": int(len(frame)),
        "direct_expected_calibration_error": float(
            np.average(
                calibration.loc[calibration["method"].eq("Direct range"), "calibration_gap"].abs(),
                weights=calibration.loc[calibration["method"].eq("Direct range"), "observations"],
            )
        ),
        "synthetic_expected_calibration_error": float(
            np.average(
                calibration.loc[calibration["method"].eq("Threshold-implied"), "calibration_gap"].abs(),
                weights=calibration.loc[calibration["method"].eq("Threshold-implied"), "observations"],
            )
        ),
    }
    return calibration, diagnostics


def _multicategory_scores(
    probability: np.ndarray, outcome: np.ndarray
) -> dict[str, float]:
    winner = int(np.argmax(outcome))
    cumulative_error = np.cumsum(probability)[:-1] - np.cumsum(outcome)[:-1]
    return {
        "brier": float(np.sum(np.square(probability - outcome))),
        "log_loss": float(-math.log(max(float(probability[winner]), 0.005))),
        "ranked_probability_score": float(np.mean(np.square(cumulative_error))),
    }


def _bootstrap_mean_interval(
    values: np.ndarray, seed_offset: int, replications: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + seed_offset)
    draws = rng.integers(0, len(values), size=(replications, len(values)))
    boot = values[draws].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def whole_surface_analysis(
    panel: pd.DataFrame, contracts: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    """Audit and reconcile complete 30-state surfaces at 180 minutes."""

    state_counts = contracts.groupby("event_ticker").size()
    full_events = set(state_counts.loc[state_counts.eq(30)].index)
    candidates = panel.loc[
        panel["event_ticker"].isin(full_events)
        & panel["minutes_to_close"].eq(180)
    ].copy()
    counts = candidates.groupby(["event_ticker", "end_period_ts"])[
        "state_index"
    ].nunique()
    complete_keys = counts.loc[counts.eq(30)].index
    random = np.random.default_rng(SEED + 808)
    event_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for event_ticker, end_period_ts in complete_keys:
        frame = candidates.loc[
            candidates["event_ticker"].eq(event_ticker)
            & candidates["end_period_ts"].eq(end_period_ts)
        ].sort_values("state_index")
        direct = frame["direct_mid"].to_numpy(dtype=float)
        synthetic = frame["synthetic_mid"].to_numpy(dtype=float)
        outcome = frame["outcome"].to_numpy(dtype=float)
        direct_spreads = frame["direct_spread"].to_numpy(dtype=float)
        threshold_midpoints = frame.iloc[:-1]["high_mid"].to_numpy(dtype=float)
        threshold_bids = frame.iloc[:-1]["high_bid"].to_numpy(dtype=float)
        threshold_asks = frame.iloc[:-1]["high_ask"].to_numpy(dtype=float)
        threshold_spreads = threshold_asks - threshold_bids

        direct_coherent = project_simplex(direct)
        synthetic_coherent = project_simplex(synthetic)
        fused_coherent = fused_distribution(
            direct,
            threshold_midpoints,
            direct_spreads,
            threshold_spreads,
        )
        consistency = consistent_price_system(
            frame["direct_bid"].to_numpy(dtype=float),
            frame["direct_ask"].to_numpy(dtype=float),
            threshold_bids,
            threshold_asks,
        )

        for method, probability in [
            ("direct", direct_coherent),
            ("synthetic", synthetic_coherent),
            ("fused", fused_coherent),
        ]:
            for metric, value in _multicategory_scores(
                probability, outcome
            ).items():
                score_rows.append(
                    {
                        "event_ticker": event_ticker,
                        "index_name": str(frame["index_name"].iloc[0]),
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )

        absolute_exact = np.abs(direct - synthetic)
        active = np.maximum(direct, synthetic) >= 0.05
        upward_active = active[:-1]
        downward_active = active[1:]
        adjacent_errors = np.concatenate(
            [
                np.abs(direct[:-1] - synthetic[1:]),
                np.abs(direct[1:] - synthetic[:-1]),
            ]
        )
        adjacent_active_errors = np.concatenate(
            [
                np.abs(direct[:-1] - synthetic[1:])[upward_active],
                np.abs(direct[1:] - synthetic[:-1])[downward_active],
            ]
        )
        permutation_all = np.empty(1_000)
        permutation_active = np.empty(1_000)
        for draw in range(1_000):
            permuted = random.permutation(synthetic)
            permutation_all[draw] = float(np.mean(np.abs(direct - permuted)))
            permutation_active[draw] = float(
                np.mean(np.abs(direct[active] - permuted[active]))
            )

        event_rows.append(
            {
                "event_ticker": event_ticker,
                "index_name": str(frame["index_name"].iloc[0]),
                "direct_midpoint_mass": float(direct.sum()),
                "direct_bid_mass": float(frame["direct_bid"].sum()),
                "direct_ask_mass": float(frame["direct_ask"].sum()),
                "synthetic_negative_states": int((synthetic < -EPS).sum()),
                "synthetic_negative_mass": float(-synthetic[synthetic < 0].sum()),
                "threshold_executable_monotonic": bool(
                    frame["synthetic_ask"].ge(-EPS).all()
                ),
                "direct_partition_executable": bool(
                    frame["direct_bid"].sum() <= 1 + EPS
                    and frame["direct_ask"].sum() >= 1 - EPS
                ),
                "pairwise_gross_candidate": bool(frame["gross_lock"].gt(EPS).any()),
                "joint_feasible": consistency.feasible,
                "minimum_uniform_slack": consistency.minimum_uniform_slack,
                "exact_mae_all": float(absolute_exact.mean()),
                "adjacent_mae_all": float(adjacent_errors.mean()),
                "permuted_mae_all": float(permutation_all.mean()),
                "exact_mae_active": float(absolute_exact[active].mean()),
                "adjacent_mae_active": float(adjacent_active_errors.mean()),
                "permuted_mae_active": float(permutation_active.mean()),
                "active_states": int(active.sum()),
            }
        )

    events = pd.DataFrame(event_rows)
    long_scores = pd.DataFrame(score_rows)
    scores = (
        long_scores.groupby(["method", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            median=("value", "median"),
            events=("event_ticker", "nunique"),
            standard_error=("value", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
        )
    )
    bootstrap_rows: list[dict[str, object]] = []
    for metric, metric_frame in long_scores.groupby("metric"):
        for method in ["synthetic", "fused"]:
            estimate, low, high, p_value = _event_bootstrap_difference(
                metric_frame, method, "direct"
            )
            bootstrap_rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "benchmark": "direct",
                    "difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                }
            )

    placebo_rows: list[dict[str, object]] = []
    mapping_columns = {
        "Exact rule mapping": "exact_mae",
        "Adjacent-state placebo": "adjacent_mae",
        "Within-surface permutation": "permuted_mae",
    }
    for sample in ["all", "active"]:
        exact = events[f"exact_mae_{sample}"].to_numpy(dtype=float)
        for position, (mapping, stem) in enumerate(mapping_columns.items()):
            values = events[f"{stem}_{sample}"].to_numpy(dtype=float)
            low, high = _bootstrap_mean_interval(
                values, 900 + position + (0 if sample == "all" else 20)
            )
            difference = values - exact
            difference_low, difference_high = _bootstrap_mean_interval(
                difference, 950 + position + (0 if sample == "all" else 20)
            )
            placebo_rows.append(
                {
                    "sample": sample,
                    "mapping": mapping,
                    "events": int(len(values)),
                    "mean_absolute_gap": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "difference_vs_exact": float(difference.mean()),
                    "difference_ci_low": difference_low,
                    "difference_ci_high": difference_high,
                }
            )
    placebo = pd.DataFrame(placebo_rows)
    diagnostics = {
        "horizon_minutes": 180,
        "complete_surface_events": int(len(events)),
        "direct_midpoint_mass_median": float(events["direct_midpoint_mass"].median()),
        "direct_midpoint_mass_p25": float(events["direct_midpoint_mass"].quantile(0.25)),
        "direct_midpoint_mass_p75": float(events["direct_midpoint_mass"].quantile(0.75)),
        "direct_partition_executable_surfaces": int(
            events["direct_partition_executable"].sum()
        ),
        "threshold_executable_monotonic_surfaces": int(
            events["threshold_executable_monotonic"].sum()
        ),
        "joint_feasible_surfaces": int(events["joint_feasible"].sum()),
        "joint_feasible_rate": float(events["joint_feasible"].mean()),
        "joint_infeasible_without_pairwise_candidate": int(
            ((~events["joint_feasible"]) & (~events["pairwise_gross_candidate"])).sum()
        ),
        "minimum_slack_median": float(events["minimum_uniform_slack"].median()),
        "minimum_slack_p75": float(events["minimum_uniform_slack"].quantile(0.75)),
        "minimum_slack_max": float(events["minimum_uniform_slack"].max()),
        "surfaces_with_negative_synthetic_midpoint_state": int(
            events["synthetic_negative_states"].gt(0).sum()
        ),
        "active_states_mean": float(events["active_states"].mean()),
        "permutations_per_surface": 1_000,
    }
    return scores, pd.DataFrame(bootstrap_rows), placebo, diagnostics, events


def robustness_table(panel: pd.DataFrame) -> pd.DataFrame:
    specifications: list[tuple[str, pd.DataFrame]] = [
        ("Baseline", panel),
        ("Interior states only", panel.loc[panel["state_kind"].eq("interior")]),
        ("Exclude final 10 minutes", panel.loc[panel["minutes_to_close"].gt(10)]),
        (
            "Combined spread ≤ 10¢",
            panel.loc[panel["combined_spread"].le(0.10)],
        ),
        (
            "Five-minute timestamps",
            panel.loc[panel["end_period_ts"].mod(300).eq(0)],
        ),
        ("S&P 500", panel.loc[panel["index_name"].eq("S&P 500")]),
        ("Nasdaq-100", panel.loc[panel["index_name"].eq("Nasdaq-100")]),
    ]
    rows: list[dict[str, object]] = []
    for name, frame in specifications:
        rows.append(
            {
                "specification": name,
                "observations": len(frame),
                "events": frame["event_ticker"].nunique(),
                "median_abs_gap": frame["abs_mid_gap"].median(),
                "p90_abs_gap": frame["abs_mid_gap"].quantile(0.90),
                "gross_candidate_rate": frame["gross_lock"].gt(0).mean(),
                "member_positive_rate": frame["member_positive"].mean(),
                "one_maker_positive_rate": frame["one_maker_positive"].mean(),
                "two_maker_positive_rate": frame["two_maker_positive"].mean(),
                "non_direct_positive_rate": frame["non_direct_positive"].mean(),
            }
        )
    return pd.DataFrame(rows)


def figure_payoff_identity() -> None:
    x = np.linspace(-1, 3, 801)
    low = (x >= 0).astype(float)
    high = (x >= 2).astype(float)
    direct = ((x >= 0) & (x < 2)).astype(float)
    difference = low - high

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    axes[0].step(x, low, where="post", color=BLUE, lw=2.2, label=r"$1\{S_T\geq K_i\}$")
    axes[0].step(
        x,
        high,
        where="post",
        color=GOLD,
        lw=2.2,
        label=r"$1\{S_T\geq K_{i+1}\}$",
    )
    axes[0].set_title("Nested threshold payoffs")
    axes[0].legend(loc="upper left")
    axes[1].step(
        x, direct, where="post", color=NAVY, lw=3, label="Direct range claim"
    )
    axes[1].step(
        x,
        difference,
        where="post",
        color=TEAL,
        lw=1.8,
        ls="--",
        label="Threshold difference",
    )
    axes[1].set_title("Exact terminal-payoff replication")
    axes[1].legend(loc="upper right")
    for axis in axes:
        axis.axvline(0, color=GREY, lw=0.8, ls=":")
        axis.axvline(2, color=GREY, lw=0.8, ls=":")
        axis.set_xticks([0, 2], [r"$K_i$", r"$K_{i+1}$"])
        axis.set_ylim(-0.05, 1.16)
        axis.set_xlabel("Reported index value at settlement")
    axes[0].set_ylabel("Terminal payout ($)")
    fig.suptitle(
        "One outcome is traded through two payoff-equivalent representations",
        x=0.5,
        y=1.03,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_1_payoff_identity.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_1_payoff_identity.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_price_coherence(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    hb = ax.hexbin(
        panel["synthetic_mid"],
        panel["direct_mid"],
        gridsize=62,
        extent=(-0.1, 1.1, 0, 1),
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    ax.plot([-0.1, 1.1], [-0.1, 1.1], color=RED, lw=1.5, ls="--")
    ax.set_xlim(-0.08, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Threshold-implied midpoint")
    ax.set_ylabel("Direct range midpoint")
    ax.set_title("Direct and replicated prices are close—but not identical")
    colorbar = fig.colorbar(hb, ax=ax, pad=0.02)
    colorbar.set_label("Observations per hexagon (log scale)")
    ax.text(
        0.03,
        0.96,
        f"N = {len(panel):,} exact state-minutes",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": LIGHT_GREY, "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_2_price_coherence.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_2_price_coherence.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_gap_by_horizon(panel: pd.DataFrame) -> None:
    order = ["120–180", "60–120", "30–60", "0–30"]
    stats_frame = (
        panel.groupby(["index_name", "time_bucket"], observed=True)[
            "abs_mid_gap"
        ]
        .agg(
            median="median",
            p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    stats_frame["time_bucket"] = stats_frame["time_bucket"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for axis, index_name in zip(axes, ["S&P 500", "Nasdaq-100"]):
        frame = stats_frame.loc[stats_frame["index_name"].eq(index_name)].set_index(
            "time_bucket"
        )
        x = np.arange(len(order))
        axis.plot(
            x,
            100 * frame.loc[order, "median"],
            marker="o",
            lw=2.2,
            color=BLUE,
            label="Median",
        )
        axis.plot(
            x,
            100 * frame.loc[order, "p90"],
            marker="s",
            lw=1.8,
            color=GOLD,
            label="90th percentile",
        )
        axis.set_xticks(x, order)
        axis.set_title(index_name)
        axis.set_xlabel("Minutes to settlement")
        axis.legend()
    axes[0].set_ylabel("Absolute midpoint gap (¢)")
    fig.suptitle(
        "Representation gaps through the final three trading hours",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_3_gap_by_horizon.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_3_gap_by_horizon.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_quoted_locks(panel: pd.DataFrame) -> None:
    order = ["120–180", "60–120", "30–60", "0–30"]
    frame = (
        panel.groupby("time_bucket", observed=True)
        .agg(
            gross=("gross_lock", lambda x: (x > 0).mean()),
            member=("member_positive", "mean"),
            one_maker=("one_maker_positive", "mean"),
            non_direct=("non_direct_positive", "mean"),
        )
        .reindex(order)
    )
    x = np.arange(len(order))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - 1.5 * width, frame["gross"], width, color=BLUE, label="Before fees")
    ax.bar(
        x - 0.5 * width,
        frame["member"],
        width,
        color=TEAL,
        label="Direct member: all taker",
    )
    ax.bar(
        x + 0.5 * width,
        frame["one_maker"],
        width,
        color=GREY,
        label="Direct member: one maker leg",
    )
    ax.bar(
        x + 1.5 * width,
        frame["non_direct"],
        width,
        color=RED,
        label="Non-direct: all taker",
    )
    ax.set_xticks(x, order)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share of synchronous state-minutes")
    ax.set_xlabel("Minutes to settlement")
    ax.set_title("Execution assumptions dominate apparent quoted locks")
    ax.legend(ncol=1)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_4_quoted_locks.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_4_quoted_locks.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_error_correction(
    regressions: pd.DataFrame, iv_results: pd.DataFrame
) -> None:
    ols_full = regressions.loc[
        regressions["specification"].eq("Primary: combined spread ≤ 10¢")
    ].copy()
    ols_full["estimator"] = "OLS: full sample"
    ols_common = regressions.loc[
        regressions["specification"].eq("OLS on overidentified-IV sample")
    ].copy()
    ols_common["estimator"] = "OLS: IV sample"
    iv = iv_results.copy()
    iv["estimator"] = "IV: same sample"
    base = pd.concat([ols_full, ols_common, iv], ignore_index=True)
    base["label"] = base["outcome"].map(
        {
            "delta_direct": "Direct range price",
            "delta_synthetic": "Threshold-implied price",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True)
    for axis, outcome, title in zip(
        axes,
        ["delta_direct", "delta_synthetic"],
        ["Direct range response", "Threshold-implied response"],
    ):
        piece = base.loc[base["outcome"].eq(outcome)].set_index("estimator")
        order = ["OLS: full sample", "OLS: IV sample", "IV: same sample"]
        piece = piece.loc[order]
        y = np.arange(3)
        axis.errorbar(
            piece["coefficient"],
            y,
            xerr=np.vstack(
                [
                    piece["coefficient"] - piece["ci_low"],
                    piece["ci_high"] - piece["coefficient"],
                ]
            ),
            fmt="o",
            color=NAVY,
            ecolor=BLUE,
            capsize=4,
            markersize=7,
        )
        axis.axvline(0, color=GREY, ls="--", lw=1)
        axis.set_yticks(y, order)
        axis.set_title(title)
        axis.set_xlabel("Next-minute response")
        for position, row in enumerate(piece.itertuples()):
            axis.text(
                row.ci_high + 0.012,
                position,
                f"{row.coefficient:.3f}",
                va="center",
                fontsize=9,
            )
    fig.suptitle(
        "Common-sample OLS isolates the estimator change from sample selection",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_5_error_correction.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_5_error_correction.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_price_and_calibration(
    price_deciles: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.15))
    x = np.arange(1, 11)
    axes[0].plot(
        x,
        100 * price_deciles["gross_candidate_rate"],
        color=BLUE,
        marker="o",
        lw=2,
        label="Before fees",
    )
    axes[0].plot(
        x,
        100 * price_deciles["member_positive_rate"],
        color=TEAL,
        marker="s",
        lw=2,
        label="Direct-member taker fees",
    )
    axes[0].plot(
        x,
        100 * price_deciles["non_direct_positive_rate"],
        color=RED,
        marker="^",
        lw=1.8,
        label="Non-direct taker fees",
    )
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Reference-price decile (low to high)")
    axes[0].set_ylabel("Positive quoted-lock rate (%)")
    axes[0].set_title("A. Candidate rates by price level")
    axes[0].legend(fontsize=8)

    for method, color, marker in [
        ("Direct range", BLUE, "o"),
        ("Threshold-implied", TEAL, "s"),
    ]:
        piece = calibration.loc[calibration["method"].eq(method)]
        axes[1].errorbar(
            piece["mean_forecast"],
            piece["realised_frequency"],
            yerr=np.vstack(
                [
                    piece["realised_frequency"] - piece["realised_ci_low"],
                    piece["realised_ci_high"] - piece["realised_frequency"],
                ]
            ),
            color=color,
            marker=marker,
            lw=1.8,
            capsize=3,
            label=method,
        )
    axes[1].plot([0, 0.72], [0, 0.72], color=GREY, ls="--", lw=1.2)
    axes[1].set_xlim(-0.01, 0.72)
    axes[1].set_ylim(-0.01, 0.80)
    axes[1].set_xlabel("Mean quoted midpoint")
    axes[1].set_ylabel("Realised frequency")
    axes[1].set_title("B. Reliability at 180 minutes")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Price discreteness and calibration are visible in the low-probability tail",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_10_price_and_calibration.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_10_price_and_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_forecast_scores(scores: pd.DataFrame) -> None:
    frame = scores.loc[scores["metric"].eq("brier")].copy()
    methods = ["direct", "synthetic", "fused"]
    colors = [BLUE, GOLD, TEAL]
    labels = ["Direct", "Threshold-implied", "Spread-weighted fusion"]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for method, color, label in zip(methods, colors, labels):
        piece = frame.loc[frame["method"].eq(method)].sort_values(
            "horizon", ascending=False
        )
        ax.plot(
            piece["horizon"],
            piece["mean"],
            marker="o",
            lw=2,
            color=color,
            label=label,
        )
    ax.invert_xaxis()
    ax.set_xticks([120, 60, 30, 10])
    ax.set_xlabel("Forecast horizon (minutes before settlement)")
    ax.set_ylabel("Event-balanced Brier score (lower is better)")
    ax.set_title("Paired fixed-horizon forecast comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_6_forecast_scores.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_6_forecast_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_case_study(panel: pd.DataFrame) -> dict[str, object]:
    eligible = (
        panel.groupby(["event_ticker", "state_index"])
        .agg(
            observations=("mid_gap", "size"),
            member_positive_minutes=("member_positive", "sum"),
            max_member_net_lock=("member_net_lock", "max"),
            max_abs_gap=("abs_mid_gap", "max"),
        )
        .reset_index()
    )
    eligible = eligible.loc[
        eligible["observations"].ge(45)
        & eligible["member_positive_minutes"].gt(0)
    ]
    chosen = eligible.sort_values(
        ["member_positive_minutes", "max_member_net_lock"],
        ascending=False,
    ).iloc[0]
    frame = panel.loc[
        panel["event_ticker"].eq(chosen.event_ticker)
        & panel["state_index"].eq(chosen.state_index)
    ].sort_values("minutes_to_close", ascending=False)
    fig, axes = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    x = frame["minutes_to_close"]
    axes[0].fill_between(
        x,
        frame["direct_bid"],
        frame["direct_ask"],
        color=BLUE,
        alpha=0.18,
        label="Direct bid–ask",
    )
    axes[0].plot(x, frame["direct_mid"], color=BLUE, lw=1.8, label="Direct midpoint")
    axes[0].fill_between(
        x,
        frame["synthetic_bid"],
        frame["synthetic_ask"],
        color=GOLD,
        alpha=0.20,
        label="Replicated bid–ask bound",
    )
    axes[0].plot(
        x,
        frame["synthetic_mid"],
        color=GOLD,
        lw=1.8,
        label="Threshold-implied midpoint",
    )
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(ncol=2, fontsize=8)
    low_level = frame["low_level"].iloc[0]
    high_level = frame["high_level"].iloc[0]
    if pd.notna(low_level) and pd.notna(high_level):
        state_label = f"{low_level:,.0f}–{high_level - 0.01:,.2f}"
    elif pd.isna(low_level):
        state_label = f"below {high_level:,.0f}"
    else:
        state_label = f"{low_level:,.0f} or above"
    axes[0].set_title(
        f"Case study: {frame['index_name'].iloc[0]}, {frame['date'].iloc[0]}, "
        f"{state_label}"
    )
    axes[1].axhline(0, color=GREY, lw=1)
    axes[1].plot(x, 100 * frame["mid_gap"], color=RED, lw=1.6)
    axes[1].fill_between(
        x,
        0,
        100 * frame["mid_gap"],
        where=frame["mid_gap"].ge(0),
        color=RED,
        alpha=0.15,
    )
    axes[1].set_ylabel("Direct − implied (¢)")
    axes[1].set_xlabel("Minutes to settlement")
    axes[1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_7_case_study.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_7_case_study.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "event_ticker": str(chosen.event_ticker),
        "state_index": int(chosen.state_index),
        "observations": int(chosen.observations),
        "member_positive_minutes": int(chosen.member_positive_minutes),
        "max_member_net_lock": float(chosen.max_member_net_lock),
        "max_absolute_gap": float(chosen.max_abs_gap),
        "date": str(frame["date"].iloc[0]),
        "index_name": str(frame["index_name"].iloc[0]),
    }


def figure_surface_snapshot(panel: pd.DataFrame) -> dict[str, object]:
    counts = (
        panel.groupby(["event_ticker", "end_period_ts"])
        .size()
        .reset_index(name="states")
    )
    chosen = counts.sort_values(
        ["states", "end_period_ts"], ascending=[False, True]
    ).iloc[0]
    frame = panel.loc[
        panel["event_ticker"].eq(chosen.event_ticker)
        & panel["end_period_ts"].eq(chosen.end_period_ts)
    ].sort_values("state_index")
    threshold_midpoints = frame.iloc[:-1]["high_mid"].to_numpy(dtype=float)
    threshold_spreads = (
        frame.iloc[:-1]["high_ask"] - frame.iloc[:-1]["high_bid"]
    ).to_numpy(dtype=float)
    fused = fused_distribution(
        frame["direct_mid"].to_numpy(dtype=float),
        threshold_midpoints,
        frame["direct_spread"].to_numpy(dtype=float),
        threshold_spreads,
    )
    consistency = consistent_price_system(
        frame["direct_bid"].to_numpy(dtype=float),
        frame["direct_ask"].to_numpy(dtype=float),
        frame.iloc[:-1]["high_bid"].to_numpy(dtype=float),
        frame.iloc[:-1]["high_ask"].to_numpy(dtype=float),
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(
        frame["state_index"],
        frame["direct_mid"],
        marker="o",
        ms=4,
        lw=1.6,
        color=BLUE,
        label="Direct range midpoint",
    )
    ax.plot(
        frame["state_index"],
        frame["synthetic_mid"],
        marker="s",
        ms=3.5,
        lw=1.4,
        color=GOLD,
        label="Threshold-implied midpoint",
    )
    ax.plot(
        frame["state_index"],
        fused,
        marker="D",
        ms=3.2,
        lw=1.5,
        color=NAVY,
        label="Spread-weighted coherent reconciliation",
    )
    winner = frame.loc[frame["outcome"].eq(1), "state_index"]
    if len(winner):
        ax.axvline(
            float(winner.iloc[0]),
            color=TEAL,
            ls="--",
            lw=1.2,
            label="Realized state",
        )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("Ordered range state")
    ax.set_ylabel("Price ($)")
    ax.set_title(
        f"A same-minute probability-surface snapshot: "
        f"{frame['index_name'].iloc[0]}, {frame['date'].iloc[0]}, "
        f"{frame['minutes_to_close'].iloc[0]:.0f} minutes to close; "
        f"minimum joint slack={100 * consistency.minimum_uniform_slack:.2f}¢"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_8_surface_snapshot.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_8_surface_snapshot.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "event_ticker": str(chosen.event_ticker),
        "timestamp": str(frame["timestamp"].iloc[0]),
        "states": int(chosen.states),
        "minutes_to_close": float(frame["minutes_to_close"].iloc[0]),
        "index_name": str(frame["index_name"].iloc[0]),
        "joint_feasible": consistency.feasible,
        "minimum_uniform_slack": consistency.minimum_uniform_slack,
    }


def figure_whole_surface_audit(
    events: pd.DataFrame, placebo: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9))

    axes[0].hist(
        100 * events["direct_midpoint_mass"],
        bins=12,
        color=BLUE,
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].axvline(100, color=RED, ls="--", lw=1.2, label="Unit mass")
    axes[0].axvline(
        100 * events["direct_midpoint_mass"].median(),
        color=NAVY,
        lw=1.3,
        label="Median",
    )
    axes[0].set_title("A. Direct midpoint mass")
    axes[0].set_xlabel("Sum across 30 states (¢)")
    axes[0].set_ylabel("Complete surfaces")
    axes[0].legend(fontsize=8)

    ordered_slack = np.sort(100 * events["minimum_uniform_slack"].to_numpy())
    axes[1].scatter(
        np.arange(1, len(ordered_slack) + 1),
        ordered_slack,
        s=24,
        color=np.where(ordered_slack > EPS, RED, TEAL),
        alpha=0.9,
    )
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_title("B. Joint consistent-price test")
    axes[1].set_xlabel("Surface rank")
    axes[1].set_ylabel("Minimum uniform quote slack (¢)")

    active = placebo.loc[placebo["sample"].eq("active")].copy()
    mapping_order = [
        "Exact rule mapping",
        "Adjacent-state placebo",
        "Within-surface permutation",
    ]
    active["mapping"] = pd.Categorical(
        active["mapping"], categories=mapping_order, ordered=True
    )
    active = active.sort_values("mapping")
    means = 100 * active["mean_absolute_gap"].to_numpy()
    errors = np.vstack(
        [
            means - 100 * active["ci_low"].to_numpy(),
            100 * active["ci_high"].to_numpy() - means,
        ]
    )
    axes[2].bar(
        np.arange(3),
        means,
        yerr=errors,
        capsize=4,
        color=[TEAL, GOLD, RED],
        alpha=0.9,
    )
    axes[2].set_xticks(
        np.arange(3), ["Exact", "Adjacent\nstate", "Permuted"], fontsize=8.5
    )
    axes[2].set_title("C. Active-state placebo")
    axes[2].set_ylabel("Mean absolute gap (¢)")
    axes[2].set_xlabel("State mapping")

    fig.suptitle(
        "Whole-surface coherence: mass, joint feasibility, and a wrong-state falsification",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_9_whole_surface_audit.png", bbox_inches="tight")
    fig.savefig(FIGURES / "figure_9_whole_surface_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def save_outputs() -> dict[str, object]:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    panel, contracts, states, availability = build_panel()
    overview = describe_panel(panel, contracts, states)
    grouped = grouped_summary(panel)
    (
        ecm_bundle,
        lead_lag,
        ecm_diagnostics,
        iv_results,
        first_stage,
        noise_simulations,
    ) = error_correction_analysis(panel)
    regressions, ecm_derived, ecm_diagnostics = ecm_bundle
    (
        forecast_scores,
        forecast_bootstrap,
        forecast_by_class,
        forecast_diagnostics,
        forecast_samples,
    ) = forecast_analysis(panel, contracts)
    robustness = robustness_table(panel)
    episodes = candidate_episode_analysis(panel)
    concentration_distribution, concentration_diagnostics = (
        sample_concentration_analysis(panel)
    )
    price_deciles, price_bands = price_conditioning_analysis(panel)
    placebo_regressions, placebo_diagnostics = dynamic_mapping_placebo(panel)
    selection_results, selection_diagnostics = selection_analysis(
        panel, availability
    )
    fee_break_comparison, fee_break_results, fee_break_diagnostics = (
        fee_schedule_break_analysis(panel)
    )
    restriction_snapshots, restriction_diagnostics = (
        complete_surface_restrictions(panel, contracts)
    )
    calibration, calibration_diagnostics = calibration_analysis(
        panel, contracts
    )
    (
        surface_scores,
        surface_bootstrap,
        surface_placebo,
        surface_diagnostics,
        surface_events,
    ) = whole_surface_analysis(panel, contracts)

    panel_output_columns = [
        "index_name",
        "event_ticker",
        "state_index",
        "state_kind",
        "timestamp",
        "minutes_to_close",
        "outcome",
        "direct_mid",
        "synthetic_mid",
        "mid_gap",
        "abs_mid_gap",
        "reference_mid",
        "relative_abs_gap",
        "price_band",
        "direct_spread",
        "synthetic_spread",
        "gross_lock",
        "lock_direction",
        "member_fee",
        "non_direct_fee",
        "one_maker_fee",
        "two_maker_fee",
        "member_net_lock",
        "non_direct_net_lock",
        "one_maker_net_lock",
        "two_maker_net_lock",
        "member_positive",
        "non_direct_positive",
        "one_maker_positive",
        "two_maker_positive",
        "member_return_on_cash",
        "non_direct_return_on_cash",
    ]
    panel[panel_output_columns].to_csv(
        TABLES / "synchronous_panel_derived.csv.gz", index=False
    )
    grouped.to_csv(TABLES / "descriptive_results.csv", index=False)
    regressions.to_csv(TABLES / "error_correction_results.csv", index=False)
    iv_results.to_csv(TABLES / "error_correction_iv_results.csv", index=False)
    first_stage.to_csv(TABLES / "error_correction_iv_first_stage.csv", index=False)
    noise_simulations.to_csv(TABLES / "noise_null_simulations.csv", index=False)
    ecm_derived.to_csv(TABLES / "error_correction_derived.csv", index=False)
    lead_lag.to_csv(TABLES / "lead_lag_results.csv", index=False)
    forecast_scores.to_csv(TABLES / "forecast_scores.csv", index=False)
    forecast_bootstrap.to_csv(TABLES / "forecast_bootstrap.csv", index=False)
    forecast_by_class.to_csv(TABLES / "forecast_scores_by_class.csv", index=False)
    forecast_samples[
        [
            "event_ticker",
            "index_name",
            "horizon",
            "state_index",
            "outcome",
            "forecast_direct",
            "forecast_synthetic",
            "forecast_fused",
            "distance_to_horizon",
        ]
    ].to_csv(TABLES / "forecast_sample_derived.csv", index=False)
    robustness.to_csv(TABLES / "robustness_results.csv", index=False)
    episodes.to_csv(TABLES / "candidate_episode_results.csv", index=False)
    concentration_distribution.to_csv(
        TABLES / "sample_concentration_distribution.csv", index=False
    )
    price_deciles.to_csv(TABLES / "price_decile_results.csv", index=False)
    price_bands.to_csv(TABLES / "price_band_results.csv", index=False)
    placebo_regressions.to_csv(
        TABLES / "dynamic_mapping_placebo.csv", index=False
    )
    selection_results.to_csv(TABLES / "selection_results.csv", index=False)
    fee_break_comparison.to_csv(
        TABLES / "fee_schedule_break_comparison.csv", index=False
    )
    fee_break_results.to_csv(
        TABLES / "fee_schedule_break_results.csv", index=False
    )
    restriction_snapshots.to_csv(
        TABLES / "complete_surface_restrictions.csv", index=False
    )
    calibration.to_csv(TABLES / "calibration_results.csv", index=False)
    surface_scores.to_csv(TABLES / "surface_scores.csv", index=False)
    surface_bootstrap.to_csv(TABLES / "surface_score_bootstrap.csv", index=False)
    surface_placebo.to_csv(TABLES / "surface_placebo_results.csv", index=False)

    figure_payoff_identity()
    figure_price_coherence(panel)
    figure_gap_by_horizon(panel)
    figure_quoted_locks(panel)
    figure_error_correction(regressions, iv_results)
    figure_forecast_scores(forecast_scores)
    case_study = figure_case_study(panel)
    surface_snapshot = figure_surface_snapshot(panel)
    figure_whole_surface_audit(surface_events, surface_placebo)
    figure_price_and_calibration(price_deciles, calibration)

    pre_close_episodes = episodes.loc[
        episodes["scope"].eq("Pre-close only")
    ].set_index("screen")
    episode_diagnostics = {
        str(screen): {
            "eligible_state_minutes": int(row.eligible_state_minutes),
            "positive_state_minutes": int(row.positive_state_minutes),
            "positive_rate": float(row.positive_rate),
            "events": int(row.events),
            "event_state_pairs": int(row.event_state_pairs),
            "episodes": int(row.episodes),
            "median_episode_minutes": float(row.median_episode_minutes),
            "p90_episode_minutes": float(row.p90_episode_minutes),
            "maximum_episode_minutes": int(row.maximum_episode_minutes),
            "median_best_profit_cents": float(row.median_best_profit_cents),
            "maximum_best_profit_cents": float(row.maximum_best_profit_cents),
            "aggregate_best_profit_cents": float(
                row.aggregate_best_profit_cents
            ),
            "median_return_on_cash_percent": float(
                row.median_return_on_cash_percent
            ),
            "median_holding_minutes_at_best": float(
                row.median_holding_minutes_at_best
            ),
        }
        for screen, row in pre_close_episodes.iterrows()
    }

    manifest = {
        "analysis_version": "3.2.0",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "random_seed": SEED,
        "overview": overview,
        "error_correction": ecm_diagnostics,
        "dynamic_mapping_placebo": placebo_diagnostics,
        "sample_concentration": concentration_diagnostics,
        "selection": selection_diagnostics,
        "fee_schedule_break": fee_break_diagnostics,
        "complete_surface_restrictions": restriction_diagnostics,
        "calibration": calibration_diagnostics,
        "forecast": forecast_diagnostics,
        "whole_surface": surface_diagnostics,
        "candidate_episodes_pre_close": episode_diagnostics,
        "case_study": case_study,
        "surface_snapshot": surface_snapshot,
        "definitions": {
            "synchronous": "same one-minute candle end timestamp for every required leg",
            "quoted_lock_candidate": (
                "positive separation of minute-close top-of-book quote intervals "
                "before depth, latency, and fill risk"
            ),
            "member_fee": (
                "July 2026 general taker formula with centi-cent balance precision "
                "for direct exchange members"
            ),
            "non_direct_fee": (
                "same taker formula with whole-cent one-lot balance precision for "
                "non-direct members"
            ),
            "maker_sensitivity": (
                "one or more legs assigned the published default maker multiplier "
                "of zero; resting orders are not assumed to fill atomically"
            ),
            "candidate_episode": (
                "consecutive positive one-minute candle ends for the same event-state, "
                "separated by any missing or non-positive minute"
            ),
            "consistent_price_system": (
                "a non-negative 30-state vector summing to one whose range "
                "probabilities and threshold survival probabilities all lie inside "
                "their recorded bid-ask intervals"
            ),
            "return_on_cash": (
                "net one-lot lock divided by cash paid for a complementary-claim "
                "implementation whose terminal payout is fixed; excludes capital "
                "netting, queue risk, and opportunity cost"
            ),
        },
    }
    (RESULTS / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    output = save_outputs()
    print(json.dumps(output, indent=2, sort_keys=True))
