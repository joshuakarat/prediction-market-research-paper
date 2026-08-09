#!/usr/bin/env python3
"""Collect matched event metadata and one-minute top-of-book candles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from o2p.api import KalshiPublicClient  # noqa: E402
from o2p.core import match_contracts, range_state_rows  # noqa: E402


SERIES = [
    {
        "index_name": "S&P 500",
        "range_series": "KXINX",
        "threshold_series": "KXINXU",
    },
    {
        "index_name": "Nasdaq-100",
        "range_series": "KXNASDAQ100",
        "threshold_series": "KXNASDAQ100U",
    },
]

SAMPLE_START = datetime(2026, 4, 13, tzinfo=timezone.utc)
WINDOW_MINUTES = 180


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def cache_name(value: str) -> str:
    return value.replace("/", "_").replace("?", "_").replace("&", "_")


def list_events(
    client: KalshiPublicClient,
    series_ticker: str,
    raw_dir: Path,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor = ""
    page = 0
    while True:
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": "settled",
            "min_close_ts": int(SAMPLE_START.timestamp()),
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        payload = client.get_json(
            "events",
            params,
            cache_path=raw_dir / "event_lists" / f"{series_ticker}_{page:03d}.json.gz",
        )
        events.extend(payload.get("events", []))
        cursor = payload.get("cursor", "")
        if not cursor:
            break
        page += 1
    return [
        event
        for event in events
        if str(event.get("event_ticker", "")).endswith("H1600")
    ]


def fetch_event_pair(
    client: KalshiPublicClient,
    series_config: dict[str, str],
    event: dict[str, Any],
    raw_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    range_ticker = event["event_ticker"]
    threshold_ticker = range_ticker.replace(
        series_config["range_series"], series_config["threshold_series"], 1
    )
    range_payload = client.get_json(
        f"events/{range_ticker}",
        {"with_nested_markets": "true"},
        cache_path=raw_dir / "events" / f"{range_ticker}.json.gz",
    )
    threshold_payload = client.get_json(
        f"events/{threshold_ticker}",
        {"with_nested_markets": "true"},
        cache_path=raw_dir / "events" / f"{threshold_ticker}.json.gz",
    )
    return range_payload, threshold_payload


def nested_value(candle: dict[str, Any], section: str, key: str) -> float | None:
    block = candle.get(section) or {}
    value = block.get(f"{key}_dollars", block.get(key))
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scalar_value(candle: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = candle.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def response_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "markets" in payload:
        market_blocks = payload["markets"]
    else:
        market_blocks = [
            {
                "market_ticker": payload.get("ticker"),
                "candlesticks": payload.get("candlesticks", []),
            }
        ]

    rows: list[dict[str, Any]] = []
    for market in market_blocks:
        ticker = market.get("market_ticker", market.get("ticker"))
        for candle in market.get("candlesticks", []):
            rows.append(
                {
                    "ticker": ticker,
                    "end_period_ts": candle.get("end_period_ts"),
                    "yes_bid": nested_value(candle, "yes_bid", "close"),
                    "yes_ask": nested_value(candle, "yes_ask", "close"),
                    "last_price": nested_value(candle, "price", "close"),
                    "previous_price": nested_value(candle, "price", "previous"),
                    "volume": scalar_value(candle, "volume_fp", "volume"),
                    "open_interest": scalar_value(
                        candle, "open_interest_fp", "open_interest"
                    ),
                }
            )
    return rows


def candle_tasks(
    matches: list[dict[str, Any]],
    range_states: list[dict[str, Any]],
    live_cutoff: datetime,
    raw_dir: Path,
) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in matches:
        by_event.setdefault(row["event_ticker"], []).append(row)
    direct_tickers_by_event: dict[str, set[str]] = {}
    for row in range_states:
        direct_tickers_by_event.setdefault(row["event_ticker"], set()).add(
            row["direct_ticker"]
        )

    tasks: list[dict[str, Any]] = []
    for event_ticker, event_matches in sorted(by_event.items()):
        close_time = parse_time(event_matches[0]["direct_close_time"])
        start = close_time - timedelta(minutes=WINDOW_MINUTES)
        tickers = sorted(
            {
                ticker
                for row in event_matches
                for ticker in (
                    row.get("direct_ticker"),
                    row.get("low_ticker"),
                    row.get("high_ticker"),
                )
                if ticker
            }
            | direct_tickers_by_event.get(event_ticker, set())
        )

        if close_time >= live_cutoff:
            chunk_size = min(100, max(1, math.floor(9_000 / (WINDOW_MINUTES + 1))))
            for chunk_number, offset in enumerate(range(0, len(tickers), chunk_size)):
                chunk = tickers[offset : offset + chunk_size]
                tasks.append(
                    {
                        "kind": "batch",
                        "path": "markets/candlesticks",
                        "params": {
                            "market_tickers": ",".join(chunk),
                            "start_ts": int(start.timestamp()),
                            "end_ts": int(close_time.timestamp()),
                            "period_interval": 1,
                            "include_latest_before_start": "true",
                        },
                        "cache": raw_dir
                        / "candles"
                        / event_ticker
                        / f"live_batch_{chunk_number:02d}.json.gz",
                    }
                )
        else:
            for ticker in tickers:
                tasks.append(
                    {
                        "kind": "historical",
                        "path": f"historical/markets/{ticker}/candlesticks",
                        "params": {
                            "start_ts": int(start.timestamp()),
                            "end_ts": int(close_time.timestamp()),
                            "period_interval": 1,
                        },
                        "cache": raw_dir
                        / "candles"
                        / event_ticker
                        / f"{cache_name(ticker)}.json.gz",
                    }
                )
    return tasks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--requests-per-second", type=float, default=15.0)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    client = KalshiPublicClient(requests_per_second=args.requests_per_second)
    cutoff_payload = client.get_json(
        "historical/cutoff",
        cache_path=raw_dir / "historical_cutoff.json.gz",
    )
    live_cutoff = parse_time(cutoff_payload["market_settled_ts"])

    event_jobs: list[tuple[dict[str, str], dict[str, Any]]] = []
    for config in SERIES:
        for event in list_events(client, config["range_series"], raw_dir):
            event_jobs.append((config, event))

    print(f"Matched event candidates: {len(event_jobs)}", flush=True)
    matches: list[dict[str, Any]] = []
    range_states: list[dict[str, Any]] = []
    validation_warnings: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_event_pair, client, config, event, raw_dir): (
                config,
                event,
            )
            for config, event in event_jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            config, event = futures[future]
            range_payload, threshold_payload = future.result()
            event_matches, warnings = match_contracts(
                range_payload, threshold_payload, config["index_name"]
            )
            matches.extend(event_matches)
            if event_matches:
                range_states.extend(
                    range_state_rows(range_payload, config["index_name"])
                )
            if warnings:
                validation_warnings.append(
                    {
                        "event_ticker": event["event_ticker"],
                        "warnings": warnings,
                    }
                )
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"Event metadata: {completed}/{len(futures)}; "
                    f"matched states={len(matches)}",
                    flush=True,
                )

    matches.sort(
        key=lambda row: (row["index_name"], row["strike_date"], row["state_index"])
    )
    write_csv(processed_dir / "contract_map.csv", matches)
    range_states.sort(
        key=lambda row: (row["index_name"], row["strike_date"], row["state_index"])
    )
    write_csv(processed_dir / "range_states.csv", range_states)

    tasks = candle_tasks(matches, range_states, live_cutoff, raw_dir)
    print(f"Candlestick requests: {len(tasks)}", flush=True)
    quotes_path = processed_dir / "quotes.csv.gz"
    quote_fields = [
        "ticker",
        "end_period_ts",
        "yes_bid",
        "yes_ask",
        "last_price",
        "previous_price",
        "volume",
        "open_interest",
    ]
    row_count = 0
    candle_failures: list[dict[str, Any]] = []
    with gzip.open(quotes_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=quote_fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    client.get_json,
                    task["path"],
                    task["params"],
                    task["cache"],
                    2,
                ): task
                for task in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:  # Network gaps are recorded, never imputed.
                    candle_failures.append(
                        {
                            "path": task["path"],
                            "cache": str(task["cache"].relative_to(ROOT)),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                rows = response_rows(payload)
                writer.writerows(rows)
                row_count += len(rows)
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        f"Candles: {completed}/{len(futures)}; rows={row_count:,}",
                        flush=True,
                    )

    event_summary: dict[str, dict[str, Any]] = {}
    for row in range_states:
        event_summary.setdefault(
            row["event_ticker"],
            {
                "index_name": row["index_name"],
                "event_ticker": row["event_ticker"],
                "threshold_event_ticker": next(
                    match["threshold_event_ticker"]
                    for match in matches
                    if match["event_ticker"] == row["event_ticker"]
                ),
                "strike_date": row["strike_date"],
                "close_time": row["direct_close_time"],
                "states": 0,
                "matched_states": 0,
            },
        )
        event_summary[row["event_ticker"]]["states"] += 1
    for row in matches:
        event_summary[row["event_ticker"]]["matched_states"] += 1
    write_csv(processed_dir / "events.csv", list(event_summary.values()))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_start": SAMPLE_START.isoformat(),
        "sample_end": max(row["strike_date"] for row in matches),
        "live_historical_cutoff": live_cutoff.isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "event_count": len(event_summary),
        "state_count": len(matches),
        "full_range_state_count": len(range_states),
        "quote_row_count": row_count,
        "candle_request_count": len(tasks),
        "candle_failure_count": len(candle_failures),
        "candle_failures": candle_failures,
        "validation_warning_count": len(validation_warnings),
        "validation_warnings": validation_warnings,
    }
    with (processed_dir / "collection_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
