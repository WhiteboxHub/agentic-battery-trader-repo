"""
Battery Decision Support Agent — CLI entry point.

Runs the 4-agent pipeline:
  [Analyst ∥ Market] → Writer → Critic (revision loop) → final report

Usage:
    python main.py data/BLYTHB1_20260126.csv
    python main.py data/BLYTHB1_20260126.csv --output output/report.md --quiet
    python main.py data/BLYTHB1_20260126.csv --max-revisions 0   # disable revision loop

The agent prints a live trace per agent and writes the final Markdown report to output/.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent.tools import load_and_validate
from agent.orchestrator import run_pipeline_sync
from scripts.visualize import visualize

load_dotenv()


def _infer_metadata(filepath: str) -> tuple[str, str]:
    """Extract battery ID and date from the CSV (first row of data)."""
    import pandas as pd

    df_peek = pd.read_csv(filepath, nrows=5, parse_dates=["START_DATETIME"])
    df_peek.columns = [c.strip() for c in df_peek.columns]

    battery_id = "BLYTHB1"
    filename = Path(filepath).stem.upper()
    for part in filename.split("_"):
        if len(part) >= 5 and part.isalnum() and any(c.isdigit() for c in part) and any(c.isalpha() for c in part):
            battery_id = part
            break

    date_str = "Unknown date"
    if "START_DATETIME" in df_peek.columns:
        ts = pd.to_datetime(df_peek["START_DATETIME"].iloc[0], utc=False)
        date_str = ts.strftime("%Y-%m-%d")

    return battery_id, date_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Battery Decision Support Agent — multi-agent pipeline."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the battery interval data CSV file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the Markdown report (default: output/report_<BATTERY>_<DATE>.md).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-agent trace output. Report is still printed.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=int(os.environ.get("MAX_REVISIONS", 2)),
        help="Maximum Writer revisions the Critic can request (default: 2).",
    )
    parser.add_argument(
        "--approval-threshold",
        type=float,
        default=float(os.environ.get("APPROVAL_THRESHOLD", 8.5)),
        help="Minimum Critic score (0–10) required to approve the report (default: 8.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the CSV and print dataset metadata without calling any LLM. "
            "Useful for checking data schema before spending API credits."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from: {csv_path}")
    try:
        df = load_and_validate(csv_path)
    except ValueError as exc:
        print(f"ERROR: Schema validation failed — {exc}", file=sys.stderr)
        sys.exit(1)

    battery_id, date_str = _infer_metadata(csv_path)
    print(f"Asset: {battery_id}  |  Date: {date_str}  |  Rows: {len(df)}")

    if args.dry_run:
        scenarios = df["SCENARIO_NAME"].unique().tolist()
        schedules = df["SCHEDULE_TYPE"].unique().tolist()
        print(f"Scenarios : {scenarios}")
        print(f"Schedules : {schedules}")
        print(f"Columns   : {list(df.columns)}")
        print("\nDry-run complete — schema is valid. Remove --dry-run to run the full pipeline.")
        sys.exit(0)

    result = run_pipeline_sync(
        df=df,
        battery_id=battery_id,
        date=date_str,
        max_revisions=args.max_revisions,
        approval_threshold=args.approval_threshold,
        verbose=not args.quiet,
    )

    report = result.report
    if not report.strip():
        print("ERROR: Pipeline returned an empty report.", file=sys.stderr)
        sys.exit(1)

    # Write report to file
    output_path = args.output
    if output_path is None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        safe_date = date_str.replace("-", "")
        output_path = str(output_dir / f"report_{battery_id}_{safe_date}.md")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_footer = (
        f"\n\n---\n"
        f"*Generated: {timestamp}*  \n"
        f"*Pipeline: 4-agent (Analyst ∥ Market → Writer → Critic)*  \n"
        f"*Eval score: {result.eval_score:.1f}/10  |  "
        f"Revisions: {result.revision_count}  |  "
        f"Tool calls: {len(result.all_tool_calls)}  |  "
        f"Elapsed: {result.elapsed_seconds:.1f}s*\n"
    )
    full_report = report + metadata_footer

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\nReport saved to: {output_path}")

    # Auto-generate visualization dashboard
    try:
        vis_path = str(Path(output_path).with_suffix(".png")).replace("report_", "vis_")
        visualize(csv_path, vis_path)
        print(f"Visual dashboard saved to: {vis_path}\n")
    except Exception as exc:
        print(f"WARNING: Could not generate visualization — {exc}")
    print("=" * 64)
    print(full_report)


if __name__ == "__main__":
    main()
