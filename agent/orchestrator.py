"""
Orchestrator — coordinates the 4-agent battery analysis pipeline.

Pipeline flow:
  1. [Parallel]  Analyst Agent + Market Agent run concurrently via asyncio.
  2. [Sequential] Writer Agent produces a Markdown report from merged findings.
  3. [Loop]      Critic Agent evaluates the report; requests revision if score < threshold.
                 Writer revises. Repeat up to max_revisions times.
  4. Return PipelineResult with final report + all metadata.

Usage:
    import asyncio
    from agent.orchestrator import run_pipeline
    result = asyncio.run(run_pipeline(df, battery_id="BLYTHB1", date="2026-01-26"))
    print(result.report)

Or synchronously from a non-async context:
    from agent.orchestrator import run_pipeline_sync
    result = run_pipeline_sync(df, battery_id="BLYTHB1", date="2026-01-26")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from agent.analyst_agent import run_analyst, AnalystResult
from agent.market_agent import run_market, MarketResult
from agent.writer_agent import write_report, revise_report
from agent.critic_agent import evaluate_report, CriticResult


# ---------------------------------------------------------------------------
# Pipeline return type
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete result of the multi-agent analysis pipeline.

    Attributes
    ----------
    report                : Final approved Markdown report.
    analyst_findings      : Structured findings dict from the Analyst Agent.
    market_findings       : Structured findings dict from the Market Agent.
    analyst_tool_calls    : Ordered tool call trace from the Analyst Agent.
    market_tool_calls     : Ordered tool call trace from the Market Agent.
    revision_count        : Number of Writer revisions triggered by the Critic.
    eval_score            : Final Critic average score (0–10).
    eval_details          : Per-dimension scores and grounding check results.
    critic_decision       : "APPROVE" or "REVISE" (final Critic decision).
    elapsed_seconds       : Wall-clock time for the full pipeline.
    """

    report: str
    analyst_findings: dict
    market_findings: dict
    analyst_tool_calls: list[dict] = field(default_factory=list)
    market_tool_calls: list[dict] = field(default_factory=list)
    revision_count: int = 0
    eval_score: float = 0.0
    eval_details: dict = field(default_factory=dict)
    critic_decision: str = "APPROVE"
    elapsed_seconds: float = 0.0

    @property
    def all_tool_calls(self) -> list[dict]:
        """Combined tool call trace from both specialist agents."""
        return self.analyst_tool_calls + self.market_tool_calls


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    df: pd.DataFrame,
    battery_id: str,
    date: str,
    max_revisions: int = 2,
    approval_threshold: float = 7.5,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full 4-agent pipeline asynchronously.

    Parameters
    ----------
    df                  : Validated DataFrame from load_and_validate().
    battery_id          : Battery identifier string.
    date                : Analysis date string.
    max_revisions       : Maximum Writer revisions the Critic can trigger.
    approval_threshold  : Minimum average score (0–10) for Critic approval.
    verbose             : Print per-agent progress traces.

    Returns
    -------
    PipelineResult with final report and full execution metadata.
    """
    start_time = datetime.now()

    if verbose:
        print(f"\n{'='*64}")
        print(f"  Battery Decision Support Agent  —  Multi-Agent Pipeline")
        print(f"  Asset: {battery_id}  |  Date: {date}")
        print(f"  Agents: Orchestrator → [Analyst ∥ Market] → Writer → Critic")
        print(f"{'='*64}\n")

    # ------------------------------------------------------------------ #
    # Step 1: Run Analyst + Market agents concurrently                    #
    # ------------------------------------------------------------------ #

    if verbose:
        print("  [Orchestrator] Launching Analyst and Market agents in parallel...\n")

    loop = asyncio.get_event_loop()

    analyst_task = loop.run_in_executor(
        None,
        lambda: run_analyst(df=df, battery_id=battery_id, date=date, verbose=verbose),
    )
    market_task = loop.run_in_executor(
        None,
        lambda: run_market(df=df, battery_id=battery_id, date=date, verbose=verbose),
    )

    analyst_result: AnalystResult
    market_result: MarketResult
    analyst_result, market_result = await asyncio.gather(analyst_task, market_task)

    if verbose:
        print(
            f"  [Orchestrator] Analyst done ({len(analyst_result.tool_calls)} tool calls) | "
            f"Market done ({len(market_result.tool_calls)} tool calls).\n"
        )

    analyst_findings = analyst_result.findings
    market_findings = market_result.findings

    # ------------------------------------------------------------------ #
    # Step 2: Writer produces initial draft                               #
    # ------------------------------------------------------------------ #

    if verbose:
        print("  [Orchestrator] Passing merged findings to Writer Agent...\n")

    draft = write_report(
        analyst_findings=analyst_findings,
        market_findings=market_findings,
        battery_id=battery_id,
        date=date,
        verbose=verbose,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Critic revision loop                                        #
    # ------------------------------------------------------------------ #

    revision_count = 0
    critic_result: CriticResult | None = None

    for attempt in range(max_revisions + 1):
        if verbose:
            iter_label = f"attempt {attempt + 1}/{max_revisions + 1}"
            print(f"  [Orchestrator] Sending draft to Critic Agent ({iter_label})...\n")

        critic_result = evaluate_report(
            report_text=draft,
            analyst_findings=analyst_findings,
            market_findings=market_findings,
            approval_threshold=approval_threshold,
            verbose=verbose,
        )

        if critic_result.approved:
            if verbose:
                print(f"  [Orchestrator] Critic APPROVED after {revision_count} revision(s).\n")
            break

        if attempt < max_revisions:
            revision_count += 1
            if verbose:
                print(
                    f"  [Orchestrator] Critic requested revision {revision_count}. "
                    f"Sending to Writer...\n"
                )
            draft = revise_report(
                draft=draft,
                revision_request=critic_result.revision_request,
                analyst_findings=analyst_findings,
                market_findings=market_findings,
                battery_id=battery_id,
                date=date,
                verbose=verbose,
            )
        else:
            if verbose:
                print(
                    f"  [Orchestrator] Max revisions ({max_revisions}) reached. "
                    f"Accepting best draft.\n"
                )
            break

    elapsed = (datetime.now() - start_time).total_seconds()

    if verbose:
        score = critic_result.average_score if critic_result else 0.0
        print(f"{'='*64}")
        print(f"  Pipeline complete in {elapsed:.1f}s")
        print(
            f"  Revisions: {revision_count}  |  "
            f"Eval score: {score:.1f}/10  |  "
            f"Tool calls: {len(analyst_result.tool_calls) + len(market_result.tool_calls)}"
        )
        print(f"{'='*64}\n")

    eval_details: dict = {}
    if critic_result:
        eval_details = {
            "recommendation_specificity": critic_result.recommendation_specificity,
            "driver_accuracy": critic_result.driver_accuracy,
            "trader_usefulness": critic_result.trader_usefulness,
            "grounding_issues": critic_result.grounding_issues,
            "grounding_verified": critic_result.grounding_verified,
            "summary": critic_result.summary,
        }

    return PipelineResult(
        report=draft,
        analyst_findings=analyst_findings,
        market_findings=market_findings,
        analyst_tool_calls=analyst_result.tool_calls,
        market_tool_calls=market_result.tool_calls,
        revision_count=revision_count,
        eval_score=critic_result.average_score if critic_result else 0.0,
        eval_details=eval_details,
        critic_decision=critic_result.decision if critic_result else "APPROVE",
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Synchronous convenience wrapper
# ---------------------------------------------------------------------------

def run_pipeline_sync(
    df: pd.DataFrame,
    battery_id: str,
    date: str,
    max_revisions: int = 2,
    approval_threshold: float = 7.5,
    verbose: bool = True,
) -> PipelineResult:
    """Synchronous wrapper around run_pipeline() for use in non-async contexts.

    Equivalent to asyncio.run(run_pipeline(...)) but handles the case where
    an event loop is already running (e.g. in Jupyter notebooks).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside an existing event loop (e.g. Jupyter)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                run_pipeline(df, battery_id, date, max_revisions, approval_threshold, verbose),
            )
            return future.result()
    else:
        return asyncio.run(
            run_pipeline(df, battery_id, date, max_revisions, approval_threshold, verbose)
        )
