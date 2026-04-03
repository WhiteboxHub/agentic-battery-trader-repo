"""
Writer Agent — Markdown report formatting.

Responsibilities:
  - Receives merged findings JSON from the Orchestrator (analyst + market data).
  - Makes a single LLM call with no tools — pure text generation.
  - Produces a structured Markdown report following the output format in
    WRITER_SYSTEM_PROMPT.
  - Can also handle revision requests from the Critic Agent (second call).

Design note: The Writer never calls tools and never sees the raw DataFrame.
It can only write what it was given in the findings JSON, which is the primary
hallucination-prevention mechanism in the multi-agent pipeline.
"""

from __future__ import annotations

import json

from agent.agent import _make_openai_client
from agent.prompts import WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE, WRITER_REVISION_TEMPLATE


def _call_llm(client, model: str, system: str, user: str) -> str:
    """Single LLM call — no tools."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _findings_to_str(findings: dict) -> str:
    """Serialise a findings dict to a compact JSON string for the prompt."""
    return json.dumps(findings, indent=2, default=str)


def write_report(
    analyst_findings: dict,
    market_findings: dict,
    battery_id: str,
    date: str,
    verbose: bool = True,
) -> str:
    """Generate the initial Markdown report from analyst and market findings.

    Parameters
    ----------
    analyst_findings  : Dict returned by analyst_agent.run_analyst().
    market_findings   : Dict returned by market_agent.run_market().
    battery_id        : Battery identifier for report header.
    date              : Date string for report header.
    verbose           : Print progress.

    Returns
    -------
    Markdown report string.
    """
    client, model, _cfg = _make_openai_client("writer")

    if verbose:
        print("  [Writer] Composing report from findings...")

    user_message = WRITER_USER_TEMPLATE.format(
        battery_id=battery_id,
        date=date,
        analyst_findings=_findings_to_str(analyst_findings),
        market_findings=_findings_to_str(market_findings),
    )

    report = _call_llm(client, model, WRITER_SYSTEM_PROMPT, user_message)

    if verbose:
        print(f"  [Writer] Report drafted ({len(report)} chars).\n")

    return report


def revise_report(
    draft: str,
    revision_request: str,
    analyst_findings: dict,
    market_findings: dict,
    battery_id: str,
    date: str,
    verbose: bool = True,
) -> str:
    """Revise a draft report based on Critic Agent feedback.

    Parameters
    ----------
    draft             : The previous draft report to revise.
    revision_request  : Specific issues and instructions from the Critic.
    analyst_findings  : Analyst findings dict (for number verification).
    market_findings   : Market findings dict.
    battery_id        : Battery identifier.
    date              : Date string.
    verbose           : Print progress.

    Returns
    -------
    Revised Markdown report string.
    """
    client, model, _cfg = _make_openai_client("writer")

    if verbose:
        print("  [Writer] Revising report based on Critic feedback...")

    user_message = WRITER_REVISION_TEMPLATE.format(
        draft=draft,
        revision_request=revision_request,
        analyst_findings=_findings_to_str(analyst_findings),
        market_findings=_findings_to_str(market_findings),
    )

    revised = _call_llm(client, model, WRITER_SYSTEM_PROMPT, user_message)

    if verbose:
        print(f"  [Writer] Revision complete ({len(revised)} chars).\n")

    return revised
