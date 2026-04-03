"""
Critic Agent — report evaluation and revision feedback.

Responsibilities:
  - Runs two checks independently before making a decision:
    1. Deterministic grounding check — extracts numbers from the report and
       verifies each against the findings dict with 2% tolerance. No LLM call.
    2. LLM quality score — rates recommendation specificity, driver accuracy,
       and trader usefulness on a 0–10 scale.
  - Combines both checks into a CriticResult.
  - If score < approval_threshold (default 7.5/10): generates a specific
    revision_request string for the Writer Agent.
  - If score >= threshold: APPROVE and return the report.

The revision loop is controlled by the Orchestrator (max 2 revisions by default).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from agent.agent import _make_openai_client
from agent.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE

# ---------------------------------------------------------------------------
# Grounding check helpers (deterministic — no LLM)
# ---------------------------------------------------------------------------

_GROUNDING_TOLERANCE = 0.02   # 2% relative tolerance


def _extract_numbers(text: str) -> set[float]:
    """Extract all numbers from text, stripping commas and currency symbols."""
    cleaned = re.sub(r"[$,]", "", text)
    raw = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    result = set()
    for r in raw:
        try:
            v = float(r)
            if not (math.isnan(v) or math.isinf(v)):
                result.add(v)
        except ValueError:
            pass
    return result


def _numbers_match(reported: float, truth: float, tol: float = _GROUNDING_TOLERANCE) -> bool:
    if truth == 0:
        return abs(reported) < 1e-4
    return abs(reported - truth) / abs(truth) <= tol


def _build_ground_truth(analyst_findings: dict, market_findings: dict) -> dict[str, float]:
    """Extract the key numbers from findings dicts that MUST appear in the report."""
    gt: dict[str, float] = {}

    gap = analyst_findings.get("gap", {})
    if gap:
        gt["historical_revenue"] = float(gap.get("historical_revenue", 0))
        gt["perfect_revenue"] = float(gap.get("perfect_revenue", 0))
        gt["gap_abs"] = float(gap.get("gap_abs", 0))
        gt["gap_pct"] = float(gap.get("gap_pct", 0))

    soc = analyst_findings.get("soc", {})
    hist_soc = soc.get("historical", {})
    perf_soc = soc.get("perfect", {})
    if hist_soc:
        gt["hist_avg_soc_top_quartile"] = float(hist_soc.get("avg_soc_top_price_quartile", 0))
        gt["hist_pct_time_at_min"] = float(hist_soc.get("pct_time_at_min", 0))
    if perf_soc:
        gt["perf_avg_soc_top_quartile"] = float(perf_soc.get("avg_soc_top_price_quartile", 0))

    fcast = market_findings.get("forecast_accuracy", {})
    if fcast:
        gt["forecast_mape"] = float(fcast.get("mape", 0))
        gt["pct_underforecast"] = float(fcast.get("pct_underforecast", 0))

    return {k: v for k, v in gt.items() if v != 0}


def run_grounding_check(
    report_text: str,
    analyst_findings: dict,
    market_findings: dict,
) -> tuple[list[str], list[str]]:
    """Verify key numbers in the report against ground-truth findings.

    Returns
    -------
    (issues, verified) where:
        issues   : list of hallucinated/missing number descriptions
        verified : list of confirmed-correct number descriptions
    """
    ground_truth = _build_ground_truth(analyst_findings, market_findings)
    report_numbers = _extract_numbers(report_text)

    issues = []
    verified = []

    for label, truth in ground_truth.items():
        found = any(_numbers_match(n, truth) for n in report_numbers)
        if found:
            verified.append(f"✓ {label}: {truth}")
        else:
            issues.append(
                f"✗ {label}: expected ~{truth} but no matching number found in report"
            )

    return issues, verified


# ---------------------------------------------------------------------------
# CriticResult
# ---------------------------------------------------------------------------

@dataclass
class CriticResult:
    """Outcome of the Critic Agent's evaluation."""

    decision: str                    # "APPROVE" or "REVISE"
    grounding_issues: list[str]      # Numbers in report not matching findings
    grounding_verified: list[str]    # Numbers confirmed correct
    recommendation_specificity: int  # 0–10
    driver_accuracy: int             # 0–10
    trader_usefulness: int           # 0–10
    average_score: float             # 0–10
    revision_request: str            # Instructions for Writer; empty if APPROVE
    summary: str                     # 1-2 sentence critique summary
    llm_raw: dict = field(default_factory=dict)  # Full LLM response for debugging

    @property
    def approved(self) -> bool:
        return self.decision == "APPROVE"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate_report(
    report_text: str,
    analyst_findings: dict,
    market_findings: dict,
    approval_threshold: float = 7.5,
    verbose: bool = True,
) -> CriticResult:
    """Evaluate a report draft and return a CriticResult.

    Steps:
      1. Run deterministic grounding check (no LLM).
      2. Run LLM quality scoring.
      3. Combine: if grounding issues exist OR avg score < threshold → REVISE.

    Parameters
    ----------
    report_text         : The Markdown report draft to evaluate.
    analyst_findings    : Findings dict from AnalystResult.findings.
    market_findings     : Findings dict from MarketResult.findings.
    approval_threshold  : Score (out of 10) required for APPROVE (default 7.5).
    verbose             : Print evaluation summary.

    Returns
    -------
    CriticResult with decision, scores, and revision_request if needed.
    """
    # Step 1: Deterministic grounding check
    grounding_issues, grounding_verified = run_grounding_check(
        report_text, analyst_findings, market_findings
    )

    # Step 2: LLM quality scoring
    client, model, cfg = _make_openai_client("critic")

    user_message = CRITIC_USER_TEMPLATE.format(
        report=report_text[:4500],
        analyst_findings=json.dumps(analyst_findings, indent=2, default=str)[:2000],
        market_findings=json.dumps(market_findings, indent=2, default=str)[:1500],
    )

    # Append explicit JSON instruction for models that don't support json_mode
    system_prompt = CRITIC_SYSTEM_PROMPT
    if not cfg.json_mode_supported:
        system_prompt += (
            "\n\nIMPORTANT: Your entire response must be a single valid JSON object. "
            "Do not include any text, markdown, or explanation outside the JSON."
        )

    create_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
    }
    if cfg.json_mode_supported:
        create_kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**create_kwargs)
        raw = response.choices[0].message.content or "{}"
        # Strip markdown fences local models sometimes add
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        llm_scores = json.loads(raw)
    except Exception as exc:
        llm_scores = {
            "decision": "APPROVE",
            "recommendation_specificity": 5,
            "driver_accuracy": 5,
            "trader_usefulness": 5,
            "average_score": 5.0,
            "revision_request": "",
            "summary": f"LLM judge call failed: {exc}",
            "grounding_issues": [],
        }

    rec_spec = int(llm_scores.get("recommendation_specificity", 5))
    drv_acc = int(llm_scores.get("driver_accuracy", 5))
    trd_use = int(llm_scores.get("trader_usefulness", 5))
    avg_score = (rec_spec + drv_acc + trd_use) / 3.0
    llm_decision = llm_scores.get("decision", "APPROVE")
    llm_revision = llm_scores.get("revision_request", "")
    summary = llm_scores.get("summary", "")

    # Step 3: Combine grounding + LLM to make final decision
    has_grounding_issues = len(grounding_issues) > 0
    score_too_low = avg_score < approval_threshold
    llm_wants_revision = llm_decision == "REVISE"

    if has_grounding_issues or score_too_low or llm_wants_revision:
        decision = "REVISE"
        # Build a specific revision request
        parts = []
        if grounding_issues:
            parts.append("GROUNDING ERRORS — fix these numbers:\n" + "\n".join(grounding_issues))
        if llm_revision:
            parts.append("QUALITY ISSUES:\n" + llm_revision)
        if score_too_low and not llm_revision:
            parts.append(
                f"SCORES TOO LOW (avg {avg_score:.1f}/10, need {approval_threshold}/10): "
                f"specificity={rec_spec}, driver_accuracy={drv_acc}, usefulness={trd_use}. "
                "Make recommendations more operationally specific with concrete numbers and timing."
            )
        revision_request = "\n\n".join(parts)
    else:
        decision = "APPROVE"
        revision_request = ""

    result = CriticResult(
        decision=decision,
        grounding_issues=grounding_issues,
        grounding_verified=grounding_verified,
        recommendation_specificity=rec_spec,
        driver_accuracy=drv_acc,
        trader_usefulness=trd_use,
        average_score=round(avg_score, 2),
        revision_request=revision_request,
        summary=summary,
        llm_raw=llm_scores,
    )

    if verbose:
        icon = "APPROVED" if result.approved else "REVISION REQUESTED"
        print(f"\n  [Critic] {icon} — score {avg_score:.1f}/10  "
              f"(spec={rec_spec} | driver={drv_acc} | usefulness={trd_use})")
        if grounding_issues:
            print(f"  [Critic] Grounding issues: {len(grounding_issues)}")
            for issue in grounding_issues:
                print(f"    {issue}")
        if grounding_verified:
            print(f"  [Critic] Verified: {len(grounding_verified)} key numbers correct")
        if result.summary:
            print(f"  [Critic] {result.summary}\n")

    return result
