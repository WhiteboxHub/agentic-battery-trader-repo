"""
Analyst Agent — financial, dispatch, SOC, and timing analysis.

Responsibilities:
  - Calls 4 tools: get_financial_summary, get_dispatch_comparison,
    analyze_soc_patterns, compare_dispatch_timing.
  - Returns a structured analyst_findings dict (not Markdown).
  - The Writer Agent uses this dict as its primary source for financial
    figures, SOC data, and direction conflict statistics.

This agent uses the shared _react_loop() from agent.py and is designed
to run concurrently with market_agent.py via asyncio.
"""

from __future__ import annotations

import json

import pandas as pd

from agent.agent import _make_openai_client, _react_loop, AgentRunResult
from agent.prompts import ANALYST_SYSTEM_PROMPT, ANALYST_USER_TEMPLATE
from agent.tools import TOOL_FUNCTIONS

# ---------------------------------------------------------------------------
# Tool schemas for the 4 analyst tools
# ---------------------------------------------------------------------------

ANALYST_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_summary",
            "description": (
                "Return total Historical and Perfect revenue, the absolute gap ($), "
                "and the percentage gap. Uses cleared (realized) rows only. CALL THIS FIRST."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dispatch_comparison",
            "description": (
                "Compare charge/discharge cycle counts, total energy throughput (MWh), "
                "and SOC statistics between Historical and Perfect scenarios."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_soc_patterns",
            "description": (
                "Return SOC distribution relative to price levels: quartiles, % time at "
                "min/max SOC, average SOC during high/low-price intervals, and SOC values "
                "at the top-10 price spike intervals."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_dispatch_timing",
            "description": (
                "Identify direction conflicts: intervals where Historical charged while Perfect "
                "discharged (or vice versa). Returns conflict count, revenue impact, and the "
                "top-10 worst conflicts."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Analyst tool names — used to scope dispatch
_ANALYST_TOOL_NAMES = {s["function"]["name"] for s in ANALYST_TOOL_SCHEMAS}


def _dispatch(name: str, arguments: dict, df: pd.DataFrame) -> str:
    """Dispatch Analyst tool calls with the DataFrame bound."""
    if name not in _ANALYST_TOOL_NAMES:
        return json.dumps({"error": f"Tool '{name}' is not available to the Analyst Agent."})
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(df), default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _parse_findings(raw_text: str) -> dict:
    """Extract the JSON findings dict from the LLM's final output.

    The analyst is instructed to return pure JSON. If the model wraps it in
    markdown fences or adds explanation text, this function strips that away.
    """
    text = raw_text.strip()

    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text wrapped so the pipeline doesn't crash
    return {"parse_error": "Analyst output was not valid JSON", "raw": raw_text[:500]}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class AnalystResult:
    """Return type from the Analyst Agent."""

    def __init__(self, findings: dict, tool_calls: list[dict], iterations: int):
        self.findings = findings
        self.tool_calls = tool_calls
        self.tool_names: list[str] = [c["name"] for c in tool_calls]
        self.iterations = iterations


def run_analyst(
    df: pd.DataFrame,
    battery_id: str,
    date: str,
    verbose: bool = True,
) -> AnalystResult:
    """Run the Analyst Agent and return structured findings.

    Parameters
    ----------
    df          : Validated DataFrame.
    battery_id  : Battery identifier for the prompt.
    date        : Date string for the prompt.
    verbose     : Print tool call trace.

    Returns
    -------
    AnalystResult with .findings dict and .tool_calls trace.
    """
    client, model, _cfg = _make_openai_client("analyst")

    messages: list[dict] = [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {"role": "user", "content": ANALYST_USER_TEMPLATE.format(battery_id=battery_id, date=date)},
    ]

    dispatch_fn = lambda name, args: _dispatch(name, args, df)

    raw_output, trace, iters = _react_loop(
        client=client,
        model=model,
        messages=messages,
        tool_schemas=ANALYST_TOOL_SCHEMAS,
        dispatch_fn=dispatch_fn,
        max_iterations=8,
        verbose=verbose,
        label="Analyst",
    )

    findings = _parse_findings(raw_output)
    return AnalystResult(findings=findings, tool_calls=trace, iterations=iters)
