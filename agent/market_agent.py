"""
Market Agent — high-price interval identification and forecast accuracy.

Responsibilities:
  - Calls 2 tools: identify_high_price_intervals, get_price_forecast_accuracy.
  - Returns a structured market_findings dict (not Markdown).
  - The Writer Agent uses this dict for specific interval evidence and
    forecast bias data in the secondary driver section.

This agent runs concurrently with analyst_agent.py via asyncio.gather().
"""

from __future__ import annotations

import json

import pandas as pd

from agent.agent import _make_openai_client, _react_loop
from agent.prompts import MARKET_SYSTEM_PROMPT, MARKET_USER_TEMPLATE
from agent.tools import TOOL_FUNCTIONS

# ---------------------------------------------------------------------------
# Tool schemas for the 2 market tools
# ---------------------------------------------------------------------------

MARKET_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "identify_high_price_intervals",
            "description": (
                "Return the top-N intervals by cleared price with side-by-side Historical vs "
                "Perfect revenue, dispatch direction, and a missed_opportunity flag. Shows "
                "which specific market events drove the largest portion of the gap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top price intervals to return (default 20).",
                        "default": 20,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_forecast_accuracy",
            "description": (
                "Measure how well Historical price forecasts tracked actual cleared prices. "
                "Returns MAPE, mean error (bias direction), RMSE, and Pearson correlation. "
                "Systematic under-bias explains why the battery dispatched early and was "
                "empty when extreme price spikes occurred."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

_MARKET_TOOL_NAMES = {s["function"]["name"] for s in MARKET_TOOL_SCHEMAS}


def _dispatch(name: str, arguments: dict, df: pd.DataFrame) -> str:
    """Dispatch Market tool calls with the DataFrame bound."""
    if name not in _MARKET_TOOL_NAMES:
        return json.dumps({"error": f"Tool '{name}' is not available to the Market Agent."})
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        if name == "identify_high_price_intervals":
            top_n = int(arguments.get("top_n", 20))
            return json.dumps(fn(df, top_n=top_n), default=str)
        return json.dumps(fn(df), default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _parse_findings(raw_text: str) -> dict:
    """Extract the JSON findings dict from the LLM's final output."""
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {"parse_error": "Market output was not valid JSON", "raw": raw_text[:500]}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class MarketResult:
    """Return type from the Market Agent."""

    def __init__(self, findings: dict, tool_calls: list[dict], iterations: int):
        self.findings = findings
        self.tool_calls = tool_calls
        self.tool_names: list[str] = [c["name"] for c in tool_calls]
        self.iterations = iterations


def run_market(
    df: pd.DataFrame,
    battery_id: str,
    date: str,
    verbose: bool = True,
) -> MarketResult:
    """Run the Market Agent and return structured market findings.

    Parameters
    ----------
    df          : Validated DataFrame.
    battery_id  : Battery identifier for the prompt.
    date        : Date string for the prompt.
    verbose     : Print tool call trace.

    Returns
    -------
    MarketResult with .findings dict and .tool_calls trace.
    """
    client, model, _cfg = _make_openai_client("market")

    messages: list[dict] = [
        {"role": "system", "content": MARKET_SYSTEM_PROMPT},
        {"role": "user", "content": MARKET_USER_TEMPLATE.format(battery_id=battery_id, date=date)},
    ]

    dispatch_fn = lambda name, args: _dispatch(name, args, df)

    raw_output, trace, iters = _react_loop(
        client=client,
        model=model,
        messages=messages,
        tool_schemas=MARKET_TOOL_SCHEMAS,
        dispatch_fn=dispatch_fn,
        max_iterations=5,
        verbose=verbose,
        label="Market",
    )

    findings = _parse_findings(raw_output)
    return MarketResult(findings=findings, tool_calls=trace, iterations=iters)
