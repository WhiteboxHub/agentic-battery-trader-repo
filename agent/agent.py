"""
Shared ReAct loop utility for the battery performance multi-agent system.

This module provides:
  - _react_loop()    : Generic ReAct loop used by Analyst and Market agents.
  - AgentRunResult   : Return type carrying the final text and tool call trace.
  - run_agent()      : Legacy single-agent entry point (backwards compat).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd
from openai import OpenAI

from agent.config import get_llm_config, LLMConfig
from agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from agent.tools import TOOL_FUNCTIONS


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class AgentRunResult:
    """Holds the final output text and a structured record of tool usage.

    Attributes
    ----------
    report        : The final text output (Markdown report or findings JSON string).
    tool_calls    : Ordered list of dicts:
                    {"name": str, "arguments": dict, "result_preview": str}
    tool_names    : Ordered list of tool names called (convenience alias).
    iterations    : Number of LLM turns taken.
    """

    def __init__(self, report: str, tool_calls: list[dict], iterations: int):
        self.report = report
        self.tool_calls = tool_calls
        self.tool_names: list[str] = [c["name"] for c in tool_calls]
        self.iterations = iterations

    def __str__(self) -> str:
        return self.report


# ---------------------------------------------------------------------------
# Shared ReAct loop
# ---------------------------------------------------------------------------

def _make_openai_client(agent_name: str = "") -> tuple[OpenAI, str, LLMConfig]:
    """Build an OpenAI-compatible client from config.yaml + environment variables.

    Parameters
    ----------
    agent_name : Optional agent identifier for per-agent model overrides
                 ("analyst", "market", "writer", "critic").

    Returns
    -------
    (OpenAI client, model string, LLMConfig)
    """
    cfg = get_llm_config(agent_name)
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    return client, cfg.model, cfg


def _react_loop(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tool_schemas: list[dict],
    dispatch_fn: Callable[[str, dict], str],
    max_iterations: int = 10,
    verbose: bool = True,
    label: str = "Agent",
) -> tuple[str, list[dict], int]:
    """Generic ReAct (Reasoning + Acting) loop.

    Drives the LLM through tool calls until it produces a final text response
    or max_iterations is reached.

    Parameters
    ----------
    client          : Configured OpenAI-compatible client.
    model           : Model identifier string.
    messages        : Initial message list (system + user prompt already included).
    tool_schemas    : OpenAI function-calling schemas for available tools.
    dispatch_fn     : Callable(tool_name, arguments) -> JSON string result.
    max_iterations  : Safety cap on LLM turns.
    verbose         : Print tool call/result trace to stdout.
    label           : Agent name for trace output (e.g. "Analyst", "Market").

    Returns
    -------
    (final_text, tool_call_trace, iteration_count)
    """
    iteration = 0
    tool_call_count = 0
    tool_call_trace: list[dict] = []

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.1,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        messages.append(message.model_dump(exclude_unset=False))

        if finish_reason == "stop" or not message.tool_calls:
            if verbose:
                print(f"\n  [{label}] Done — {tool_call_count} tool calls.\n")
            return message.content or "", tool_call_trace, iteration

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            tool_call_count += 1
            if verbose:
                arg_str = json.dumps(arguments) if arguments else "(no args)"
                print(f"  [{label}:{tool_call_count}] → {tool_name}({arg_str})")

            result_json = dispatch_fn(tool_name, arguments)

            if verbose:
                preview = result_json[:200] + "..." if len(result_json) > 200 else result_json
                print(f"             ← {preview}\n")

            tool_call_trace.append({
                "name": tool_name,
                "arguments": arguments,
                "result_preview": result_json[:300],
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_json,
            })

    # Max iterations reached — force a final answer
    if verbose:
        print(f"\n  [{label}] Max iterations ({max_iterations}) reached. Forcing final answer.\n")

    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of tool calls. "
            "Please produce your final output now using the tool outputs gathered so far."
        ),
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content or "", tool_call_trace, max_iterations


# ---------------------------------------------------------------------------
# Backwards-compatible single-agent entry point (legacy)
# ---------------------------------------------------------------------------

# All 6 tool schemas (retained here for the legacy run_agent function)
_ALL_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_summary",
            "description": (
                "Return total revenue for Historical and Perfect scenarios plus the absolute "
                "and percentage gap. Uses cleared (realized) rows only. Call this first."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dispatch_comparison",
            "description": (
                "Compare physical battery usage between scenarios: charge/discharge cycle counts, "
                "total energy throughput (MWh), average/min/max SOC."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_high_price_intervals",
            "description": (
                "Return the top-N intervals by cleared price with side-by-side Historical vs "
                "Perfect revenue, dispatch direction, and a missed_opportunity flag."
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
            "name": "analyze_soc_patterns",
            "description": (
                "Analyse state-of-charge distribution relative to price levels for each scenario."
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
                "discharged, or vice versa."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_forecast_accuracy",
            "description": (
                "Measure how well Historical price forecasts tracked actual cleared prices."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _dispatch_all(name: str, arguments: dict[str, Any], df: pd.DataFrame) -> str:
    """Dispatch any of the 6 tool calls with the DataFrame bound."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        if name == "identify_high_price_intervals":
            result = fn(df, top_n=int(arguments.get("top_n", 20)))
        else:
            result = fn(df)
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def run_agent(
    df: pd.DataFrame,
    battery_id: str,
    date: str,
    max_iterations: int = 15,
    verbose: bool = True,
) -> AgentRunResult:
    """Legacy single-agent entry point.

    Runs all 6 tools in a single ReAct loop and returns an AgentRunResult.
    Kept for backwards compatibility — prefer run_pipeline() for new code.
    """
    client, model, cfg = _make_openai_client()

    user_message = USER_PROMPT_TEMPLATE.format(battery_id=battery_id, date=date)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print("  Battery Decision Support Agent (legacy single-agent mode)")
        print(f"  Asset: {battery_id}  |  Period: {date}  |  Model: {model}")
        print(f"{'=' * 60}\n")

    def dispatch_fn(name: str, args: dict) -> str:
        return _dispatch_all(name, args, df)

    final_text, trace, iters = _react_loop(
        client=client,
        model=model,
        messages=messages,
        tool_schemas=_ALL_TOOL_SCHEMAS,
        dispatch_fn=dispatch_fn,
        max_iterations=max_iterations,
        verbose=verbose,
        label="SingleAgent",
    )

    return AgentRunResult(report=final_text, tool_calls=trace, iterations=iters)
