# Project Design Specification: Agentic Battery Trader

## 1. System Overview
The **Agentic Battery Trader** is a multi-agent decision support system designed to analyze battery storage performance data. It identifies the "Revenue Gap" between realized operation (Historical) and an optimal counterfactual (Perfect Foresight) and provides grounded recommendations for a human battery trader.

### Core Philosophy: "Structured Reasoning over Sparse Data"
Instead of sending raw CSV data to an LLM, this system uses a **ReAct loop** where specialist agents call deterministic Python tools. This ensures:
1. **Zero Hallucination**: Every number in the report is anchored in tool output.
2. **Context Efficiency**: The LLM only sees high-level statistics, not 288+ rows of raw noise.
3. **Auditability**: Every conclusion is traced back to a specific tool call.

---

## 2. Agent Architecture
The system employs a 4-agent pipeline coordinated by an **Async Orchestrator**.

| Agent | Role | Tools Used | Output Type |
|-------|------|------------|-------------|
| **Analyst** | Quantitative Specialist | `get_financial_summary`, `get_dispatch_comparison`, `analyze_soc_patterns`, `compare_dispatch_timing` | Structured JSON Findings |
| **Market** | Market Specialist | `identify_high_price_intervals`, `get_price_forecast_accuracy` | Structured JSON Findings |
| **Writer** | Content Synthesizer | None | Professional Markdown Report |
| **Critic** | Quality Assurance | None (Grounding Rules) | Approval / Revision Request |

### The Orchestration Flow
1. **Parallel Execution**: Analyst and Market agents run concurrently to minimize latency.
2. **Synthesis**: The Orchestrator merges JSON findings into a single "Truth Context".
3. **Drafting**: The Writer generates the initial Markdown report.
4. **Validation Loop**: The Critic evaluates the report. If grounding fails or scores are low, the Writer must revise (up to 2 times).

---

## 3. Reliability Mechanisms

### Grounding (Deterministic vs. LLM)
Every report undergoes a **Dual-Grounding Check**:
1. **Regex Extraction**: A script extracts every number from the Markdown report.
2. **Value Matching**: These numbers are compared against the JSON Findings with a 2% tolerance. If any number is "new" (unfindable in the data), it is rejected.
3. **LLM Judge**: A separate LLM call evaluates the *logic* of the recommendations (Score ≥ 8.5/10).

### Schema Validation
The `load_and_validate()` tool ensures that any CSV dropped into the system follows the specified schema precisely before a single LLM call is made, preventing "garbage in, garbage out" scenarios.

---

## 4. Performance Metrics
- **Quantification**: Historical vs. Perfect Foresight revenue gap ($) and (%).
- **Diagnostics**:
    - `pct_time_at_min_soc`: Identifies if the battery was empty during peaks.
    - `conflict_pct`: Identifies how often the system moved in the opposite direction of optimal.
    - `mape_under_forecast`: Measures systematic bias in the price forecast.

---

## 5. Extensibility
The toolset in `agent/tools.py` is modular. New analysis (e.g., carbon impact, degradation costs) can be added as functions and registered in the `TOOL_FUNCTIONS` registry for agents to discover.
