# Battery Decision Support Agent

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Providers](https://img.shields.io/badge/LLM-OpenRouter%20%7C%20OpenAI%20%7C%20Ollama-orange)

An LLM-powered **multi-agent system** that analyses battery storage performance data and produces actionable trading recommendations.

The system compares **Historical** operation against **Perfect Foresight** to quantify the revenue gap, identify what drove it, and generate grounded recommendations — entirely through structured tool calls. The LLM never reads raw data.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API credentials
cp .env.example .env
# Edit .env — fill in OPENROUTER_API_KEY (or OPENAI_API_KEY) and MODEL

# 3. Run
##python main.py data/BLYTHB1_20260126.csv
python main.py data/Takehome_Problem_Agentic_Battery_Analysis_System_BLYTHB1_20260126.csv
```

The Markdown report is written to `output/report_<BATTERY>_<DATE>.md` and printed to stdout.

### Automated Visualization
The system now automatically generates a **Visual Performance Dashboard** (PNG) for every analysis run, plotting Price vs. State of Charge (SOC) to provide immediate context for the revenue gap.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | `output/report_*.md` | Custom report path |
| `--quiet` | off | Suppress per-agent trace |
| `--max-revisions N` | `2` | Max Writer revisions the Critic can trigger |
| `--approval-threshold X` | `7.5` | Critic score (0–10) needed to approve |

---

## System Architecture

Four specialised LLM agents coordinated by an **Orchestrator**:

```mermaid
flowchart TD
    CLI["<b>main.py</b>\nrun_pipeline_sync()"] --> Orch

    subgraph Orch["<b>Orchestrator</b> — orchestrator.py"]
        P1["1 · launch parallel agents"]
        P2["2 · Writer composes report"]
        P3["3 · Critic revision loop\nup to 2 revisions"]
        P1 --> P2 --> P3
    end

    Orch -->|"asyncio.gather — parallel"| Analyst
    Orch -->|"asyncio.gather — parallel"| Market

    subgraph Analyst["<b>Analyst Agent</b> — analyst_agent.py"]
        A1[get_financial_summary]
        A2[get_dispatch_comparison]
        A3[analyze_soc_patterns]
        A4[compare_dispatch_timing]
        A1 --> A2 --> A3 --> A4
    end

    subgraph Market["<b>Market Agent</b> — market_agent.py"]
        M1[identify_high_price_intervals]
        M2[get_price_forecast_accuracy]
        M1 --> M2
    end

    Analyst -->|"analyst_findings dict"| Orch
    Market -->|"market_findings dict"| Orch
    Orch -->|"merged findings JSON"| Writer

    subgraph Writer["<b>Writer Agent</b> — writer_agent.py"]
        W1["Single LLM call — no tools\nformats Markdown from JSON"]
    end

    Writer -->|"draft report"| Critic

    subgraph Critic["<b>Critic Agent</b> — critic_agent.py"]
        C1["Deterministic grounding\n(numbers vs findings)"]
        C2["LLM quality scoring\nspecificity · driver · usefulness"]
        C1 --> C2
    end

    Critic -->|"score ≥ 7.5 → APPROVE"| Result["<b>PipelineResult</b>\n.report  .eval_score  .revision_count"]
    Critic -->|"score < 7.5 → revision_request"| Writer
```

---

## Critic Revision Loop

Evaluation is **inside** the pipeline, not a post-hoc script. Every report is reviewed before the user sees it.

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant Writer
    participant Critic

    Orch->>Writer: analyst_findings + market_findings
    Writer-->>Orch: draft_v1

    loop max 2 revisions
        Orch->>Critic: evaluate(draft, findings)
        Critic-->>Orch: CriticResult (score, issues)
        alt score ≥ 7.5 AND grounding OK
            Orch-->>Orch: APPROVE — exit
        else
            Orch->>Writer: revise(draft, revision_request)
            Writer-->>Orch: draft_v2 …
        end
    end
```

The Critic runs two checks independently:

1. **Deterministic grounding** — regex extracts all numbers from the report and compares them to the findings dict within 2% tolerance. No LLM needed.
2. **LLM quality score** — rates recommendation specificity, driver accuracy, and trader usefulness 0–10. Generates a targeted `revision_request` if any dimension falls short.

---

## Agent Responsibilities

| Agent | File | Tools | Returns |
|-------|------|-------|---------|
| **Analyst** | `analyst_agent.py` | `get_financial_summary` `get_dispatch_comparison` `analyze_soc_patterns` `compare_dispatch_timing` | `analyst_findings` dict |
| **Market** | `market_agent.py` | `identify_high_price_intervals` `get_price_forecast_accuracy` | `market_findings` dict |
| **Writer** | `writer_agent.py` | None | Markdown report string |
| **Critic** | `critic_agent.py` | None | `CriticResult` with score + revision |
| **Orchestrator** | `orchestrator.py` | Coordinates all | `PipelineResult` |

---

## Data Tools

All 6 tools live in `agent/tools.py`. They return compact JSON summaries — the LLM never sees raw CSV rows.

| Tool | Trader question it answers |
|------|---------------------------|
| `get_financial_summary` | What is the total revenue gap and %? |
| `get_dispatch_comparison` | How did charge/discharge behaviour differ between scenarios? |
| `analyze_soc_patterns` | Was the battery at the wrong SOC when prices spiked? |
| `compare_dispatch_timing` | Where did Historical and Perfect move in opposite directions? |
| `identify_high_price_intervals` | Which specific market events caused the biggest losses? |
| `get_price_forecast_accuracy` | How biased was the price forecast — and in which direction? |

---

## Project Structure

```
agentic-battery-trader/
├── agent/
│   ├── agent.py            # _react_loop() shared helper + legacy run_agent()
│   ├── prompts.py          # Per-agent system prompts (Analyst, Market, Writer, Critic)
│   ├── tools.py            # 6 data analysis tools + load_and_validate()
│   ├── analyst_agent.py    # Analyst: 4-tool ReAct loop → findings dict
│   ├── market_agent.py     # Market: 2-tool ReAct loop → findings dict
│   ├── writer_agent.py     # Writer: single LLM call → Markdown
│   ├── critic_agent.py     # Critic: grounding check + LLM judge → CriticResult
│   └── orchestrator.py     # Orchestrator: asyncio parallel + revision loop
├── data/                   # CSV input (gitignored)
├── output/                 # Generated reports (gitignored)
├── tests/
│   └── test_tools.py       # 19 unit tests for all 6 data tools
├── main.py                 # CLI entry point
├── DESIGN_SPEC.md          # Architectural deep-dive and agent contracts
├── output/
│   ├── EXAMPLE_REPORT.md   # Explicit deliverable: Sample pipeline output
│   └── vis_*.png           # Performance dashboards
├── scripts/
│   └── visualize.py        # Matplotlib engine for dashboards
├── requirements.txt
└── .env.example
```

---

## Example Q&A

These are the questions a battery trader would ask — and how the agent answers them using its tool pipeline.

---

### Q1: How much revenue did we leave on the table?

**Trader asks:** *"What was our total revenue gap vs. perfect foresight this week?"*

**How the agent answers:**
The Analyst Agent calls `get_financial_summary` as its first tool. The tool aggregates all `cleared` rows across both scenarios and returns:

```json
{
  "historical_revenue": 4851225.73,
  "perfect_revenue":    8058419.02,
  "gap_abs":            3207193.28,
  "gap_pct":            39.8,
  "interval_count":     288
}
```

The Writer uses these exact numbers in the financial summary table. The Critic verifies them against the tool output within 2% before approving.

**Report section produced:**

| Metric | Value |
|--------|-------|
| Historical Revenue | $4,851,225.73 |
| Perfect Revenue | $8,058,419.02 |
| Revenue Gap | **$3,207,193.28 (39.8%)** |
| Intervals Analysed | 288 |

---

### Q2: Why did we underperform — what was the main reason?

**Trader asks:** *"Was our underperformance mainly a timing problem, an SOC problem, or a forecasting problem?"*

**How the agent answers:**
The Analyst Agent calls three tools in sequence to test each hypothesis:

1. `analyze_soc_patterns` → Historical SOC at the top-10 price spikes averaged **13.15 MWh** vs Perfect's **72.97 MWh**. The battery was at minimum SOC **32.29%** of all intervals.
2. `compare_dispatch_timing` → **17 direction conflicts** (5.9% of intervals) where Historical charged while Perfect discharged, costing **−$43,385**.
3. `get_dispatch_comparison` → Historical discharged 388 MWh vs Perfect's 466 MWh — the battery was underutilised during peaks.

The agent synthesises: the primary driver is **SOC depletion before peak windows**, not timing conflicts (which were a smaller secondary factor).

**Report section produced:**

> **Primary Driver: Suboptimal SOC during high-price intervals**
>
> Historical SOC at the top-10 price spikes averaged only 13.15 MWh, versus Perfect's 72.97 MWh.
> The battery was at minimum SOC 32.29% of all intervals — unable to discharge when prices hit
> $20,300/MWh at 20:35, missing $342,564 in that single interval alone.

---

### Q3: Was the price forecast the problem?

**Trader asks:** *"Did the forecast lead us to make the wrong dispatch decisions?"*

**How the agent answers:**
The Market Agent calls `get_price_forecast_accuracy`:

```json
{
  "mape":              65.43,
  "mean_error":       -1009.83,
  "pct_underforecast": 57.64,
  "correlation":       0.87
}
```

A mean error of **−$1,009/MWh** means the model systematically predicted prices too low. 57.64% of intervals were underforecasts — which caused the battery to dispatch early at moderate prices, leaving it empty when the $20,000+ spike arrived.

**Report section produced:**

> **Secondary Factor: Systematic price forecast under-bias (MAPE 65.43%, 57.64% underforecast)**
>
> The forecast consistently under-predicted prices by ~$1,010/MWh on average, causing early
> discharge that depleted SOC before the evening spike window.

---

### Q4: What should we do differently tomorrow?

**Trader asks:** *"Give me two concrete things I can change in operations."*

**How the agent answers:**
The Writer Agent receives the merged findings and must follow a strict recommendation format: action, rationale, expected benefit, tradeoff — with specific numbers drawn from the findings.

**Report section produced:**

**Recommendation 1 — Protect SOC for the evening peak window**

| | |
|---|---|
| **Action** | Maintain minimum 100 MWh SOC from 17:00–21:30. Use the morning charging window (when prices are below $100/MWh) to pre-fill. |
| **Rationale** | Historical SOC at the top-10 price events averaged 13.15 MWh. Perfect Foresight held 72.97 MWh at those same moments. The entire $3.2M gap concentrates in this window. |
| **Expected Benefit** | Recover an estimated ~$1.5M of the gap by being available to discharge during $1,000–$20,000/MWh events. |
| **Tradeoff** | Requires holding capacity in reserve, reducing revenue from midday moderate-price opportunities. |

**Recommendation 2 — Re-calibrate the forecast model to correct the under-bias**

| | |
|---|---|
| **Action** | Add a +$1,010/MWh bias correction term to all forecasts. Target MAPE below 40% and underforecast rate below 35%. |
| **Rationale** | Mean forecast error of −$1,009.83/MWh caused the battery to treat evening peaks as moderate-price windows and dispatch early. |
| **Expected Benefit** | Improved dispatch decisions during high-volatility periods; better alignment between planned bids (expected) and cleared outcomes. |
| **Tradeoff** | Over-correction risks charging too aggressively, increasing battery degradation and missing low-price charging windows. |

---

### Q5: How do I know the report numbers are accurate?

**Trader asks:** *"Can I trust these figures, or is the AI making them up?"*

**How the agent answers:**
Every number in the report passes through the Critic Agent's **deterministic grounding check** before the report is delivered. The check extracts all numeric values from the report text and compares them to the tool output dicts within 2% tolerance. Any number that cannot be traced to a tool output triggers a revision request back to the Writer.

**Grounding provenance for the BLYTHB1 report:**

| Claim in report | Source tool | Status |
|----------------|-------------|--------|
| Gap = $3,207,193.28 (39.8%) | `get_financial_summary` | ✓ verified |
| Hist SOC at spikes = 13.15 MWh | `analyze_soc_patterns` | ✓ verified |
| Perfect SOC at spikes = 72.97 MWh | `analyze_soc_patterns` | ✓ verified |
| Missed $342,564 at 20:35 | `identify_high_price_intervals` | ✓ verified |
| MAPE = 65.43% | `get_price_forecast_accuracy` | ✓ verified |
| Mean forecast error = −$1,009.83/MWh | `get_price_forecast_accuracy` | ✓ verified |
| 57.64% underforecast | `get_price_forecast_accuracy` | ✓ verified |
| 17 direction conflicts | `compare_dispatch_timing` | ✓ verified |

---

## Expected Data Schema

The system generalises to any CSV matching this schema — no code changes required.

| Column | Type | Description |
|--------|------|-------------|
| `SCENARIO_NAME` | string | `historical` or `perfect` |
| `SCHEDULE_TYPE` | string | `expected` or `cleared` |
| `START_DATETIME` | datetime | Interval start (local time) |
| `SOC` | float | State of charge at end of interval (MWh) |
| `CHARGE_ENERGY` | float | Energy charged (MWh ≥ 0) |
| `DISCHARGE_ENERGY` | float | Energy discharged (MWh ≥ 0) |
| `PRICE_ENERGY` | float | Cleared energy price ($/MWh) |
| `REVENUE` | float | Revenue from energy market ($) |

---

## Evaluation Framework

The Critic Agent rates every report on four dimensions before approving:

| Dimension | Method | Scale |
|-----------|--------|-------|
| **Grounding** | Deterministic (regex + tolerance check) | Pass / Fail per number |
| **Recommendation specificity** | LLM judge | 0–10 |
| **Driver accuracy** | LLM judge | 0–10 |
| **Trader usefulness** | LLM judge | 0–10 |

**Approval rule:** grounding passes AND average LLM score ≥ 7.5 / 10.  
If either fails, the Critic generates a targeted `revision_request` and the Writer revises (up to 2 times).

---

## Running Tests

```bash
python -m pytest tests/ -v
```

19 unit tests cover `load_and_validate()` and all 6 data tools using synthetic DataFrames.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **4-agent specialisation** | Analyst + Market run in parallel. Each has a smaller tool set and tighter context — less hallucination risk, faster execution. |
| **No raw data to LLM** | All 6 tools return compact JSON summaries. The LLM never sees individual rows. |
| **Structured findings dict, not Markdown** | Analyst and Market return dicts. Writer gets clean, parseable input. Critic can do deterministic number matching. |
| **Critic inside the loop** | Evaluation happens before the report is delivered. Bad drafts are revised automatically. |
| **Deterministic + LLM grounding** | Regex catches fabricated numbers deterministically; LLM judge catches reasoning failures that regex cannot. |
| **Schema-driven generalisation** | `load_and_validate()` uses dynamic column references. Drop in any conforming CSV and re-run — no code changes needed. |
