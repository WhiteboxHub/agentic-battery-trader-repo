"""
Battery Performance Visualizer — Plot Price vs. SOC for battery storage.
This tool generates a "Performance Dashboard" comparing Historical vs. Perfect Foresight.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def visualize(csv_path: str, output_path: str = None) -> None:
    # 1. Load and prepare data
    df = pd.read_csv(csv_path, parse_dates=["START_DATETIME"])
    df["SCENARIO_NAME"] = df["SCENARIO_NAME"].str.strip().str.lower()
    df["SCHEDULE_TYPE"] = df["SCHEDULE_TYPE"].str.strip().str.lower()
    
    # Filter to 'cleared' rows which represent realized operation
    cleared = df[df["SCHEDULE_TYPE"] == "cleared"].copy()
    cleared = cleared.sort_values("START_DATETIME")
    
    hist = cleared[cleared["SCENARIO_NAME"] == "historical"]
    perf = cleared[cleared["SCENARIO_NAME"] == "perfect"]
    
    if len(hist) == 0 or len(perf) == 0:
        print("ERROR: Could not find both 'historical' and 'perfect' scenarios in data.")
        return

    # 2. Setup Figure
    plt.style.use('bmh')  # cleaner look
    fig, (ax_price, ax_soc) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Battery Performance Dashboard: {Path(csv_path).stem}", fontsize=16)

    # 3. Plot 1: Energy Price (Top)
    # Price is same for both scenarios, just use hist for values
    ax_price.plot(hist["START_DATETIME"], hist["PRICE_ENERGY"], color="#34495e", label="Market Price ($/MWh)")
    ax_price.set_ylabel("Price ($/MWh)")
    ax_price.grid(True, alpha=0.3)
    
    # Optional: Highlight extreme price spikes
    spikes = hist[hist["PRICE_ENERGY"] > 1000]
    if not spikes.empty:
        ax_price.scatter(spikes["START_DATETIME"], spikes["PRICE_ENERGY"], color="red", s=20, label="High Price Spike (>$1000)")

    ax_price.legend(loc="upper right")

    # 4. Plot 2: State of Charge (Bottom)
    ax_soc.plot(hist["START_DATETIME"], hist["SOC"], label="Historical SOC", color="#e67e22", linewidth=1.5)
    ax_soc.plot(perf["START_DATETIME"], perf["SOC"], label="Perfect Foresight SOC", color="#27ae60", linewidth=1.5, linestyle="--")
    
    # Fill gaps for visibility
    ax_soc.fill_between(hist["START_DATETIME"], hist["SOC"], color="#e67e22", alpha=0.1)
    ax_soc.fill_between(perf["START_DATETIME"], perf["SOC"], color="#27ae60", alpha=0.05)

    ax_soc.set_ylabel("SOC (MWh)")
    ax_soc.set_xlabel("Time (Local)")
    ax_soc.grid(True, alpha=0.3)
    ax_soc.legend(loc="upper right")

    # 5. Save/Show
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Dashboard saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize battery performance data.")
    parser.add_argument("csv_path", help="Path to battery data CSV")
    parser.add_argument("--output", help="Output PNG path (optional)")
    args = parser.parse_args()
    
    csv_stem = Path(args.csv_path).stem
    out = args.output or f"output/vis_{csv_stem}.png"
    
    visualize(args.csv_path, out)
