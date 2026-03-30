#!/usr/bin/env python3
"""Generate benchmark comparison charts from oha JSON results.

Usage: python3 chart.py [results_dir] [output_dir]
  results_dir defaults to ./results
  output_dir  defaults to ./charts
"""

import json
import os
import re
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("error: matplotlib required — pip install matplotlib", file=sys.stderr)
    sys.exit(1)

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
CHARTS_DIR = sys.argv[2] if len(sys.argv) > 2 else "charts"

# gateway -> scenario -> rps -> {p50, p90, p99, rps_actual, success}
data = defaultdict(lambda: defaultdict(dict))

# Filename patterns: {gw}-{scenario}-{rps}rps.json or {gw}-concurrent-{conc}.json
FILE_RE = re.compile(r"^(.+?)-(.+?)-(\d+)(rps|)\.json$")

for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith(".json"):
        continue
    m = FILE_RE.match(fname)
    if not m:
        continue
    gw, scenario, level = m.group(1), m.group(2), int(m.group(3))

    path = os.path.join(RESULTS_DIR, fname)
    try:
        with open(path) as f:
            j = json.load(f)
    except (json.JSONDecodeError, OSError):
        continue

    p = j.get("latencyPercentiles", {})
    s = j.get("summary", {})
    if not p.get("p50"):
        continue

    data[gw][scenario][level] = {
        "p50": p["p50"] * 1000,  # seconds -> ms
        "p90": p["p90"] * 1000,
        "p99": p["p99"] * 1000,
        "rps": s.get("requestsPerSec", 0),
        "success": s.get("successRate", 0),
    }

if not data:
    print(f"No results found in {RESULTS_DIR}/", file=sys.stderr)
    sys.exit(1)

os.makedirs(CHARTS_DIR, exist_ok=True)

# Collect all scenarios and gateways
gateways = sorted(data.keys())
all_scenarios = sorted({s for gw in data.values() for s in gw})

COLORS = {
    "direct": "#999999",
    "crabllm": "#e74c3c",
    "bifrost": "#3498db",
    "litellm": "#2ecc71",
}

def gw_color(gw):
    return COLORS.get(gw, "#95a5a6")


def chart_latency_vs_rps(scenario):
    """Line chart: P50/P99 latency vs RPS for each gateway."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for gw in gateways:
        points = data[gw].get(scenario, {})
        if not points:
            continue
        levels = sorted(points.keys())
        p50 = [points[l]["p50"] for l in levels]
        p99 = [points[l]["p99"] for l in levels]

        color = gw_color(gw)
        ax.plot(levels, p50, "o-", color=color, label=f"{gw} P50")
        ax.plot(levels, p99, "s--", color=color, alpha=0.5, label=f"{gw} P99")

    ax.set_xlabel("Target RPS")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency vs RPS — {scenario}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"latency-{scenario}.png"), dpi=150)
    plt.close(fig)


def chart_overhead_bars():
    """Grouped bar chart: gateway overhead at a fixed RPS across scenarios."""
    # Pick the highest common RPS level
    common_levels = None
    for gw in gateways:
        for scenario in all_scenarios:
            levels = set(data[gw].get(scenario, {}).keys())
            if levels:
                common_levels = levels if common_levels is None else common_levels & levels
    if not common_levels:
        return

    rps_level = max(common_levels)
    scenarios_with_data = [
        s for s in all_scenarios
        if all(rps_level in data[gw].get(s, {}) for gw in gateways)
    ]
    if not scenarios_with_data:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(scenarios_with_data))
    width = 0.8 / len(gateways)

    for i, gw in enumerate(gateways):
        p50s = [data[gw][s][rps_level]["p50"] for s in scenarios_with_data]
        offset = (i - len(gateways) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], p50s, width, label=gw, color=gw_color(gw))
        for bar, val in zip(bars, p50s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("P50 Latency (ms)")
    ax.set_title(f"Gateway Overhead Comparison — {rps_level} RPS")
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios_with_data, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "overhead-comparison.png"), dpi=150)
    plt.close(fig)


def chart_success_rates():
    """Heatmap-style chart showing success rates across gateways and scenarios."""
    # Collect all (scenario, rps) pairs
    pairs = sorted({
        (s, l)
        for gw in data.values()
        for s, levels in gw.items()
        for l in levels
    })
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(gateways) * 0.8)))
    labels = [f"{s}@{l}" for s, l in pairs]

    for gi, gw in enumerate(gateways):
        rates = []
        for s, l in pairs:
            entry = data[gw].get(s, {}).get(l)
            rates.append(entry["success"] if entry else 0)
        color = gw_color(gw)
        ax.barh(
            [gi + j * (len(gateways) + 1) for j in range(len(pairs))],
            rates, color=color, label=gw if gi == 0 or True else None,
        )

    # Simpler approach: grouped bars
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    x = range(len(pairs))
    width = 0.8 / len(gateways)

    for i, gw in enumerate(gateways):
        rates = []
        for s, l in pairs:
            entry = data[gw].get(s, {}).get(l)
            rates.append(entry["success"] if entry else 0)
        offset = (i - len(gateways) / 2 + 0.5) * width
        ax2.bar([xi + offset for xi in x], rates, width, label=gw, color=gw_color(gw))

    ax2.set_xlabel("Scenario @ RPS")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Success Rates")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(CHARTS_DIR, "success-rates.png"), dpi=150)
    plt.close(fig)
    plt.close(fig2)


# Generate all charts
for scenario in all_scenarios:
    chart_latency_vs_rps(scenario)

chart_overhead_bars()
chart_success_rates()

print(f"Charts saved to {CHARTS_DIR}/")
for f in sorted(os.listdir(CHARTS_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
