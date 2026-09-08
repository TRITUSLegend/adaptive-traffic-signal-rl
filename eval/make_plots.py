"""
Turn the evaluation CSVs into the figures used in REPORT.md.

Run (after eval/evaluate.py has produced its CSVs):
    python eval/make_plots.py

Outputs, into results/plots/:
    training_curve.png        did the agent actually learn, and did it plateau?
    controller_comparison.png waiting time, queue and throughput side by side
    robustness.png            performance on demand patterns never trained on
    queue_over_time.png       what the hour actually looks like, second by second

DESIGN NOTES
------------
The colour order is fixed: each controller keeps the same colour in every
figure, so a reader who learns "orange is the tuned fixed-time plan" in one
chart can carry that across the report. Colours are never assigned by rank -
if they were, a controller would change colour whenever it moved up or down.

Three of the five palette colours fall below a 3:1 contrast ratio against the
page, so every bar is directly labelled with its value; identity and magnitude
are then both readable without relying on colour alone.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # render to files; no interactive window needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from eval.paths import LOGS_DIR, PLOTS_DIR, trace_path  # noqa: E402

# The rounded bars are positioned in pixel coordinates, which are only valid
# for one dpi. The figure is therefore CREATED at the same dpi it is SAVED
# at; letting savefig re-render at a different dpi would move every bar.
FIG_DPI = 160

# --- palette ---------------------------------------------------------------
# Validated categorical palette (see the data-visualisation guidelines): the
# slot order is what keeps adjacent colours distinguishable to colour-blind
# readers, so it must not be reshuffled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"

COLOURS = {
    "fixed-time": "#2a78d6",        # slot 1 blue
    "fixed-time-tuned": "#eb6834",  # slot 2 orange
    "greedy-queue": "#1baf7a",      # slot 3 aqua
    "dqn": "#eda100",               # slot 4 yellow
    "ppo": "#e87ba4",               # slot 5 magenta
}
LABELS = {
    "fixed-time": "Fixed-time (untuned)",
    "fixed-time-tuned": "Fixed-time (Webster-tuned)",
    "greedy-queue": "Greedy queue heuristic",
    "dqn": "DQN (learned)",
    "ppo": "PPO (learned)",
}
ORDER = ["fixed-time", "fixed-time-tuned", "greedy-queue", "dqn", "ppo"]

MAIN_SCENARIOS = ["balanced", "ns_heavy", "peak"]
ROBUSTNESS_SCENARIOS = ["ew_heavy", "surge", "light"]
SCENARIO_LABELS = {
    "balanced": "balanced\n(400 veh/h each)",
    "ns_heavy": "ns_heavy  *TRAINED*\n(600 NS / 300 EW)",
    "peak": "peak\n(700 NS / 450 EW)",
    "ew_heavy": "ew_heavy\n(demand mirrored)",
    "surge": "surge\n(+30% volume)",
    "light": "light\n(-40% volume)",
}


def style_axes(ax, ylabel: str = "", title: str = "") -> None:
    """Recessive grid and axes, so the data is the loudest thing on the chart."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9.5)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold",
                     loc="left", pad=10)


def rounded_bar(ax, x, value, width, colour, radius_px=5.0):
    """
    Draw a bar with a rounded top and a flat base.

    matplotlib has no rounded bars. The obvious workaround - a rounded box in
    data coordinates - does not work, because the rounding radius would then be
    measured in *data* units, which differ wildly between the x axis (scenario
    index, range ~3) and the y axis (seconds, range ~60). The corner would be
    invisible on one axis and enormous on the other.

    So we place the box in DISPLAY (pixel) coordinates instead, where a radius
    of 5 px means 5 px on both axes. The box is extended below the axis floor
    by the radius and clipped to the axes, which leaves the top corners rounded
    and the base flat - a bar, not a lozenge.

    This requires the figure layout to be final, so call it only after
    `fig.canvas.draw()`.
    """
    base = ax.get_ylim()[0]
    x0, y0 = ax.transData.transform((x - width / 2, base))
    x1, y1 = ax.transData.transform((x + width / 2, value))

    box_w = x1 - x0
    box_h = y1 - y0
    if box_h <= 0:
        return None
    radius = min(radius_px, box_w / 2.0, box_h / 2.0)

    patch = FancyBboxPatch(
        (x0, y0 - radius), box_w, box_h + radius,
        boxstyle="round,pad=0,rounding_size=%f" % radius,
        linewidth=0, facecolor=colour, zorder=3,
        transform=None,          # coordinates are already in pixels
        clip_on=True, clip_box=ax.bbox,
    )
    ax.add_patch(patch)
    return patch


def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def present_controllers(summary: list) -> list:
    found = {row["controller"] for row in summary}
    return [c for c in ORDER if c in found]


def lookup(summary: list, controller: str, scenario: str, field: str) -> float:
    for row in summary:
        if row["controller"] == controller and row["scenario"] == scenario:
            return to_float(row.get(field))
    return float("nan")


# ---------------------------------------------------------------------------
# Figure 1: the training curve
# ---------------------------------------------------------------------------
def plot_training_curve(summary: list) -> Path | None:
    """
    Show that learning happened and then stopped improving.

    We plot mean queue length rather than the raw reward, because they are the
    same quantity up to a constant factor and "vehicles queued" is a unit a
    reader can reason about. The fixed-time baseline is drawn as a reference
    line so the moment the agent overtakes it is visible.
    """
    curves = []
    for algo in ("dqn", "ppo"):
        rows = read_csv(LOGS_DIR / ("training_curve_%s.csv" % algo))
        if rows:
            curves.append((algo, rows))
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE, dpi=FIG_DPI)
    style_axes(ax, ylabel="Mean queue length (vehicles)",
               title="Learning progress: queue length per training episode")

    for curve_index, (algo, rows) in enumerate(curves):
        episodes = [int(r["episode"]) for r in rows]
        queues = [to_float(r["mean_queue_veh"]) for r in rows]
        colour = COLOURS[algo]

        # Raw per-episode values, kept faint: each episode is a different hour
        # of random traffic, so the scatter is real variation, not measurement
        # error, and hiding it would overstate how smooth learning was.
        ax.plot(episodes, queues, color=colour, linewidth=1, alpha=0.22, zorder=2)

        # 10-episode moving average. The window is exactly one full rotation
        # through the ten training demand files, so it averages out which
        # particular hour the agent happened to get.
        window = 10
        smoothed = [
            sum(queues[max(0, i - window + 1):i + 1]) /
            len(queues[max(0, i - window + 1):i + 1])
            for i in range(len(queues))
        ]
        ax.plot(episodes, smoothed, color=colour, linewidth=2, zorder=4,
                label="%s (10-episode average)" % LABELS[algo].split(" (")[0])
        # Both algorithms converge to almost the same queue length, so their
        # end labels would print on top of each other. Nudge them apart
        # vertically rather than letting one hide the other.
        ax.annotate("%s: %.1f" % (algo.upper(), smoothed[-1]),
                    xy=(episodes[-1], smoothed[-1]),
                    xytext=(6, 9 if curve_index == 0 else -9),
                    textcoords="offset points",
                    color=INK_SOFT, fontsize=9, va="center", fontweight="bold")

    # Reference lines for the non-learning controllers, on the trained scenario.
    for controller in ("fixed-time", "greedy-queue"):
        value = lookup(summary, controller, "ns_heavy", "mean_queue_length_veh_mean")
        if value != value:  # NaN check
            continue
        ax.axhline(value, color=COLOURS[controller], linewidth=1.5,
                   linestyle=(0, (5, 4)), zorder=3)
        ax.annotate("%s: %.1f" % (LABELS[controller], value),
                    xy=(0.012, value), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    color=INK_SOFT, fontsize=8.5)

    ax.set_xlabel("Training episode (one simulated hour each)",
                  color=INK_SOFT, fontsize=9.5)
    ax.set_ylim(bottom=0)
    # Leave room on the right for the end-of-curve labels, which would
    # otherwise be clipped at the axes edge.
    last_episode = max(int(rows[-1]["episode"]) for _, rows in curves)
    ax.set_xlim(1, last_episode * 1.09)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    fig.text(0.008, 0.015,
             "Dashed lines are the non-learning controllers on the same demand. "
             "Lower is better.",
             color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = PLOTS_DIR / "training_curve.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: the headline comparison
# ---------------------------------------------------------------------------
def plot_controller_comparison(summary: list, scenarios: list, filename: str,
                               title: str) -> Path | None:
    """Three panels - waiting time, queue, throughput - grouped by scenario."""
    scenarios = [s for s in scenarios
                 if any(r["scenario"] == s for r in summary)]
    if not scenarios:
        return None
    controllers = present_controllers(summary)

    panels = [
        ("mean_waiting_time_s",
         "Mean waiting time per vehicle (s)   ·   lower is better"),
        ("mean_queue_length_veh",
         "Mean queue length (vehicles)   ·   lower is better"),
        ("throughput_veh",
         "Throughput (vehicles/hour)   ·   higher is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), facecolor=SURFACE, dpi=FIG_DPI)

    # Pass 1: set up each panel's scale. The bars themselves cannot be drawn
    # yet, because their rounded corners are measured in pixels and the pixel
    # geometry is not final until the layout has been computed.
    plan = []
    group_width = 0.82
    # Cap the bar width so a chart with only one scenario group does not end up
    # with absurdly wide bars.
    bar_width = min(group_width / len(controllers), 0.17)

    for ax, (field, label) in zip(axes, panels):
        style_axes(ax, title=label)
        bars = []
        values_seen = []
        for slot, controller in enumerate(controllers):
            offset = bar_width * (slot - (len(controllers) - 1) / 2.0)
            for group, scenario in enumerate(scenarios):
                value = lookup(summary, controller, scenario, field + "_mean")
                if value != value:  # NaN
                    continue
                std = lookup(summary, controller, scenario, field + "_std")
                bars.append((group + offset, value, std, COLOURS[controller], slot))
                values_seen.append(value)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios],
                           fontsize=8.5, color=INK_SOFT)
        ax.set_xlim(-0.5, len(scenarios) - 0.5)

        top = max(values_seen) if values_seen else 1.0
        low = min(values_seen) if values_seen else 0.0
        spread = (top - low) / top if top else 0.0

        if field == "throughput_veh" and spread < 0.15:
            # When every controller moves roughly the same number of vehicles,
            # an axis starting at zero renders five identical-looking bars and
            # hides the comparison the panel exists to make. Truncating is
            # legitimate here because the axis is labelled and every bar
            # carries its own value - but only here. When the panel spans
            # scenarios with genuinely different totals (1000 vs 2250 veh/h),
            # the differences are real and large, so the axis starts at zero
            # and shows them honestly.
            pad = max(top - low, 1.0)
            ax.set_ylim(max(0.0, low - pad * 1.4 - 5), top + pad * 0.95 + 8)
        elif field == "throughput_veh":
            ax.set_ylim(0, top * 1.20)
        else:
            ax.set_ylim(0, top * 1.24)
        plan.append((ax, bars, field == "throughput_veh"))

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOURS[c], linewidth=0)
               for c in controllers]
    legend = fig.legend(handles, [LABELS[c] for c in controllers],
                        loc="lower center", ncol=len(controllers), frameon=False,
                        fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    fig.suptitle(title, color=INK, fontsize=13, fontweight="bold", x=0.006,
                 ha="left", y=0.985)
    fig.text(0.006, 0.075,
             "Bars are the mean of 5 independent simulated hours; the vertical "
             "line spans one standard deviation across those hours.",
             color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.canvas.draw()  # freeze the layout so pixel coordinates are valid

    # Pass 2: draw the bars, their spread markers and their value labels.
    for ax, bars, stagger in plan:
        for x, value, std, colour, slot in bars:
            rounded_bar(ax, x, value, bar_width * 0.84, colour)

            # Spread across the five random hours, drawn thin and in muted ink
            # so it reads as an annotation on the bar rather than a second
            # series.
            if std > 0:
                ax.plot([x, x], [value - std, value + std],
                        color=INK_SOFT, linewidth=1, zorder=5, alpha=0.5)

            # Direct value label. Required, not decorative: three of the five
            # palette colours fall below 3:1 contrast against this surface, so
            # the numbers must be legible without depending on the fill.
            # Throughput values are four digits on bars of near-equal
            # height, so neighbouring labels would overlap and become
            # unreadable. Alternating their height separates them.
            offset_pts = 4 + (slot % 2) * 10 if stagger else 4
            ax.annotate("%.0f" % value if value >= 100 else "%.1f" % value,
                        xy=(x, value + std), xytext=(0, offset_pts),
                        textcoords="offset points", ha="center",
                        color=INK_SOFT, fontsize=7.6, zorder=6)

    path = PLOTS_DIR / filename
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: queue over the hour
# ---------------------------------------------------------------------------
def plot_queue_over_time(scenario: str = "ns_heavy") -> Path | None:
    """
    The hour as it actually unfolds, rather than collapsed to one number.

    This is the figure that shows *how* the controllers differ: the fixed-time
    plan saws up and down as each cycle admits a fixed slug of traffic, while
    the adaptive controllers hold the queue near a steady level.
    """
    series = []
    for controller in ORDER:
        rows = read_csv(trace_path(controller, scenario))
        if rows:
            series.append((controller, rows))
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE, dpi=FIG_DPI)
    style_axes(ax, ylabel="Vehicles queued at the intersection",
               title="Queue length through one hour  (%s demand)" % scenario)

    for controller, rows in series:
        minutes = [to_float(r["sim_time"]) / 60.0 for r in rows]
        queue = [to_float(r["queue_veh"]) for r in rows]
        # A 12-sample (one minute) rolling mean. The raw signal swings with
        # every signal cycle; smoothing to the minute shows the level each
        # controller settles at without hiding the trend.
        window = 12
        smoothed = [
            sum(queue[max(0, i - window + 1):i + 1]) /
            len(queue[max(0, i - window + 1):i + 1])
            for i in range(len(queue))
        ]
        ax.plot(minutes, smoothed, color=COLOURS[controller], linewidth=2,
                label=LABELS[controller], zorder=3)

    ax.set_xlabel("Simulated time (minutes)", color=INK_SOFT, fontsize=9.5)
    # Headroom above the tallest peak so the legend does not sit on the data.
    peak = max(max(to_float(r["queue_veh"]) for r in rows) for _, rows in series)
    ax.set_ylim(0, peak * 1.30)
    ax.set_xlim(0, 60)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    fig.text(0.008, 0.015,
             "Smoothed with a 1-minute rolling mean. Lower is better.",
             color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = PLOTS_DIR / "queue_over_time.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_csv(LOGS_DIR / "evaluation_summary.csv")
    if not summary:
        raise SystemExit(
            "No evaluation_summary.csv - run:  python eval/evaluate.py")

    written = [
        plot_training_curve(summary),
        plot_controller_comparison(
            summary, MAIN_SCENARIOS, "controller_comparison.png",
            "Fixed-time vs learned control, across three demand levels"),
        plot_controller_comparison(
            summary, ROBUSTNESS_SCENARIOS, "robustness.png",
            "Robustness: demand patterns the agent was never trained on"),
        plot_queue_over_time(),
    ]

    print("Figures written to " + str(PLOTS_DIR))
    for path in written:
        if path is None:
            continue
        print("  %-28s %6.0f KB" % (path.name, path.stat().st_size / 1024))
    missing = sum(1 for p in written if p is None)
    if missing:
        print("\n%d figure(s) skipped for lack of input data." % missing)


if __name__ == "__main__":
    main()
