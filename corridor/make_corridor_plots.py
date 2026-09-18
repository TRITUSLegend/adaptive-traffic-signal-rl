"""
Figures for CORRIDOR_REPORT.md.

Run:
    python corridor/make_corridor_plots.py

Output (results/corridor/plots/):
    corridor_training_curves.png   did the three learners converge?
    corridor_comparison.png        delay and throughput per scenario
    corridor_detector_failure.png  degradation as detectors die
    corridor_transfer.png          zero-shot on networks never trained on

PALETTE
-------
The five-slot categorical palette is inherited unchanged from
`eval/make_plots.py`, so a controller that appears in both studies keeps its
colour. MA2C needs a sixth slot; the extended six were run through the
validator and pass the lightness band, chroma floor, adjacent-pair CVD
separation (worst 9.1 protan) and the normal-vision floor (worst 19.6).

The validator's contrast warning is answered the way it must be: every figure
here has the corresponding generated table beside it in the report, so no value
is available by colour alone.

TWO GOTCHAS INHERITED FROM THE SINGLE-INTERSECTION FIGURES
----------------------------------------------------------
  * A figure must be CREATED at the same dpi it is SAVED at. Creating at the
    default 100 and saving at 160 moves every element that was positioned in
    display coordinates. `FIG_DPI` pins both.
  * Never a dual-axis chart. Waiting time and throughput are different scales
    and different questions, so they get adjacent panels rather than two y-axes
    on one plot - which would let the reader draw a crossing that means nothing.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corridor import paths  # noqa: E402
from corridor.evaluate_corridor import CONTROLLER_ORDER, LABELS  # noqa: E402
from corridor.make_corridor_tables import SCENARIO_ORDER, perturbed_cell  # noqa: E402
from corridor.robustness import FAILURE_RATES  # noqa: E402

FIG_DPI = 160
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"

# Fixed order, never cycled. The first five are the single-intersection slots.
COLOURS = {
    "fixed-time": "#2a78d6",     # slot 1 blue
    "actuated": "#eb6834",       # slot 2 orange
    "greedy-queue": "#1baf7a",   # slot 3 aqua
    "idqn": "#eda100",           # slot 4 yellow
    "ippo": "#e87ba4",           # slot 5 magenta
    "ma2c": "#8b5cf6",           # slot 6 violet - added for the corridor
}

CURVES = [("idqn", "Independent DQN"), ("ippo", "Independent PPO"),
          ("ma2c", "MA2C")]


def read(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def style(ax) -> None:
    """Recessive grid and axes; the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def smooth(values: list, window: int = 5) -> list:
    """A short trailing mean, so the trend is readable without hiding it."""
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def cell_mean(rows: list, controller: str, scenario: str, metric: str):
    values = [float(r[metric]) for r in rows
              if r["controller"] == controller and r["scenario"] == scenario]
    return statistics.mean(values) if values else None


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------
def training_curves() -> bool:
    series = [(name, label, read(paths.training_curve(name)))
              for name, label in CURVES]
    series = [(n, lab, rows) for n, lab, rows in series if rows]
    if not series:
        return False

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE, dpi=FIG_DPI)
    style(ax)
    for name, label, rows in series:
        episodes = [int(r["episode"]) for r in rows]
        queues = smooth([float(r["mean_queue_veh"]) for r in rows])
        ax.plot(episodes, queues, color=COLOURS[name], linewidth=2,
                label=label, zorder=3)
        # Direct label at the line's end: identity without reading the legend.
        ax.annotate(label, xy=(episodes[-1], queues[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=COLOURS[name], fontsize=9, va="center")

    ax.set_xlabel("Training episode (one simulated hour each)", color=INK_SOFT,
                  fontsize=10)
    ax.set_ylabel("Mean queue (vehicles)", color=INK_SOFT, fontsize=10)
    # The title states what the figure IS, not what it shows - a title that
    # asserts a conclusion goes stale the moment the data moves.
    ax.set_title("Corridor training: mean queue per episode",
                 color=INK, fontsize=12, loc="left", pad=12)
    # A single series needs no legend box; the direct label already names it.
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT,
                  loc="upper right")
    ax.set_xlim(left=0)
    fig.subplots_adjust(right=0.80)
    save(fig, "corridor_training_curves.png")
    return True


# ---------------------------------------------------------------------------
# 2. Delay and throughput per scenario - two panels, never two y-axes
# ---------------------------------------------------------------------------
def comparison() -> bool:
    rows = read(paths.EVALUATION_RAW)
    if not rows:
        return False

    present = [c for c in CONTROLLER_ORDER
               if any(r["controller"] == c for r in rows)]
    scenarios = [s for s in SCENARIO_ORDER
                 if any(r["scenario"] == s for r in rows)]
    if not present or not scenarios:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), facecolor=SURFACE,
                             dpi=FIG_DPI)
    metrics = [("mean_waiting_time_s", "Mean waiting time (s)",
                "Delay: lower is better"),
               ("throughput_veh", "Throughput (vehicles/hour)",
                "Throughput: the check that delay was not bought")]

    width = 0.8 / len(present)
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        style(ax)
        for slot, controller in enumerate(present):
            xs, ys = [], []
            for group, scenario in enumerate(scenarios):
                value = cell_mean(rows, controller, scenario, metric)
                if value is None:
                    continue
                xs.append(group - 0.4 + width * (slot + 0.5))
                ys.append(value)
            # A 2px surface gap between adjacent bars keeps them separable.
            ax.bar(xs, ys, width=width * 0.88, color=COLOURS[controller],
                   label=LABELS[controller], zorder=3, linewidth=0)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, fontsize=9, color=INK_SOFT)
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)

    # One legend for both panels: identity is never colour-alone.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOURS[c], linewidth=0)
               for c in present]
    fig.legend(handles, [LABELS[c] for c in present], frameon=False,
               fontsize=9, labelcolor=INK_SOFT, ncol=len(present),
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Corridor controllers across four demand scenarios",
                 color=INK, fontsize=13, x=0.09, ha="left")
    fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.22)
    save(fig, "corridor_comparison.png")
    return True


# ---------------------------------------------------------------------------
# 3. Detector failure
# ---------------------------------------------------------------------------
def detector_failure() -> bool:
    rows = read(paths.ROBUSTNESS_RAW)
    if not rows:
        return False

    present = [c for c in CONTROLLER_ORDER
               if any(r["controller"] == c for r in rows)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE, dpi=FIG_DPI)
    style(ax)

    xs = [100 * r for r in FAILURE_RATES]
    for controller in present:
        ys = [perturbed_cell(rows, controller, rate, "zero",
                             "mean_waiting_time_s") for rate in FAILURE_RATES]
        if any(v is None for v in ys):
            continue
        ax.plot(xs, ys, color=COLOURS[controller], linewidth=2,
                marker="o", markersize=5, label=LABELS[controller], zorder=3)

    ax.set_xlabel("Lane detectors failed (%, stuck-at-zero)", color=INK_SOFT,
                  fontsize=10)
    ax.set_ylabel("Mean waiting time (s)", color=INK_SOFT, fontsize=10)
    ax.set_title("Degradation as detectors die\n"
                 "Fixed-time reads no detectors, so its line is flat by "
                 "construction - it is the reference, not a winner",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=2)
    ax.set_xticks(xs)
    fig.subplots_adjust(top=0.80)
    save(fig, "corridor_detector_failure.png")
    return True


# ---------------------------------------------------------------------------
# 4. Topology transfer
# ---------------------------------------------------------------------------
def transfer() -> bool:
    rows = read(paths.TRANSFER_RAW)
    if not rows:
        return False

    networks = []
    for row in rows:
        if row["network"] not in networks:
            networks.append(row["network"])
    present = [c for c in CONTROLLER_ORDER
               if any(r["controller"] == c for r in rows)]
    if not networks or not present:
        return False

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE, dpi=FIG_DPI)
    style(ax)
    width = 0.8 / len(present)
    for slot, controller in enumerate(present):
        xs, ys = [], []
        for group, network in enumerate(networks):
            values = [float(r["mean_waiting_time_s"]) for r in rows
                      if r["controller"] == controller
                      and r["network"] == network]
            if not values:
                continue
            xs.append(group - 0.4 + width * (slot + 0.5))
            ys.append(statistics.mean(values))
        ax.bar(xs, ys, width=width * 0.88, color=COLOURS[controller],
               label=LABELS[controller], zorder=3, linewidth=0)

    ax.set_xticks(range(len(networks)))
    ax.set_xticklabels(networks, fontsize=9, color=INK_SOFT)
    ax.set_ylabel("Mean waiting time (s)", color=INK_SOFT, fontsize=10)
    ax.set_title("Zero-shot transfer to networks never trained on\n"
                 "cologne3 is real geometry and needs an observation "
                 "projection; the corridors do not",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=3)
    fig.subplots_adjust(top=0.80)
    save(fig, "corridor_transfer.png")
    return True


def save(fig, name: str) -> None:
    paths.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.PLOTS_DIR / name
    fig.savefig(out, dpi=FIG_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("  %-34s %6d KB" % (name, out.stat().st_size // 1024))


def main() -> None:
    print("Figures written to %s"
          % paths.PLOTS_DIR.relative_to(paths.ROOT).as_posix())
    built = 0
    for label, builder in [("training curves", training_curves),
                           ("comparison", comparison),
                           ("detector failure", detector_failure),
                           ("transfer", transfer)]:
        if builder():
            built += 1
        else:
            print("  (no data yet for %s)" % label)
    if not built:
        print("  nothing to draw - run the evaluation first")


if __name__ == "__main__":
    main()
