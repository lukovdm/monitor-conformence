from collections import namedtuple
from fractions import Fraction
from math import ceil, floor, log10
import re
from typing import Any, cast
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from itertools import combinations, product

# Fields used to match experiments across methods when pairing. ``seed`` is
# included so that, when a benchmark is run under several seeds, each seed of one
# method is paired with the *same* seed of the other (giving one point per seed
# rather than mispairing different seeds). For single-seed or older data with no
# ``seed`` field, ``.get("seed")`` is None on both sides, so this is a no-op.
DEFAULT_MATCH_FIELDS = [
    "name",
    "file",
    "parameters",
    "horizon",
    "threshold",
    "fp_slack",
    "fn_slack",
    "spec",
    "good_label",
    "seed",
]


# Numeric fields stored inconsistently across runs: ``Experiment.run`` sharpens
# ``threshold``/``fp_slack``/``fn_slack`` to exact ``Fraction``s on finished
# runs, while not-started / unfinished placeholders keep the original ``float``.
# Raw equality (``Fraction(3,10) == 0.3``) is then False and the pair is dropped,
# silently losing a finished run from the scatter when its counterpart never
# started. Canonicalise to a stable rounded float so the two forms match.
_NUMERIC_MATCH_FIELDS = {"threshold", "fp_slack", "fn_slack"}


def _canon_for_pair(field: str, value):
    if field not in _NUMERIC_MATCH_FIELDS or value is None:
        return value
    try:
        return round(float(Fraction(str(value))), 9)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return value


def pair_by_benchmark(
    data1: list[dict],
    data2: list[dict],
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
) -> list[tuple[dict, dict]]:
    """Pair entries from data1 and data2 that share the same benchmark.

    Matching is done on match_fields in d["experiment"]. Unmatched entries
    are silently dropped.
    """
    pairs = []
    for d1 in data1:
        for d2 in data2:
            if all(
                _canon_for_pair(f, d1["experiment"].get(f))
                == _canon_for_pair(f, d2["experiment"].get(f))
                for f in match_fields
            ):
                pairs.append((d1, d2))
                break
    return pairs


def _resolve_time(
    d: dict,
    timeout: float,
    out_of_memory: float,
    incorrect: float,
) -> float:
    """Map an experiment entry to a plot-ready time value.

    Failed experiments map to sentinel values (timeout/OOM/incorrect);
    correct experiments return results["time"].
    """
    if d["results"] is None:
        return timeout if d.get("error") == "timeout" else out_of_memory
    r = d["results"]
    threshold = d["experiment"]["threshold"]
    fp_slack = d["experiment"]["fp_slack"]
    fn_slack = d["experiment"]["fn_slack"]
    fp, fn = r.get("false_positive"), r.get("false_negative")
    if (
        fp is None
        or fn is None
        or fp < threshold - fp_slack
        or fn > threshold + fn_slack
    ):
        return incorrect
    return r["time"]


def calculate_error_lines(
    data: list[dict],
    value_func,
) -> tuple[float, float, float, float, float]:
    """Compute sentinel line positions based on the maximum observed value."""
    # Only positive values count: unfinished runs map to 0 here, and if *nothing*
    # has finished yet we must still fall back to 1.0 (a 0 bound breaks the log
    # axis in setup_loglog_comparison).
    max_time = max(
        (
            v
            for d in data
            for v in [value_func(d)]
            if isinstance(v, (int, float, Fraction)) and v > 0
        ),
        default=1.0,
    )
    timeout = max_time * 1.5
    out_of_memory = timeout * 4
    incorrect = out_of_memory * 4
    unfinished = incorrect * 4
    return float(max_time), timeout, out_of_memory, incorrect, unfinished


def setup_loglog_comparison(
    ax,
    max_lim: float,
    label1: str,
    label2: str,
    min_value: float = 1,
    sentinels: list[tuple[float, str]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_y_axis: bool = True,
):
    """Decorate a log-log comparison axes with diagonal guides, region labels,
    optional sentinel lines (timeout/OOM/...), and $10^i$ major + minor ticks.

    Call this after plotting your data points. `max_lim` is the data-driven
    upper bound used for the diagonal guides; sentinel positions extend the
    visible range automatically.
    """
    # Guard the log axis: with no finished runs the data-driven bound can be 0
    # (or below min_value), which would make log10 fail / produce no ticks.
    max_lim = max(max_lim, min_value * 10)
    ax.plot([0, max_lim], [0, max_lim], "-", color="0.5")
    ax.plot([0, max_lim], [0, max_lim / 10], "--", color="0.5", label="10x faster")
    ax.plot([0, max_lim / 10], [0, max_lim], "--", color="0.5")
    ax.plot([0, max_lim], [0, max_lim / 100], ":", color="0.5", label="100x faster")
    ax.plot([0, max_lim / 100], [0, max_lim], ":", color="0.5")

    ax.text(0.05, 0.95, f"{label1} faster", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="0.4")
    ax.text(0.95, 0.05, f"{label2} faster", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="0.4")

    upper = max_lim
    sentinel_label_map = {
        "timeout": r"$\infty$",
        "out of memory": "MO",
        "incorrect": r"$\times$",
        "unfinished": "U",
    }
    if sentinels:
        for sentinel, label in sentinels:
            ax.axhline(sentinel, color="gray", linestyle="--", label=f"{label2} {label}")
            ax.axvline(sentinel, color="gray", linestyle="--", label=f"{label1} {label}")
            upper = max(upper, sentinel)
        upper *= 2

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_value, upper)
    ax.set_ylim(min_value, upper)
    ax.grid()

    min_pow = ceil(log10(min_value))
    max_pow = ceil(log10(max_lim))
    tick_positions = [10**i for i in range(min_pow, max_pow)]
    tick_labels = [f"$10^{{{i}}}$" for i in range(min_pow, max_pow)]
    for sentinel, label in sentinels or []:
        tick_positions.append(sentinel)
        tick_labels.append(sentinel_label_map.get(label, label))
    minor_ticks = [k * 10**i for i in range(min_pow - 1, max_pow) for k in range(2, 10)]

    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xlabel(xlabel if xlabel else f"{label1} (s log)")
    if show_y_axis:
        ax.set_yticks(tick_positions, tick_labels)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_ylabel(ylabel if ylabel else f"{label2} (s log)")
    else:
        ax.set_yticks(tick_positions, [])
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_ylabel("")


# One paired point of a runtime comparison: the two source entries, their
# plot-ready (already sentinel-resolved) times, and the per-benchmark colour /
# marker / legend label. ``RuntimeComparison`` bundles the points with the
# sentinel-line positions so renderers don't recompute anything.
ComparisonPoint = namedtuple("ComparisonPoint", "d1 d2 x y color symbol label")
RuntimeComparison = namedtuple(
    "RuntimeComparison", "points max_time timeout out_of_memory incorrect"
)


def compare_runtime_data(
    data1: list[dict[str, Any]],
    data2: list[dict[str, Any]],
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
) -> RuntimeComparison:
    """Pair two methods by benchmark and resolve each run to a plot-ready point.

    Does *all* the data work behind the runtime scatter: ``pair_by_benchmark``
    matches an entry of ``data1`` with the same-benchmark/-seed entry of
    ``data2``; ``calculate_error_lines`` derives the timeout / out-of-memory /
    incorrect sentinel positions from the largest finished time; and
    ``_resolve_time`` maps every run to an (x, y) coordinate (a sentinel when it
    failed). The matplotlib ``compare_runtimes`` and the plotly
    ``plotly_compare_runtimes`` share this function and differ only in rendering.
    """
    pairs = pair_by_benchmark(data1, data2, match_fields)

    max_time, timeout, out_of_memory, incorrect, _unfinished = calculate_error_lines(
        [d for pair in pairs for d in pair],
        lambda d: d["results"]["time"] if d["results"] is not None else 0,
    )

    points = [
        ComparisonPoint(
            d1,
            d2,
            _resolve_time(d1, timeout, out_of_memory, incorrect),
            _resolve_time(d2, timeout, out_of_memory, incorrect),
            d1["color"],
            d1["symbol"],
            name_func(d1, d2),
        )
        for d1, d2 in pairs
    ]
    return RuntimeComparison(points, max_time, timeout, out_of_memory, incorrect)


def compare_runtimes(
    data1: list[dict[str, Any]],
    data2: list[dict[str, Any]],
    label1: str = "Method 1",
    label2: str = "Method 2",
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
    experiments_in_legends: bool = True,
    save_figures: bool = False,
    save_path: str = "./",
    file_name: str = "runtime",
    show_y_axis: bool = True,
    plot_kwargs: dict = {},
    min_value: float = 1,
):
    cmp = compare_runtime_data(data1, data2, match_fields, name_func)

    for p in cmp.points:
        plt.plot(
            max(p.x, min_value),
            max(p.y, min_value),
            p.symbol,
            color=p.color,
            label=p.label if experiments_in_legends else None,
            **plot_kwargs,
        )

    max_time, timeout, out_of_memory, incorrect = (
        cmp.max_time,
        cmp.timeout,
        cmp.out_of_memory,
        cmp.incorrect,
    )
    setup_loglog_comparison(
        plt.gca(),
        max_lim=max_time,
        label1=label1,
        label2=label2,
        min_value=min_value,
        sentinels=[
            (timeout, "timeout"),
            (out_of_memory, "out of memory"),
            (incorrect, "incorrect"),
        ],
        xlabel=xlabel,
        ylabel=ylabel,
        show_y_axis=show_y_axis,
    )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


# Matplotlib marker -> plotly symbol, for reusing the per-benchmark markers that
# ``add_symbol_color`` assigns when rendering the plotly scatter.
_PLOTLY_MARKERS = {
    "o": "circle", "s": "square", "D": "diamond", "d": "diamond-tall",
    ">": "triangle-right", "^": "triangle-up", "p": "pentagon", "*": "star",
    "h": "hexagon", "H": "hexagon2", "+": "cross", "x": "x",
}


def plotly_compare_runtimes(
    data1: list[dict[str, Any]],
    data2: list[dict[str, Any]],
    label1: str = "Method 1",
    label2: str = "Method 2",
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    title: str | None = None,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
    meta_func=None,
    min_value: float = 1.0,
    width: int | None = None,
    height: int = 600,
):
    """Interactive plotly twin of ``compare_runtimes``.

    Reuses ``compare_runtime_data`` for *all* data work (same benchmark pairing,
    sentinel positions and per-point colour/marker), so this function only draws.
    A log-log scatter of ``label1`` (x) vs ``label2`` (y) runtimes per benchmark:
    points above the diagonal are where ``label1`` is faster. Adds y=x / 10x /
    100x speed-up guides, timeout (∞) / out-of-memory (MO) / incorrect (✗)
    sentinel lines+ticks, and hover with the benchmark and both times. Returns a
    bare ``go.Figure`` (wrap in ``mo.ui.plotly`` for click/selection events).

    ``meta_func(d1, d2) -> list`` may attach extra per-point values to each
    marker's ``customdata`` (appended after the four hover fields), e.g. row
    identifiers so a caller can map a selected point back to its source runs.
    """
    import plotly.graph_objects as go
    import plotly.colors as pcolors

    cmp = compare_runtime_data(data1, data2, match_fields, name_func)

    max_lim = max(cmp.max_time, min_value * 10)
    sentinels = [
        (cmp.timeout, "∞"),
        (cmp.out_of_memory, "MO"),
        (cmp.incorrect, "✗"),
    ]
    upper = max([max_lim, *(s for s, _ in sentinels)]) * 2

    fig = go.Figure()

    # Diagonal speed-up guides (y = r * x): equal, then 10x and 100x both ways.
    # Clip each line so it lives inside [min_value, max_lim]^2 — otherwise the
    # guides would extend past the sentinel (∞ / MO / ✗) lines and clutter the
    # corners of the plot.
    for r, dash, gname in [
        (1.0, "solid", "equal"),
        (0.1, "dash", "10× faster"),
        (10.0, "dash", None),
        (0.01, "dot", "100× faster"),
        (100.0, "dot", None),
    ]:
        if r >= 1:
            x0, y0, x1, y1 = min_value, min_value * r, max_lim / r, max_lim
        else:
            x0, y0, x1, y1 = min_value / r, min_value, max_lim, max_lim * r
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="gray", dash=dash, width=1),
                name=gname,
                showlegend=gname is not None,
                hoverinfo="skip",
            )
        )

    # Sentinel guides: vertical = label1 failed, horizontal = label2 failed.
    for s, _ in sentinels:
        fig.add_hline(y=s, line=dict(color="lightgray", dash="dash", width=1))
        fig.add_vline(x=s, line=dict(color="lightgray", dash="dash", width=1))

    # One trace per benchmark name, each with its own colour *and* symbol so the
    # legend swatch is unambiguous (the matplotlib version's per-point colours
    # encode the seed/param variant, which made an interactive legend read as all
    # one colour). The specific variant/seed stays available on hover.
    by_name: dict = {}
    for p in cmp.points:
        by_name.setdefault(p.d1["experiment"]["name"], []).append(p)
    _palette = pcolors.qualitative.Dark24
    _name_color = {n: _palette[i % len(_palette)] for i, n in enumerate(sorted(by_name))}
    for name, pts in by_name.items():
        p0 = pts[0]
        fig.add_trace(
            go.Scatter(
                x=[max(p.x, min_value) for p in pts],
                y=[max(p.y, min_value) for p in pts],
                mode="markers",
                name=name,
                marker=dict(
                    color=_name_color[name],
                    symbol=_PLOTLY_MARKERS.get(p0.symbol, "circle"),
                    size=9,
                    line=dict(width=0.5, color="black"),
                ),
                customdata=[
                    [
                        p.label,
                        round(p.x, 2),
                        round(p.y, 2),
                        p.d1["experiment"].get("seed"),
                        *(meta_func(p.d1, p.d2) if meta_func else []),
                    ]
                    for p in pts
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    + f"{label1}: " + "%{customdata[1]} s<br>"
                    + f"{label2}: " + "%{customdata[2]} s<br>"
                    + "seed %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper", text=f"{label1} faster",
        showarrow=False, font=dict(color="gray", size=11), xanchor="left", yanchor="top",
    )
    fig.add_annotation(
        x=0.98, y=0.02, xref="paper", yref="paper", text=f"{label2} faster",
        showarrow=False, font=dict(color="gray", size=11), xanchor="right", yanchor="bottom",
    )

    # Ticks: powers of ten over the data range plus the sentinel markers.
    lo_pow, hi_pow = floor(log10(min_value)), ceil(log10(max_lim))
    tickvals = [10**i for i in range(lo_pow, hi_pow)]
    ticktext = [f"10<sup>{i}</sup>" for i in range(lo_pow, hi_pow)]
    for s, t in sentinels:
        tickvals.append(s)
        ticktext.append(t)

    axis = dict(
        type="log",
        range=[log10(min_value), log10(upper)],
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    # ``scaleanchor``/``scaleratio`` lock a unit on the x-axis to the same pixel
    # distance as a unit on the y-axis, so the speed-up diagonals stay at 45°
    # and equal runtimes always look equal — independent of the figure shape.
    fig.update_layout(
        title=title or f"{label1} vs {label2} runtime",
        xaxis=dict(title=f"{label1} (s)", **axis),
        yaxis=dict(
            title=f"{label2} (s)", scaleanchor="x", scaleratio=1, **axis
        ),
        height=height,
        width=width,
        autosize=width is None,
        legend=dict(title="benchmark"),
        template="plotly_white",
    )
    return fig


def compare_monitor_sizes(
    data1: list[dict[str, Any]],
    data2: list[dict[str, Any]],
    label1: str = "Method 1",
    label2: str = "Method 2",
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_scale: bool = True,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
    experiments_in_legends: bool = True,
    save_figures: bool = False,
    save_path: str = "./",
    file_name: str = "monitor_sizes",
    show_y_axis: bool = True,
    plot_kwargs: dict = {},
):
    pairs = pair_by_benchmark(data1, data2, match_fields)

    max_mon_states = max(
        (
            v
            for d1, d2 in pairs
            for v in [
                d1["results"]["monitor_states"] if d1["results"] is not None else 0,
                d2["results"]["monitor_states"] if d2["results"] is not None else 0,
            ]
            if isinstance(v, int)
        ),
        default=10,
    )
    max_lim = max_mon_states * 1.5

    plt.plot([0, max_lim], [0, max_lim], "k-", linewidth=1)
    plt.plot([0, max_lim * 10], [0, max_lim], "k--", label="10x smaller")
    plt.plot([0, max_lim], [0, max_lim * 10], "k--")
    plt.plot([0, max_lim * 100], [0, max_lim], "k:", label="100x smaller")
    plt.plot([0, max_lim], [0, max_lim * 100], "k:")
    plt.fill_between(
        [0, max_lim],
        [0, max_lim],
        max_lim,
        color="lightgreen",
        alpha=0.2,
        label=f"{label1} is smaller",
    )
    plt.fill_between(
        [0, max_lim],
        0,
        [0, max_lim],
        color="lightcoral",
        alpha=0.2,
        label=f"{label2} is smaller",
    )

    for d1, d2 in pairs:
        if d1["results"] is None or d2["results"] is None:
            continue
        r1, r2 = d1["results"], d2["results"]
        threshold = d1["experiment"]["threshold"]
        fp_slack = d1["experiment"]["fp_slack"]
        fn_slack = d1["experiment"]["fn_slack"]
        ms1, ms2 = r1.get("monitor_states"), r2.get("monitor_states")
        if not isinstance(ms1, int) or not isinstance(ms2, int):
            continue
        if (
            r1.get("false_positive") is None
            or r1.get("false_negative") is None
            or r1["false_positive"] < threshold - fp_slack
            or r1["false_negative"] > threshold + fn_slack
            or r2.get("false_positive") is None
            or r2.get("false_negative") is None
            or r2["false_positive"] < threshold - fp_slack
            or r2["false_negative"] > threshold + fn_slack
        ):
            continue
        plt.plot(
            ms1,
            ms2,
            d1["symbol"],
            color=d1["color"],
            label=name_func(d1, d2) if experiments_in_legends else None,
            **plot_kwargs,
        )

    plt.xlim(1, max_lim)
    plt.ylim(1, max_lim)
    plt.grid()
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(xlabel if xlabel else rf"{label1} $|\mathcal{{A}}|$ (log)")
    if not show_y_axis:
        plt.ylabel("")
        plt.yticks([10**i for i in range(0, 4)], [""] * 4)
    else:
        plt.ylabel(
            ylabel if ylabel else rf"{label2} $|\mathcal{{A}}|$ (log)",
            ha="center",
            y=0.43,
        )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


def compare_thresholds(
    data1: list[dict],
    data2: list[dict],
    colors,
    label1: str = "Method 1",
    label2: str = "Method 2",
    match_fields: list[str] = DEFAULT_MATCH_FIELDS,
    threshold: float = 0.3,
    fn_slack: float = 0.05,
    fp_slack: float = 0.2,
    title: str | None = None,
    figsize: tuple = (10, 6),
    xlabel: str | None = None,
    ylabel: str | None = None,
    name_func=lambda d1, d2: f"{d1['experiment']['name']} {d1['experiment']['variant']}",
    show_y_axis: bool = True,
):
    pairs = pair_by_benchmark(data1, data2, match_fields)

    plt.axhline(y=threshold, color="gray", linestyle="--")
    plt.axhline(y=threshold + fn_slack, color="r", linestyle="--")
    plt.axvline(x=threshold, color="gray", linestyle="--")
    plt.axvline(x=threshold - fp_slack, color="r", linestyle="--")
    plt.fill_betweenx(
        [0, threshold + fn_slack],
        threshold - fp_slack,
        1,
        color="lightgreen",
        alpha=0.3,
    )

    xmin, ymin = 1.0, 1.0
    xmax, ymax = 0.0, 0.0

    for i, (d1, d2) in enumerate(pairs):
        r1 = d1["results"] or {}
        r2 = d2["results"] or {}
        fp1 = float(r1.get("false_positive") or 0)
        fn1 = float(r1.get("false_negative") or 1)
        fp2 = float(r2.get("false_positive") or 0)
        fn2 = float(r2.get("false_negative") or 1)

        plt.plot([fp1], [fn1], d1["symbol"], color=d1["color"], label=name_func(d1, d2))
        plt.plot([fp1, fp2], [fn1, fn2], color=colors(i % colors.N))
        xmin = min(xmin, fp1, fp2)
        ymin = min(ymin, fn1, fn2)
        xmax = max(xmax, fp1, fp2)
        ymax = max(ymax, fn1, fn2)

    plt.xlabel(
        xlabel
        if xlabel
        else "False Positives threshold\n(minimal risk for trace in monitor)"
    )
    if not show_y_axis:
        plt.ylabel("")
        plt.yticks([])
    else:
        plt.ylabel(
            ylabel
            if ylabel
            else "False Negatives threshold\n(maximal risk for trace not in monitor)"
        )
    plt.legend(bbox_to_anchor=(1.05, 1.02), loc="upper left")
    if title:
        plt.title(title)
    plt.xlim(max(0, xmin * 0.95), min(1, xmax * 1.05))
    plt.ylim(max(0, ymin * 0.95), min(1, ymax * 1.05))
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    plt.show()


def compare_thresholds_bar(
    datasets: list[tuple[str, list[dict]]],
    bottom_func,
    threshold: float = 0.3,
    fn_slack: float = 0.0,
    fp_slack: float = 0.0,
    bundle: int = 1,
    fig_size: tuple = (10, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_figures: bool = False,
    save_path: str = "./",
    file_name: str = "thresholds",
    show_y_axis: bool = True,
):
    """Bar chart of FP/FN thresholds across experiments.

    Args:
        datasets: List of (label, data_list) pairs, one per method to compare.
        bottom_func: Function mapping a data entry to its x-axis label.
    """
    tab_colors = plt.get_cmap("tab20")
    fig, ax = plt.subplots()

    # Use the first dataset's list to determine experiment count
    _, first_data = datasets[0]
    n = len(first_data)

    ax.fill_betweenx(
        [0, threshold - fp_slack], -1, n / bundle + 1, color="lightgreen", alpha=0.3
    )
    ax.fill_betweenx(
        [threshold + fn_slack, 1], -1, n / bundle + 1, color="lightcoral", alpha=0.5
    )
    ax.fill_betweenx(
        [threshold - fp_slack, threshold + fn_slack],
        -1,
        n / bundle + 1,
        color="lightgrey",
        alpha=0.5,
    )
    ax.axhline(y=threshold, color="grey", linestyle="--")
    ax.axhline(y=threshold + fn_slack, color="grey", linestyle="-", linewidth=1)
    ax.axhline(y=threshold - fp_slack, color="grey", linestyle="-", linewidth=1)

    found_thresholds = [
        (
            [
                (
                    float(d["results"]["false_positive"])
                    if d["results"] is not None
                    and not isinstance(d["results"].get("false_positive"), str)
                    and d["results"].get("false_positive") is not None
                    else 1.0
                )
                for d in data
            ],
            [
                (
                    float(d["results"]["false_negative"])
                    if d["results"] is not None
                    and not isinstance(d["results"].get("false_negative"), str)
                    and d["results"].get("false_negative") is not None
                    else 0.0
                )
                for d in data
            ],
        )
        for _, data in datasets
    ]

    if bundle == 1:
        bundled = found_thresholds
        exp_names = [bottom_func(d) for d in first_data]
    else:
        bundled = []
        exp_names = []
        for fp_vals, fn_vals in found_thresholds:
            fps_b, fns_b = [], []
            for j in range(0, len(fn_vals), bundle):
                chunk_fn = fn_vals[j : j + bundle]
                max_idx = chunk_fn.index(max(chunk_fn))
                fps_b.append(fp_vals[j : j + bundle][max_idx])
                fns_b.append(chunk_fn[max_idx])
                exp_names.append(
                    "/".join(bottom_func(first_data[k]) for k in range(j, j + bundle))
                )
            bundled.append((fps_b, fns_b))

    n_datasets = len(datasets)
    bar_width = 1 / (n_datasets * 2) - 0.05
    index = range(ceil(n / bundle))

    for i, ((key_fp, key_fn), (label, _)) in enumerate(zip(bundled, datasets)):
        ax.bar(
            [j + i * 2 * bar_width for j in index],
            [-(1 - t) for t in key_fp],
            bar_width,
            bottom=1,
            label=f"{label} — in monitor",
            color=tab_colors((i * 2 + 6) % tab_colors.N),
        )
        ax.bar(
            [j + (i * 2 + 1) * bar_width for j in index],
            key_fn,
            bar_width,
            label=f"{label} — out of monitor",
            color=tab_colors((i * 2 + 4) % tab_colors.N),
        )

    if not show_y_axis:
        plt.ylabel("")
        ax.set_yticklabels([""] * len(ax.get_yticks()))
    else:
        plt.ylabel(ylabel if ylabel else "risk threshold")
    plt.xticks(
        [i + bar_width * (n_datasets - 0.5) for i in index],
        exp_names,
        rotation=90,
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    plt.xlim(-0.5, n / bundle)
    if title:
        plt.title(title)
    fig.set_size_inches(*fig_size)
    if save_figures:
        plt.savefig(f"{save_path}/{file_name}.pgf", bbox_inches="tight")
    plt.show()


def any_frac_to_float(x):
    if isinstance(x, Fraction):
        return float(x)
    return x


def compare_runtime_by_params(
    exp_data: list[dict],
    param_keys: list[str],
    time_key: str = "time",
    figsize: tuple = (20, 10),
    fit_all: bool = True,
    title: str | None = None,
    plot_kwargs: dict = {},
):
    """Compare runtime across different parameter settings.

    Creates a scatter subplot for every pair of param_keys value combinations,
    showing how runtime changes between two parameter configurations.
    """
    param_values = {k: set() for k in param_keys}
    param_groups: dict = {}

    for data in exp_data:
        if data["results"] is None:
            continue
        group_key = tuple(
            (k, str(any_frac_to_float(v)))
            for k, v in data["experiment"].items()
            if k not in param_keys
            and k not in ["variant", "result_json_file", "short_name", "variant_hash"]
        ) + (
            (lambda m: m.group(0) if m else None)(
                re.compile(r"intermediate_monitor=(\d+\.\d+)").search(
                    data["experiment"]["variant"]
                )
            ),
        )
        if group_key not in param_groups:
            param_groups[group_key] = {}

        for k in param_keys:
            param_values[k].add(data["experiment"][k])

        param_value = tuple(data["experiment"][k] for k in param_keys)
        if param_value in param_groups[group_key]:
            raise ValueError(
                f"Duplicate data for parameter setting {param_value} in group {group_key}"
            )
        param_groups[group_key][param_value] = data

    for group_key, group_data in param_groups.items():
        if len(group_data) < 4:
            print(f"Missing data for {group_key}: only {len(group_data)} combinations.")

    param_combinations = list(product(*[sorted(param_values[k]) for k in param_keys]))
    param_param_combinations = list(combinations(param_combinations, 2))
    num_plots = len(param_param_combinations)

    fig, axes = plt.subplots(nrows=ceil(num_plots / 3), ncols=3, figsize=figsize)
    fig.suptitle(
        title
        or f"Runtime comparison by parameter combinations: {', '.join(param_keys)}",
        fontsize=12,
    )
    axes = axes.flatten()
    plot_idx = 0

    max_time, timeout, out_of_memory, incorrect, unfinished = calculate_error_lines(
        exp_data,
        lambda d: d["results"][time_key] if d["results"] is not None else 0,
    )

    for params1, params2 in param_param_combinations:
        ax = axes[plot_idx]
        plot_idx += 1
        max_x, max_y = 0.0, 0.0

        for group_key, group_data in param_groups.items():
            time1: float | None = None
            time2: float | None = None
            colors = []
            hashes = [None, None]

            for param in (params1, params2):
                if param in group_data:
                    data = group_data[param]
                    if data["results"] is None:
                        tv = (
                            timeout if data.get("error") == "timeout" else out_of_memory
                        )
                    else:
                        tv = data["results"].get(time_key)
                        if tv is None:
                            continue
                        if isinstance(tv, str) and "/" in tv:
                            tv = float(Fraction(tv))
                        fp = data["results"].get("false_positive")
                        fn = data["results"].get("false_negative")
                        thresh = data["experiment"]["threshold"]
                        if (
                            fp is None
                            or fn is None
                            or fp < thresh - data["experiment"]["fp_slack"]
                            or fn > thresh + data["experiment"]["fn_slack"]
                        ):
                            tv = incorrect

                    if param == params1:
                        time1 = tv
                        hashes[0] = data["experiment"].get("variant_hash")
                    else:
                        time2 = tv
                        hashes[1] = data["experiment"].get("variant_hash")
                    colors.append(data["color"])
                elif param == params1:
                    time1 = unfinished
                else:
                    time2 = unfinished

            if time1 is not None and time2 is not None:
                rel_diff = abs(time1 - time2) / max(time1, time2)
                if rel_diff >= 0.9 and max(time1, time2) < timeout:
                    print(
                        f"Large relative difference ({rel_diff:.4f}: {time1} vs {time2}) "
                        f"for params {params1} vs {params2} in hashes {hashes}"
                    )

            max_x = max(max_x, cast(float, time1) if time1 is not None else 0)
            max_y = max(max_y, cast(float, time2) if time2 is not None else 0)

            if not colors:
                continue
            ax.scatter(
                time1,
                time2,
                marker=data["symbol"],
                color="None",
                edgecolor=sorted(colors)[0],
                label=f"{data['experiment']['name']} {data['experiment']['variant']}",
                **plot_kwargs,
            )

        ax.plot([0, timeout], [0, timeout], r"-", color="0.5")
        ax.plot([0, timeout], [0, timeout / 10], "--", color="0.5", label="10x slower")
        ax.plot([0, timeout], [0, timeout / 100], ":", color="0.5", label="100x slower")
        ax.plot([0, timeout / 10], [0, timeout], "--", color="0.5")
        ax.plot([0, timeout / 100], [0, timeout], ":", color="0.5")
        for sentinel, label in [
            (timeout, "timeout"),
            (out_of_memory, "OOM"),
            (incorrect, "incorrect"),
            (unfinished, "unfinished"),
        ]:
            ax.axline(
                (0, sentinel),
                (sentinel, sentinel),
                color="gray",
                linestyle="--",
                label=f"Param 2 {label}",
            )
            ax.axline(
                (sentinel, 0),
                (sentinel, sentinel),
                color="gray",
                linestyle="--",
                label=f"Param 1 {label}",
            )

        ax.loglog()
        ticks = [10**i for i in range(-1, int(log10(max_time)) + 1)] + [
            timeout,
            out_of_memory,
            incorrect,
            unfinished,
        ]
        tick_labels = [f"$10^{{{i}}}$" for i in range(-1, int(log10(max_time)) + 1)] + [
            r"$\infty$",
            "ERR",
            r"$\times$",
            "U",
        ]
        ax.set_yticks(ticks, tick_labels)
        ax.set_xticks(ticks, tick_labels)
        ax.set_xlabel(", ".join(f"{v}" for k, v in zip(param_keys, params1)))
        ax.set_ylabel(", ".join(f"{v}" for k, v in zip(param_keys, params2)))
        ax.grid(True, which="major", ls="--")
        if fit_all:
            ax.set_xlim(0.1, unfinished * 2)
            ax.set_ylim(0.1, unfinished * 2)
        else:
            ax.set_xlim(0.1, max_x * 1.2)
            ax.set_ylim(0.1, max_y * 1.2)

    return fig, axes


def runtime_by_params(
    exp_data: list[dict],
    params: list[tuple[tuple[str, str] | list[tuple[str, str]], str]],
    time_key: str = "time",
    title: str | None = None,
    figsize: tuple = (10, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    fit_line: bool = False,
    name_func=lambda d: f"{d['experiment']['name']} {d['experiment']['variant']}",
    experiments_in_legends: bool = True,
    show_y_axis: bool = True,
    plot_kwargs: dict = {},
):
    fig, axes = plt.subplots(nrows=ceil(len(params) / 2), ncols=2, figsize=figsize)
    axes = axes.flatten()

    for i, (select_keys, type_plot) in enumerate(params):
        ax = axes[i]

        if fit_line:
            symbol_points: dict = {}
            for d in exp_data:
                if d["results"] is None:
                    continue
                val = d[select_keys[0]][select_keys[1]]
                t = d["results"][time_key]
                if (
                    t is None
                    or val is None
                    or isinstance(val, str)
                    or isinstance(t, str)
                ):
                    continue
                symbol_points.setdefault(d["symbol"], []).append((val, t))

            for symbol, points in symbol_points.items():
                xs, ys = zip(*points)
                x_line = np.linspace(min(xs), max(xs), 100)

                def exp_func(x, a, b):
                    return a * np.exp(b * x)

                try:
                    popt, _ = curve_fit(exp_func, xs, ys, p0=(1, 0.1))
                    ax.plot(
                        x_line,
                        exp_func(x_line, *popt),
                        "--",
                        marker=symbol,
                        markevery=0.1,
                        color="gray",
                        label=f"Exp fit: {popt[0]:.2f}*e^({popt[1]:.4f}x)",
                    )
                except RuntimeError:
                    pass

        if type_plot == "box":
            groups: dict = {}
            for d in exp_data:
                if d["results"] is None:
                    continue
                if isinstance(select_keys, list):
                    val = "\n".join(str(d[k][sk]) for k, sk in select_keys)
                else:
                    val = d[select_keys[0]][select_keys[1]]
                val = "None" if val is None else val
                t = d["results"][time_key]
                if t is not None:
                    groups.setdefault(str(val), []).append(t)
            labels, data_list = zip(*groups.items())
            ax.boxplot(data_list, labels=labels, showmeans=True, showfliers=False)
        else:
            for data in exp_data:
                if data["results"] is None:
                    continue
                if isinstance(select_keys, list):
                    val = str(tuple(data[k][sk] for k, sk in select_keys))
                else:
                    val = data[select_keys[0]][select_keys[1]]
                ax.plot(
                    val,
                    data["results"][time_key],
                    data["symbol"],
                    color=data["color"],
                    label=name_func(data),
                    **plot_kwargs,
                )
            if type_plot == "log":
                ax.set_yscale("log")

        if isinstance(select_keys, list):
            ax.set_xlabel(
                xlabel
                or ", ".join(sk.replace("_", " ").capitalize() for _, sk in select_keys)
            )
        else:
            ax.set_xlabel(xlabel or select_keys[1].replace("_", " ").capitalize())

        if not show_y_axis:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            ax.set_ylabel(ylabel or f"Run Time (s log)")
        ax.set_title(
            title or f"{type_plot.replace('_', ' ').capitalize()} run {time_key}"
        )
        ax.grid(True, which="both", ls="--")

    handles, labels = ax.get_legend_handles_labels()
    if experiments_in_legends:
        fig.legend(handles, labels, bbox_to_anchor=(1, 0.9), loc="upper left")
    plt.tight_layout()
    plt.show()


def runtime_from_logs(logpath: str):
    entries: dict[str, float] = {}
    example_msg: dict[str, str] = {}
    time_pattern = re.compile(r"\((\d+(?:\.\d+))s\)")

    with open(logpath, "r") as f:
        for line in f:
            if "(s)" in line:
                continue
            match = time_pattern.search(line)
            if match:
                try:
                    elapsed = float(match.group(1))
                    _, _, loc, msg = [s.strip() for s in line.split(" - ", 4)]
                    entries[loc] = entries.get(loc, 0) + elapsed
                    example_msg.setdefault(loc, msg)
                except ValueError:
                    pass
    return entries, example_msg
