"""Shared helpers for the runtime-breakdown analysis notebooks.

Both ``notebooks/Analysis.ipynb`` (comparing learning configurations) and
``notebooks/compare_conditional_methods.ipynb`` (comparing conditional-probability
methods) group experiment entries into a ``groups`` mapping ``label -> entries``
and then draw the same per-group x per-benchmark breakdown matrix / summaries.
Keeping that logic here means the notebooks only differ in how they build
``groups``.
"""

from __future__ import annotations

import os
import re
from collections import Counter, namedtuple
from fractions import Fraction

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tover.analysis.plots import pair_by_benchmark
from tover.utils.helpers import str_to_float

# Experiments store their runtime split across several fields. Two layouts
# exist in the data:
#   new format -> results["product_time"/"paynt_time"/"eq_time"/
#                 "reference_language_time"] and
#                 results["learning_stats"]["learning_time"/"smt_time"]
#   old format -> results["lstar_time"/"counterexample_time"/...]
# We map both onto a common set of stacked components below. "smt", "reference"
# and "counterexample" only appear for some variants (0 elsewhere); "other"
# soaks up any remaining unattributed time so the components sum to results["time"].
TIME_COMPONENTS = [
    ("learning", "tab:blue"),  # pure L* / L# learning (learning_time or lstar_time)
    ("smt", "tab:purple"),  # SMT solving (apartness), some L# variants only
    ("reference", "tab:cyan"),  # reference-language DFA construction (rl variants)
    ("product", "tab:orange"),  # building the MC x monitor product MDP
    ("paynt", "tab:green"),  # PAYNT synthesis / verification
    ("eq", "tab:red"),  # equivalence-oracle model checking
    ("counterexample", "tab:brown"),  # counterexample extraction (old format)
    ("other", "lightgray"),  # unattributed remainder (= time - sum of above)
]

# Fields identifying the same benchmark across groups (so they share a column).
MATCH_FIELDS = [
    "name",
    "file",
    "parameters",
    "horizon",
    "threshold",
    "fp_slack",
    "fn_slack",
    "spec",
    "good_label",
]
# These match fields are numeric but stored inconsistently: runs that executed
# get an exact Fraction string (e.g. "3/10") because run() sharpens them, while
# not-started placeholder entries keep the original float (0.3). Canonicalise
# them to a rounded float so the two representations identify the same benchmark.
NUMERIC_MATCH_FIELDS = {"threshold", "fp_slack", "fn_slack"}


def _canon_field(field, value):
    if value is None:
        return None
    if field in NUMERIC_MATCH_FIELDS:
        try:
            return round(str_to_float(str(value)), 9)
        except (ValueError, ZeroDivisionError):
            pass
    return str(value)


def time_components(d):
    """Map an experiment onto the stacked TIME_COMPONENTS (seconds).

    Returns a dict component-name -> seconds, or None if the run has no results.
    """
    r = d["results"]
    if r is None:
        return None
    ls = r.get("learning_stats") or {}

    comp = {
        "learning": (ls.get("learning_time") or r.get("lstar_time")) or 0.0,
        "smt": ls.get("smt_time") or 0.0,
        "reference": r.get("reference_language_time") or 0.0,
        "product": r.get("product_time") or 0.0,
        "paynt": r.get("paynt_time") or 0.0,
        "eq": r.get("eq_time") or 0.0,
        "counterexample": r.get("counterexample_time") or 0.0,
    }
    total = r.get("time")
    if sum(comp.values()) > total + 0.1:
        raise ValueError(
            f"components sum to {sum(comp.values())} > total {total} in {d.get('json_path')}"
        )
    comp["other"] = max(0.0, total - sum(comp.values())) if total is not None else 0.0
    return comp


def runtime(d):
    """Total runtime (seconds) of an experiment, or None if it has no results."""
    r = d["results"]
    if r is None:
        if d["error"] is not None:
            return float("nan")
        return float("inf")
    return r["time"]


def bench_key(d):
    e = d["experiment"]
    return tuple(_canon_field(f, e.get(f)) for f in MATCH_FIELDS)


def bench_label(d):
    e = d["experiment"]
    extra = e.get("file") or (e.get("parameters") or {}).get("constants", "")
    extra = str(extra).split("/")[-1]
    return f"{e.get('name')}\n{extra}\nh={e.get('horizon')}"


def bench_row(d):
    """The ``method_table`` row index for an entry: ``(name, file, horizon)``."""
    e = d["experiment"]
    return (e.get("name"), str(e.get("file") or "").split("/")[-1], e.get("horizon"))


# A metric aggregated over the seed-runs of one (group, benchmark) cell. The same
# benchmark run under several `seed`s shares a bench_key (seed is not a MATCH_FIELD),
# so those runs collapse into one cell; this records their mean and (sample) std,
# the number of seeds that produced a value (n) and the total seed-runs (n_total),
# so partial failures stay visible.
Agg = namedtuple("Agg", ["mean", "std", "n", "n_total"])


def entries_by_bench(ds):
    """Group a single group's entries by benchmark key -> list of its seed-runs."""
    out: dict = {}
    for d in ds:
        out.setdefault(bench_key(d), []).append(d)
    return out


def _aggregate(values, n_total):
    """Aggregate seed values (Nones dropped) into an Agg, or None if all missing."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return Agg(float(np.mean(vals)), std, len(vals), n_total)


def _summarize_reason(entries):
    """Most common failure reason across a cell's seed-runs (or None)."""
    reasons = [r for d in entries if (r := failure_reason(d)) is not None]
    return Counter(reasons).most_common(1)[0][0] if reasons else None


def _group_components(groups):
    """For each group, map benchmark key -> mean components across its seed-runs."""
    out = {}
    for label, ds in groups.items():
        by_key = {}
        for k, entries in entries_by_bench(ds).items():
            comps = [c for d in entries if (c := time_components(d)) is not None]
            if comps:
                by_key[k] = {
                    cn: float(np.mean([c[cn] for c in comps]))
                    for cn, _ in TIME_COMPONENTS
                }
        out[label] = by_key
    return out


def _group_component_std(groups):
    """For each group, map benchmark key -> per-phase std across seeds.

    Returns ``{label: {bench_key: {component: std}}}``. Only cells with at least
    two finished seeds get an entry; used to draw a ±std whisker on each phase
    block in ``plot_runtime_breakdown``.
    """
    out = {}
    for label, ds in groups.items():
        by_key = {}
        for k, entries in entries_by_bench(ds).items():
            comps = [c for d in entries if (c := time_components(d)) is not None]
            if len(comps) >= 2:
                by_key[k] = {
                    cn: float(np.std([c[cn] for c in comps], ddof=1))
                    for cn, _ in TIME_COMPONENTS
                }
        out[label] = by_key
    return out


def plot_runtime_breakdown(groups, title=None, ncols=4, sharey=False):
    """Faceted stacked runtime-breakdown bars: one subplot per benchmark.

    ``groups`` is an ordered mapping ``label -> list of experiment entries``. Each
    benchmark (model) gets its own subplot; inside it there is one stacked bar per
    group (in ``groups`` order), broken into the ``TIME_COMPONENTS`` phases in
    seconds. Because every subplot covers a single benchmark, the y-axis is linear
    and auto-scaled to that benchmark's runtime, so even small phases stay readable
    without the log axis the combined plot needed — at the cost of not comparing
    totals across benchmarks (which is not meaningful anyway).

    ``ncols`` controls the subplot grid width; ``sharey`` ties all subplots to a
    common y-axis (off by default, since the whole point is per-benchmark scaling).
    Groups keep a fixed x-position across every subplot so a method sits in the
    same column everywhere; a method with no result for a benchmark gets a muted
    ``×`` instead of a bar.

    When a benchmark ran under several seeds, each phase block carries a small ±std
    (across seeds) whisker at the block's top edge (where its segment ends).
    Deterministic phases (std 0) get none, and the whiskers within a bar are dodged
    horizontally so they stay readable; their height identifies which (coloured) block.
    """
    nonempty = {g: ds for g, ds in groups.items() if ds}
    group_components = _group_components(nonempty)
    group_components = {g: bk for g, bk in group_components.items() if bk}
    group_names = list(group_components)
    n_groups = len(group_names)
    component_std = _group_component_std(nonempty)  # per-phase ±std across seeds

    col_label = {}
    for ds in nonempty.values():
        for d in ds:
            if time_components(d) is not None:
                col_label.setdefault(bench_key(d), bench_label(d))
    bench_keys = sorted(col_label, key=lambda k: col_label[k])
    n_bench = len(bench_keys)
    if n_bench == 0:
        raise ValueError("no benchmarks with runtime data to plot")

    used = {
        cn
        for bk in group_components.values()
        for c in bk.values()
        for cn, _ in TIME_COMPONENTS
        if c[cn] > 0
    }

    ncols = min(ncols, n_bench)
    nrows = -(-n_bench // ncols)  # ceil
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * (1.1 + n_groups * 0.55), nrows * 3.4),
        sharey=sharey,
        squeeze=False,
    )
    xs = np.arange(n_groups)

    for bi, k in enumerate(bench_keys):
        ax = axes[bi // ncols][bi % ncols]
        bottom = np.zeros(n_groups)
        # Record each drawn segment's top (its cumulative boundary) so we can hang a
        # ±std whisker there after stacking: seg_top[gi] -> [(cname, top), ...].
        seg_top: dict[int, list] = {gi: [] for gi in range(n_groups)}
        for cname, color in TIME_COMPONENTS:
            if cname not in used:
                continue
            vals = np.array(
                [group_components[g].get(k, {}).get(cname, 0.0) for g in group_names]
            )
            ax.bar(xs, vals, bottom=bottom, width=0.74, color=color)
            bottom += vals
            for gi in range(n_groups):
                if vals[gi] > 0:
                    seg_top[gi].append((cname, bottom[gi]))
        # A ±std (across seeds) whisker on each phase block, at the block's top edge
        # (where its segment ends). Only blocks with a non-zero std get one (most
        # phases are deterministic), and within a bar the whiskers are dodged
        # horizontally so they never overlap; their height ties them to their colour.
        for gi, g in enumerate(group_names):
            stds = component_std.get(g, {}).get(k, {})
            whiskers = [(cn, top, stds[cn]) for cn, top in seg_top[gi] if stds.get(cn)]
            if not whiskers:
                continue
            offs = np.linspace(-0.26, 0.26, len(whiskers)) if len(whiskers) > 1 else [0]
            for (cn, top, sd), off in zip(whiskers, offs):
                ax.errorbar(
                    gi + off,
                    top,
                    yerr=sd,
                    fmt=".",
                    markersize=2.5,
                    color="0.12",
                    ecolor="0.12",
                    elinewidth=0.7,
                    capsize=1.5,
                    zorder=3,
                )
        # Total on top of each present bar; muted × where the method has no result.
        head = (bottom.max() or 1.0) * 0.02
        for gi, g in enumerate(group_names):
            if k in group_components[g]:
                ax.text(
                    gi,
                    bottom[gi] + head,
                    f"{bottom[gi]:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="0.3",
                )
            else:
                ax.text(
                    gi, head, "×", ha="center", va="bottom", fontsize=10, color="0.6"
                )
        ax.set_title(col_label[k], fontsize=8, fontweight="bold", linespacing=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(group_names, fontsize=7, rotation=30, ha="right")
        ax.set_xlim(-0.6, n_groups - 0.4)
        ax.margins(y=0.12)
        if bi % ncols == 0:
            ax.set_ylabel("runtime (s)")

    for j in range(n_bench, nrows * ncols):  # hide unused cells
        axes[j // ncols][j % ncols].axis("off")

    handles = [Patch(color=c, label=n) for n, c in TIME_COMPONENTS if n in used]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        fontsize=8,
    )
    fig.suptitle(
        title or "Runtime breakdown — one subplot per benchmark, bars: group",
        y=1.04,
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


# Per-round breakdown reuses the same phase colours as TIME_COMPONENTS, but the
# components are sourced per learning round from the fine-grained timing:
#   results["learning_stats"]["round_timings"][i] -> build/smt/process/eq wall time
#   results["rounds"][i]                           -> FN/FP product & paynt split
# (the two lists are aligned by round). "eq" holds the equivalence-oracle wall
# time not attributable to product/paynt (sampling + model checking overhead).
ROUND_COMPONENTS = [
    ("learning", "tab:blue"),  # hypothesis construction (build_time)
    ("smt", "tab:purple"),  # SMT solving during build/process
    ("product", "tab:orange"),  # MC x monitor product (FN + FP)
    ("paynt", "tab:green"),  # PAYNT synthesis (FN + FP)
    ("eq", "tab:red"),  # eq-oracle: sampling + model-checking overhead
    ("counterexample", "tab:brown"),  # counterexample processing (process_time)
]


def round_components(d):
    """Per-learning-round runtime breakdown for a single experiment.

    Merges the learning-loop timings (``learning_stats.round_timings``) with the
    equivalence-oracle's per-round FN/FP split (``results.rounds``), both aligned
    by round, onto the ROUND_COMPONENTS phases. Returns a list (one dict of
    component -> seconds per round, plus ``_size`` / ``_found`` metadata) or
    ``None`` if the run carries no per-round timing.
    """
    r = d["results"]
    if r is None:
        return None
    ls = r.get("learning_stats") or {}
    timings = ls.get("round_timings") or []
    oracle = r.get("rounds") or []
    n = max(len(timings), len(oracle))
    if n == 0:
        return None

    rounds = []
    for i in range(n):
        t = timings[i] if i < len(timings) else {}
        o = oracle[i] if i < len(oracle) else {}
        product = (o.get("fn_product_time") or 0.0) + (o.get("fp_product_time") or 0.0)
        paynt = (o.get("fn_paynt_time") or 0.0) + (o.get("fp_paynt_time") or 0.0)
        sampling = o.get("sampling_time") or 0.0
        eq_total = t.get("eq_time")
        # The eq-oracle wall time (eq_total) contains product + paynt + sampling
        # plus model-checking overhead; show the remainder under "eq" so the bar
        # sums to the round's true wall time. When eq_total is absent (no
        # round_timings, e.g. L* / plain L#), fall back to the sampling time.
        if eq_total is None:
            eq = sampling
        else:
            eq = max(0.0, eq_total - product - paynt)
        rounds.append(
            {
                "learning": t.get("build_time") or 0.0,
                "smt": t.get("smt_time") or 0.0,
                "product": product,
                "paynt": paynt,
                "eq": eq,
                "counterexample": t.get("process_time") or 0.0,
                "_size": o.get("hypothesis_size") or t.get("hypothesis_size"),
                "_found": o.get("found"),
            }
        )
    return rounds


def _resolve_round_range(n_total, round_range):
    """Resolve a ``round_range`` into the list of selected 0-based round indices.

    ``round_range`` is either ``None`` (all ``n_total`` rounds) or a ``(start,
    stop)`` tuple with half-open, 0-based semantics like a Python slice: either
    bound may be ``None`` (open end), negative bounds count from the end, and
    out-of-range bounds are clamped. Lets the per-round plots zoom into a window of
    a long run. Raises if the resulting window is empty.
    """
    if round_range is None:
        return list(range(n_total))
    start, stop = round_range
    start = 0 if start is None else (start + n_total if start < 0 else start)
    stop = n_total if stop is None else (stop + n_total if stop < 0 else stop)
    sel = list(range(max(0, start), min(n_total, stop)))
    if not sel:
        raise ValueError(
            f"round_range {round_range!r} selects no rounds (run has {n_total})"
        )
    return sel


def plot_round_breakdown(d, relative=False, round_range=None, title=None, figsize=None):
    """Stacked per-round runtime-breakdown bars for a single experiment.

    One bar per learning round, stacked into the ROUND_COMPONENTS phases (seconds,
    or fraction of the round's runtime when ``relative``). Each bar is annotated
    with the hypothesis size ``|H|`` it was built from and which check found the
    counterexample (``fn`` / ``fp`` / ``samp``; ``✓`` for the final round that
    found none). ``d`` is a single experiment entry (as produced by
    ``load_experiment_data``). ``round_range=(start, stop)`` restricts the plot to
    a slice of the rounds (0-based, half-open, either bound optional) — useful for
    zooming into a long run; the x-axis keeps the original round numbers.
    """
    rounds_all = round_components(d)
    if rounds_all is None:
        raise ValueError(
            f"{d.get('json_path')}: run has no per-round timing data "
            "(needs the L#box learning path with round_timings / rounds)"
        )
    sel = _resolve_round_range(len(rounds_all), round_range)
    rounds = [rounds_all[i] for i in sel]

    n = len(rounds)
    used = {cn for r in rounds for cn, _ in ROUND_COMPONENTS if r[cn] > 0}
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize or (max(7, n * 0.55), 5))

    totals = np.array(
        [sum(r[cn] for cn, _ in ROUND_COMPONENTS) for r in rounds], dtype=float
    )
    bottom = np.zeros(n)
    for cname, color in ROUND_COMPONENTS:
        if cname not in used:
            continue
        vals = np.array([r[cname] for r in rounds], dtype=float)
        if relative:
            vals = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0)
        ax.bar(xs, vals, bottom=bottom, width=0.82, color=color, label=cname)
        bottom += vals

    # Annotate each bar with which check found the cex and the hypothesis size.
    found_label = {"fn": "fn", "fp": "fp", "sampling": "samp", None: "✓"}
    tops = bottom
    head = (tops.max() if len(tops) else 1.0) * 0.02
    for i, r in enumerate(rounds):
        ax.text(
            i,
            tops[i] + head,
            found_label.get(r["_found"], str(r["_found"])),
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.3",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            f"{sel[i] + 1}\n|H|={r['_size']}"
            if r["_size"] is not None
            else f"{sel[i] + 1}"
            for i, r in enumerate(rounds)
        ],
        fontsize=8,
        linespacing=0.9,
    )
    ax.set_xlabel("learning round")
    ax.set_xlim(-0.5, n - 0.5)
    if relative:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("fraction of round runtime")
    else:
        ax.set_ylabel("runtime (s)")
    ax.margins(y=0.08)

    handles = [Patch(color=c, label=n) for n, c in ROUND_COMPONENTS if n in used]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=len(handles),
        fontsize=8,
    )
    ax.set_title(
        title
        or f"Per-round runtime breakdown — {bench_label(d).replace(chr(10), ' ')}",
        pad=32,
    )
    fig.tight_layout()
    plt.show()


# Hatch patterns distinguishing the runs in the grouped comparison plots (one per
# run, cycled). The phase colours stay shared (ROUND_COMPONENTS), so the hatch is
# the only per-run channel.
_COMPARE_HATCHES = ["", "//", "..", "xx", "\\\\", "++"]


def plot_round_breakdown_compare(
    runs, relative=False, round_range=None, title=None, figsize=None
):
    """Grouped per-round runtime breakdown for several runs side by side.

    Like ``plot_round_breakdown`` but overlays several runs (e.g. two methods on
    the same benchmark) for direct comparison: ``runs`` maps a label -> experiment
    entry. For every learning round, each run gets its own stacked bar (the
    ROUND_COMPONENTS phases, seconds or fraction when ``relative``), the bars
    dodged side by side so the phase composition can be read round-by-round. The
    phase *colours* are shared across runs; the per-run *hatch* (see legend)
    tells the bars apart. Each bar is annotated with which check found the cex
    (``fn``/``fp``/``samp``/``✓``). Runs may have different round counts; the
    x-axis spans the longest, labelled with the round number. ``round_range=(start,
    stop)`` restricts the plot to a slice of the rounds (0-based, half-open, either
    bound optional, applied by the shared round index) — useful for zooming into a
    long run.
    """
    parsed = {}
    for label, run in runs.items():
        rounds = round_components(run)
        if rounds is None:
            raise ValueError(
                f"{label!r}: run has no per-round timing data "
                "(needs the L#box learning path with round_timings / rounds)"
            )
        parsed[label] = rounds

    labels = list(parsed)
    n_runs = len(labels)
    # Selected (global, 0-based) round indices, shared across runs so bar i of each
    # run is the same round; a run shorter than the window just contributes fewer.
    sel = _resolve_round_range(max(len(r) for r in parsed.values()), round_range)
    n = len(sel)
    used = {
        cn
        for rounds in parsed.values()
        for gi in sel
        if gi < len(rounds)
        for cn, _ in ROUND_COMPONENTS
        if rounds[gi][cn] > 0
    }
    xs = np.arange(n)
    width = 0.82 / n_runs
    fig, ax = plt.subplots(figsize=figsize or (max(8, n * n_runs * 0.42), 5))

    found_label = {"fn": "fn", "fp": "fp", "sampling": "samp", None: "✓"}
    for ri, label in enumerate(labels):
        rounds_all = parsed[label]
        # Positions (within the window) and the rounds this run actually reaches.
        present = [(pos, gi) for pos, gi in enumerate(sel) if gi < len(rounds_all)]
        if not present:
            continue
        positions = np.array([pos for pos, _ in present])
        rounds = [rounds_all[gi] for _, gi in present]
        offset = (ri - (n_runs - 1) / 2) * width
        xpos = positions + offset
        hatch = _COMPARE_HATCHES[ri % len(_COMPARE_HATCHES)]
        totals = np.array(
            [sum(r[cn] for cn, _ in ROUND_COMPONENTS) for r in rounds], dtype=float
        )
        bottom = np.zeros(len(rounds))
        for cname, color in ROUND_COMPONENTS:
            if cname not in used:
                continue
            vals = np.array([r[cname] for r in rounds], dtype=float)
            if relative:
                vals = np.divide(
                    vals, totals, out=np.zeros_like(vals), where=totals > 0
                )
            ax.bar(
                xpos,
                vals,
                bottom=bottom,
                width=width * 0.92,
                color=color,
                hatch=hatch,
                edgecolor="0.25",
                linewidth=0.4,
            )
            bottom += vals
        head = (bottom.max() if len(bottom) else 1.0) * 0.02
        for i, r in enumerate(rounds):
            ax.text(
                xpos[i],
                bottom[i] + head,
                found_label.get(r["_found"], str(r["_found"])),
                ha="center",
                va="bottom",
                fontsize=6,
                color="0.3",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([str(gi + 1) for gi in sel], fontsize=8)
    ax.set_xlabel("learning round")
    ax.set_xlim(-0.5, n - 0.5)
    if relative:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("fraction of round runtime")
    else:
        ax.set_ylabel("runtime (s)")
    ax.margins(y=0.08)

    # Two legends: colour = phase, hatch = run (method).
    phase_handles = [Patch(color=c, label=n) for n, c in ROUND_COMPONENTS if n in used]
    run_handles = [
        Patch(
            facecolor="white",
            edgecolor="0.25",
            hatch=_COMPARE_HATCHES[ri % len(_COMPARE_HATCHES)],
            label=label,
        )
        for ri, label in enumerate(labels)
    ]
    leg1 = ax.legend(
        handles=phase_handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=len(phase_handles),
        fontsize=8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=run_handles,
        loc="lower right",
        bbox_to_anchor=(1, 1.02),
        ncol=len(run_handles),
        fontsize=8,
        title="run",
    )
    ax.set_title(title or "Per-round runtime breakdown — run comparison", pad=32)
    fig.tight_layout()
    plt.show()


# Each PAYNT synthesis call logs one line via the project logger, ending in
#   <family_size>;<avg_mdp_size>;<synthesis_time_s>;<mdp_iterations>
# (see PAYNT's SynthesizerAR). The fields are matched at the end of the log line.
_PAYNT_LINE = re.compile(r"([\d.eE+]+);([\d.]+);([\d.]+);(\d+)\s*$")

# The PAYNT per-call series, with the y-axis label and whether to use a log scale.
PAYNT_METRICS = [
    ("synthesis_time", "PAYNT time (s)", False),
    ("iterations", "MDP iterations", True),
    ("iters_per_sec", "MDP it/s", False),
    ("avg_mdp_size", "avg MDP size", False),
    ("family_size", "family size", True),
]


def parse_paynt_progress(log_path):
    """Parse a run log into the per-round PAYNT synthesis stats.

    Each equivalence-oracle round logs ``Finding counterexample for hypothesis
    with N states``; inside it PAYNT prints one stats line per synthesis call —
    for the false-negative check and, if it is reached, the false-positive check.
    Returns a list of dicts (one per PAYNT call) with the ``round`` index, the
    hypothesis size ``hyp_size``, the ``phase`` (``"fn"``/``"fp"``), the four
    values ``family_size``/``avg_mdp_size``/``synthesis_time``/``iterations`` and
    the derived ``iters_per_sec`` (``None`` when the time rounds to 0). A round
    that runs both checks yields two entries with the same ``round``. PAYNT calls
    from the final post-learning verification are excluded.
    """
    out = []
    round_idx, hyp_size, phase = -1, None, None
    with open(log_path) as f:
        for line in f:
            if "Finding counterexample for hypothesis" in line:
                m = re.search(r"hypothesis with (\d+) states", line)
                round_idx += 1
                hyp_size = int(m.group(1)) if m else None
                phase = None
                continue
            if "Finding false negative probability" in line:
                phase = "fn"
                continue
            if "Finding false positive probability" in line:
                phase = "fp"
                continue
            if "Verifying the ToVer learned monitor" in line:
                break  # subsequent PAYNT lines are the final verification, not a round
            m = _PAYNT_LINE.search(line)
            if m and round_idx >= 0:
                synthesis_time, iterations = float(m.group(3)), int(m.group(4))
                out.append(
                    {
                        "round": round_idx,
                        "hyp_size": hyp_size,
                        "phase": phase or "?",
                        "family_size": float(m.group(1)),
                        "avg_mdp_size": float(m.group(2)),
                        "synthesis_time": synthesis_time,
                        "iterations": iterations,
                        # iterations per second; None when the time rounds to 0.
                        "iters_per_sec": (
                            iterations / synthesis_time if synthesis_time > 0 else None
                        ),
                    }
                )
    return out


def plot_paynt_progress(run, round_range=None, title=None, figsize=None):
    """Plot the PAYNT synthesis stats over learning rounds for a single run.

    ``run`` is an experiment entry (uses its ``log_path``) or a log path string.
    One panel per ``PAYNT_METRICS`` series (PAYNT time, MDP iterations, average
    MDP size, family size), all sharing the round x-axis, so you can see which
    rounds PAYNT is expensive and what drives it. The false-negative and
    false-positive calls are drawn as separate series (a round may have both, so
    two points land on the same round); the x-axis is labelled with the round and
    its hypothesis size ``|H|``. Rounds resolved by the sampling oracle (no PAYNT
    call) simply have no point, leaving a visible gap. ``round_range=(start,
    stop)`` restricts the plot to a slice of the rounds (0-based, half-open, either
    bound optional) — useful for zooming into a long run.
    """
    log_path = run if isinstance(run, str) else run.get("log_path")
    if not log_path or not os.path.exists(log_path):
        raise ValueError(f"no readable log for this run: {log_path!r}")
    rows = parse_paynt_progress(log_path)
    if not rows:
        raise ValueError(f"no PAYNT progress lines found in {log_path}")
    keep = set(_resolve_round_range(max(r["round"] for r in rows) + 1, round_range))
    rows = [r for r in rows if r["round"] in keep]

    phase_style = {
        "fn": ("tab:blue", "o", "false negative"),
        "fp": ("tab:red", "s", "false positive"),
        "?": ("0.5", "x", "unknown"),
    }
    present = [p for p in phase_style if any(r["phase"] == p for r in rows)]
    hyp_by_round = {r["round"]: r["hyp_size"] for r in rows}
    rounds = sorted(hyp_by_round)
    n_rounds = rounds[-1] + 1 if rounds else 1

    n = len(PAYNT_METRICS)
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        figsize=figsize or (max(7, n_rounds * 0.32), 2.2 * n),
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, (key, ylabel, logy) in zip(axes, PAYNT_METRICS):
        for p in present:
            pts = sorted(
                (r["round"], r[key])
                for r in rows
                if r["phase"] == p and r[key] is not None
            )
            if pts:
                xs_, ys_ = zip(*pts)
                color, marker, label = phase_style[p]
                ax.plot(
                    xs_,
                    ys_,
                    marker=marker,
                    linestyle="none",
                    color=color,
                    markersize=5,
                    label=label,
                )
        ax.set_ylabel(ylabel, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("learning round")
    axes[-1].set_xticks(rounds)
    axes[-1].set_xticklabels(
        [f"{r}\n|H|={hyp_by_round[r]}" for r in rounds], fontsize=7, linespacing=0.9
    )
    axes[0].legend(loc="upper left", fontsize=8, ncol=len(present))

    label = log_path.split("/")[-1] if isinstance(run, str) else bench_label(run)
    fig.suptitle(
        title or f"PAYNT synthesis over rounds — {label.replace(chr(10), ' ')}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


def plot_paynt_progress_compare(runs, round_range=None, title=None, figsize=None):
    """Overlay PAYNT synthesis stats over rounds for several runs (e.g. methods).

    Like ``plot_paynt_progress`` but draws several runs on the same axes for
    comparison: ``runs`` maps a label -> experiment entry (or log path string).
    Same panels (one per ``PAYNT_METRICS`` series, shared round x-axis), but each
    run gets its own *colour* while its false-negative / false-positive calls keep
    distinct *markers* (○ fn, □ fp). Points of one (run, phase) series are joined
    by a line so the per-round trend is visible; rounds a run resolved by sampling
    (no PAYNT call) leave a gap. Useful for reading off which method is more
    expensive per round and what drives it. Runs may differ in round count; the
    x-axis spans the union. ``round_range=(start, stop)`` restricts the plot to a
    slice of the rounds (0-based, half-open, either bound optional, applied by the
    shared round index) — useful for zooming into a long run.
    """
    parsed = {}
    for label, run in runs.items():
        log_path = run if isinstance(run, str) else run.get("log_path")
        if not log_path or not os.path.exists(log_path):
            raise ValueError(f"no readable log for {label!r}: {log_path!r}")
        rows = parse_paynt_progress(log_path)
        if not rows:
            raise ValueError(
                f"no PAYNT progress lines found for {label!r} in {log_path}"
            )
        parsed[label] = rows

    # Restrict every run to the same window of (global) round indices.
    n_total = max(r["round"] for rows in parsed.values() for r in rows) + 1
    keep = set(_resolve_round_range(n_total, round_range))
    parsed = {
        label: [r for r in rows if r["round"] in keep] for label, rows in parsed.items()
    }
    parsed = {label: rows for label, rows in parsed.items() if rows}
    if not parsed:
        raise ValueError(f"round_range {round_range!r} selects no PAYNT calls")

    phase_marker = {"fn": "o", "fp": "s", "?": "x"}
    phase_name = {"fn": "false negative", "fp": "false positive", "?": "unknown"}
    labels = list(parsed)
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i % 10) for i, label in enumerate(labels)}

    all_rounds = sorted({r["round"] for rows in parsed.values() for r in rows})
    n_rounds = (all_rounds[-1] + 1) if all_rounds else 1

    n = len(PAYNT_METRICS)
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        figsize=figsize or (max(7, n_rounds * 0.32), 2.2 * n),
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, (key, ylabel, logy) in zip(axes, PAYNT_METRICS):
        for label in labels:
            rows = parsed[label]
            for p in phase_marker:
                pts = sorted(
                    (r["round"], r[key])
                    for r in rows
                    if r["phase"] == p and r[key] is not None
                )
                if pts:
                    xs_, ys_ = zip(*pts)
                    ax.plot(
                        xs_,
                        ys_,
                        marker=phase_marker[p],
                        linestyle="-",
                        linewidth=1,
                        color=colors[label],
                        markersize=5,
                        alpha=0.85,
                    )
        ax.set_ylabel(ylabel, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("learning round")
    axes[-1].set_xticks(all_rounds)
    axes[-1].set_xticklabels([str(r) for r in all_rounds], fontsize=7)

    # Two legends: colour = run (method), marker = fn/fp phase.
    method_handles = [
        Line2D([], [], color=colors[label], linestyle="-", label=label)
        for label in labels
    ]
    phases_present = [
        p
        for p in phase_marker
        if any(r["phase"] == p for rows in parsed.values() for r in rows)
    ]
    phase_handles = [
        Line2D(
            [],
            [],
            color="0.4",
            marker=phase_marker[p],
            linestyle="none",
            label=phase_name[p],
        )
        for p in phases_present
    ]
    leg1 = axes[0].legend(
        handles=method_handles, loc="upper left", fontsize=8, title="run"
    )
    axes[0].add_artist(leg1)
    axes[0].legend(handles=phase_handles, loc="upper right", fontsize=8, title="phase")

    fig.suptitle(title or "PAYNT synthesis over rounds — run comparison", fontsize=11)
    fig.tight_layout()
    plt.show()


def group_summary(groups):
    """Per-group success / runtime summary as a DataFrame.

    Counts how many runs finished vs failed and summarises the runtime of the
    finished ones — useful for ranking which group "works best". ``runs`` counts
    every seed-run, so ``success %`` is the per-seed-run reliability. The runtime
    stats (``median``/``total``/``max``) are computed over the per-benchmark mean
    runtime (averaging the seeds first), so they aren't inflated by seed count;
    ``benchmarks`` is how many distinct benchmarks the group solved at all.
    """
    rows = []
    for label, ds in groups.items():
        finished = [d for d in ds if d["results"] is not None]
        n = len(ds)
        per_bench = []  # mean runtime per benchmark, averaged over its seeds
        for entries in entries_by_bench(ds).values():
            rts = [
                r
                for d in entries
                if d["results"] is not None and np.isfinite(r := runtime(d))
            ]
            if rts:
                per_bench.append(float(np.mean(rts)))
        rows.append(
            {
                "group": label,
                "runs": n,
                "finished": len(finished),
                "failed": n - len(finished),
                "success %": round(100 * len(finished) / n, 1) if n else None,
                "benchmarks": len(per_bench),
                "median (s)": round(float(np.median(per_bench)), 2)
                if per_bench
                else None,
                "total (s)": round(sum(per_bench), 1) if per_bench else None,
                "max (s)": round(max(per_bench), 1) if per_bench else None,
            }
        )
    return pd.DataFrame(rows)


def find_runs(groups, group=None, benchmark=None, links=False):
    """Locate the individual seed-runs behind a ``method_table`` cell.

    Coordinates match ``method_table``: ``group`` is a method/column label, and
    ``benchmark`` selects the row — either a substring of the ``"name file h=…"``
    label or the exact ``(name, file, h)`` index tuple shown by ``method_table``.
    Because a cell aggregates several seeds, this returns **one row per seed-run**
    (with its ``seed``, sorted so failures are easy to spot), exposing the
    ``name``/``file``/``h`` row index, runtime, the per-phase timing split (the
    same ``TIME_COMPONENTS`` as ``plot_runtime_breakdown``: learning, smt,
    reference, product, paynt, eq, counterexample, other), finished flag, any
    error and the ``json`` / ``log`` paths. With ``links=True`` the paths render
    as clickable file links (a Styler).

    Examples
    --------
    >>> find_runs(groups, group="bisection")               # all bisection runs
    >>> find_runs(groups, benchmark="airport")             # all airport runs
    >>> find_runs(groups, "bisection", "airportA-3")       # substring row match
    >>> find_runs(groups, "restart", ("snakes_ladders", "mc_u_nxn.pm", 12))
    """
    want_row = tuple(benchmark) if isinstance(benchmark, (tuple, list)) else None
    rows = []
    for glabel, ds in groups.items():
        if group is not None and glabel != group:
            continue
        for d in ds:
            idx = bench_row(d)
            if want_row is not None:
                if idx != want_row:
                    continue
            elif benchmark is not None and benchmark not in bench_label(d).replace(
                "\n", " "
            ):
                continue
            try:
                rt = runtime(d)
            except Exception:
                rt = None  # keep the row so the offending json can be opened
            try:
                comp = time_components(d)
            except Exception:
                comp = None
            err = d.get("error")
            jp, lp = d.get("json_path"), d.get("log_path")
            mon = d["results"].get("dot_file") if d["results"] else None
            rows.append(
                {
                    "group": glabel,
                    "name": idx[0],
                    "file": idx[1],
                    "h": idx[2],
                    "seed": d["experiment"].get("seed"),
                    "runtime": round(rt, 2) if rt is not None else None,
                    # Per-phase timing (same split as plot_runtime_breakdown).
                    **{
                        cn: (round(comp[cn], 2) if comp is not None else None)
                        for cn, _ in TIME_COMPONENTS
                    },
                    "finished": d["results"] is not None,
                    "error": (str(err)[:50] if err else None),
                    "json": os.path.abspath(jp) if jp else None,
                    "log": os.path.abspath(lp) if lp else None,
                    "model": os.path.abspath(mon) if mon else None,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["group", "name", "file", "h", "seed"], na_position="first"
        ).reset_index(drop=True)
    if links and not df.empty:
        # vscode://file/<abs-path> opens the file in the editor; plain file://
        # links are blocked by the notebook renderer. Show the full path as the
        # link text so it is also copyable.
        def _link(p):
            return f'<a href="vscode://file{p}">{p}</a>' if p else ""

        return df.style.format({"json": _link, "log": _link})
    return df


def _metric_value(d, metric):
    r = d["results"]
    if r is None:
        return None
    if metric == "time":
        return r.get("time")
    if metric == "steps":
        ls = r.get("learning_stats") or {}
        return ls.get("steps_learning", ls.get("sul_steps"))
    if metric == "eqs":
        ls = r.get("learning_stats") or {}
        return ls.get("validity_query", ls.get("learning_rounds"))
    if metric == "monitor":
        ls = r.get("learning_stats") or {}
        return r.get("monitor_states", ls.get("automaton_size"))
    raise ValueError(f"unknown metric: {metric!r}")


METHOD_METRICS = {
    "time": "runtime (s)",
    "steps": "steps",
    "eqs": "EQs",
    "monitor": "|M|",
}

# Short labels shown in place of a missing metric value, explaining why a method
# has no (valid) result for a benchmark. "incorrect" mirrors the correctness
# definition in tover.analysis.plots._resolve_time.
FAILURE_REASONS = ("∞", "MO", "ERR", "incorrect", "—")


def failure_reason(d):
    """Why a method has no valid value for a benchmark, or None if it succeeded.

    Returns one of: ``"∞"`` (timed out), ``"MO"`` (ran out of memory),
    ``"ERR"`` (crashed with some other error), ``"incorrect"`` (finished but the
    learned monitor violates the fp/fn thresholds), or ``"—"`` (never started).
    A successful, correct run returns ``None``.
    """
    r = d["results"]
    if r is not None:
        e = d["experiment"]
        threshold, fp_slack, fn_slack = (
            e["threshold"],
            e["fp_slack"],
            e["fn_slack"],
        )
        fp, fn = r.get("false_positive"), r.get("false_negative")
        if (
            fp is None
            or fn is None
            or fp < threshold - fp_slack
            or fn > threshold + fn_slack
        ):
            return "incorrect"
        return None
    err = str(d.get("error") or "")
    if err == "timeout":
        return "∞"
    if err == "not_started":
        return "—"
    if err == "OOM" or "mem" in err.lower():
        return "MO"
    return "ERR"


def method_table(groups, metrics=("time", "steps", "eqs", "monitor")):
    """Per-benchmark table of runtime, steps, EQs and monitor size per method.

    ``groups`` is the same ordered ``label -> entries`` mapping used elsewhere,
    where each label is one learning method/configuration. Methods that never
    produced a result are dropped. Each benchmark is one row, indexed by
    ``(name, file, h)``; the leading ``benchmark`` block holds the remaining
    details (threshold ``λ`` and the MC size ``|S|``/``|T|``), followed by one
    block of columns per method giving its runtime (s), learning steps, number of
    equivalence queries (EQs) and learned-monitor size (``|M|``). Where a method
    has no valid value the cell instead holds a short reason string (see
    ``failure_reason``): ``"∞"`` (timeout), ``"MO"`` (out of memory), ``"ERR"``
    (other error), ``"incorrect"`` or ``"—"`` (not started). The result is a
    DataFrame with a two-level column index, so e.g. ``df["lstar"]`` selects one
    method's block.

    ``metrics`` selects which per-method columns to show (any subset/order of
    ``"time"``, ``"steps"``, ``"eqs"``, ``"monitor"``) — e.g.
    ``method_table(groups, metrics=("time",))`` is a benchmark x method runtime
    matrix that keeps the detailed ``benchmark`` block.

    When a benchmark was run under several ``seed``s (they share a bench key), the
    seed-runs are aggregated into one cell: each metric becomes an ``Agg`` (mean,
    std, n, n_total) over the seeds that succeeded. ``style_method_table`` renders
    this as ``mean ± std (n)`` and highlights the best mean per metric; with a
    single seed it shows just the value, exactly as before.
    """
    metric_label = METHOD_METRICS
    groups = {
        g: ds for g, ds in groups.items() if any(d["results"] is not None for d in ds)
    }

    bench_details: dict = {}  # bench key -> detail columns
    bench_metrics: dict = {}  # bench key -> per-method metric columns
    bench_index: dict = {}  # bench key -> (name, file, horizon) row index
    for label, ds in groups.items():
        # All seed-runs of a benchmark collapse to one cell; aggregate over them.
        for k, entries in entries_by_bench(ds).items():
            if k not in bench_details:
                src = next((d for d in entries if d.get("mc")), entries[0])
                e = src["experiment"]
                threshold = e.get("threshold")
                mc = src.get("mc") or {}
                bench_details[k] = {
                    ("benchmark", "λ"): float(threshold)
                    if isinstance(threshold, Fraction)
                    else threshold,
                    ("benchmark", "|S|"): mc.get("mc_states"),
                    ("benchmark", "|T|"): mc.get("mc_transitions"),
                }
                bench_index[k] = bench_row(src)
            rec = bench_metrics.setdefault(k, {})
            ok = [d for d in entries if failure_reason(d) is None]
            for m in metrics:
                # Mean ± std over the seeds that succeeded; if none did, the cell
                # holds the (most common) failure reason instead of a number.
                if ok:
                    rec[(label, metric_label[m])] = _aggregate(
                        [_metric_value(d, m) for d in ok], len(entries)
                    )
                else:
                    rec[(label, metric_label[m])] = _summarize_reason(entries)

    columns = [("benchmark", "λ"), ("benchmark", "|S|"), ("benchmark", "|T|")]
    columns += [(label, metric_label[m]) for label in groups for m in metrics]

    rows, index = [], []
    for k in sorted(bench_details, key=lambda k: bench_index[k]):
        row = dict(bench_details[k])
        row.update(bench_metrics.get(k, {}))
        rows.append([row.get(c) for c in columns])
        index.append(bench_index[k])

    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["name", "file", "h"]),
        columns=pd.MultiIndex.from_tuples(columns),
    )


def style_method_table(df):
    """Render a ``method_table`` DataFrame as a formatted, highlighted Styler.

    For every metric, the smallest value across methods is bold-highlighted on
    each row (ties all highlighted; reason cells ignored), so the best method per
    metric is easy to read off. Failure-reason cells (``"∞"``, ``"MO"``,
    ``"ERR"``, ``"incorrect"``, ``"—"``) are rendered in muted italics. A
    vertical divider is drawn before each method block to separate the methods.
    """
    methods = [m for m in dict.fromkeys(c[0] for c in df.columns) if m != "benchmark"]
    metrics = list(dict.fromkeys(c[1] for c in df.columns if c[0] in methods))

    number_format = {
        "runtime (s)": "{:.1f}",
        "λ": "{:.1f}",
        "steps": "{:,.0f}",
        "h": "{:.0f}",
        "|S|": "{:.0f}",
        "|T|": "{:.0f}",
        "EQs": "{:.0f}",
        "|M|": "{:.0f}",
    }

    def _fmt(spec):
        # Cells mix Aggs (seed mean/std), plain numbers, reason strings and NaN.
        # Render an Agg as "mean ± std (n)" — dropping the std/count for a single
        # seed so single-run tables look exactly as before — keep reason text,
        # blank out NaN, and number-format everything else.
        def f(v):
            if isinstance(v, Agg):
                mean = spec.format(v.mean)
                if v.n_total <= 1:
                    return mean
                count = f"{v.n}" if v.n == v.n_total else f"{v.n}/{v.n_total}"
                if v.n >= 2:
                    return f"{mean} ±{spec.format(v.std)} ({count})"
                return f"{mean} ({count})"
            if isinstance(v, str):
                return v
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return spec.format(v)

        return f

    styler = df.style.format(
        {
            col: _fmt(number_format[col[1]])
            for col in df.columns
            if col[1] in number_format
        }
    )

    # Per-row highlight of the smallest numeric value across methods, per metric,
    # plus muted styling for the reason cells. Done in one apply so it works on
    # the mixed number/string columns (built-in highlight_min cannot).
    best = "font-weight:bold;background-color:#00521d;color:white;"
    reason_style = "color:#999;font-style:italic;"

    def _num(v):
        # Comparable value for highlighting: an Agg ranks by its mean.
        if isinstance(v, Agg):
            return v.mean
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            return v
        return None

    def _row_styles(row):
        out = pd.Series("", index=row.index)
        for col in row.index:
            if col[0] in methods and isinstance(row[col], str):
                out[col] = reason_style
        for metric in metrics:
            cells = [(m, metric) for m in methods if (m, metric) in row.index]
            nums = [(c, v) for c in cells if (v := _num(row[c])) is not None]
            if not nums:
                continue
            mn = min(v for _, v in nums)
            for c, v in nums:
                if v == mn:
                    out[c] = best
        return out

    styler = styler.apply(_row_styles, axis=1)

    # Vertical divider before each method block (first metric column of a method).
    divider = "2px solid #888"
    table_styles = {
        (m, metrics[0]): [
            {"selector": "td", "props": f"border-left:{divider};"},
            {"selector": "th", "props": f"border-left:{divider};"},
        ]
        for m in methods
        if metrics
    }
    return styler.set_table_styles(table_styles, overwrite=False, axis=0)


def speedup_table(groups, metric="time", max_value=None):
    """Largest per-benchmark speedup between every pair of groups.

    For each unordered pair of groups, finds the benchmark where group A beats
    group B by the widest margin (and vice versa) on ``metric`` ("time" or
    "steps"). ``max_value`` optionally drops entries whose value exceeds it
    (e.g. ``3600`` to ignore timed-out runs). Returns a DataFrame with the ratio,
    both values, and the json/log paths of the two extreme runs.
    """
    rows = []
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            best_1, best_2 = None, None  # n1-faster, n2-faster
            for d1, d2 in pair_by_benchmark(groups[n1], groups[n2]):
                v1, v2 = _metric_value(d1, metric), _metric_value(d2, metric)
                if v1 is None or v2 is None or v1 <= 0 or v2 <= 0:
                    continue
                if max_value is not None and (v1 > max_value or v2 > max_value):
                    continue
                if v2 / v1 > (best_1[0] if best_1 else 1):
                    best_1 = (v2 / v1, v1, v2, d1, d2)
                if v1 / v2 > (best_2[0] if best_2 else 1):
                    best_2 = (v1 / v2, v1, v2, d1, d2)

            for winner, best in [(n1, best_1), (n2, best_2)]:
                if best is None:
                    continue
                ratio, v1, v2, d1, d2 = best
                rows.append(
                    {
                        "group A": n1,
                        "group B": n2,
                        "faster": winner,
                        "ratio": round(ratio, 2),
                        f"{metric} A": round(v1, 2),
                        f"{metric} B": round(v2, 2),
                        "json A": d1.get("json_path"),
                        "json B": d2.get("json_path"),
                        "log A": d1.get("log_path"),
                        "log B": d2.get("log_path"),
                    }
                )
    return pd.DataFrame(rows)
