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
from fractions import Fraction

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
RUNTIME_COLOR = "black"

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
    if sum(comp.values()) > total:
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


def _group_components(groups):
    """For each group, map benchmark key -> components (first entry per pair)."""
    out = {}
    for label, ds in groups.items():
        by_key = {}
        for d in ds:
            k = bench_key(d)
            if k not in by_key and (c := time_components(d)) is not None:
                by_key[k] = c
        out[label] = by_key
    return out


def plot_runtime_breakdown(groups, relative=True, title=None):
    """Clustered stacked runtime-breakdown bars on a single axis.

    ``groups`` is an ordered mapping ``label -> list of experiment entries``.
    Benchmarks form clusters along the x-axis; inside each cluster there is one
    stacked bar per group (in ``groups`` order), so the groups being compared sit
    next to each other. Bars show the phase breakdown (fraction of own runtime
    when ``relative``, else seconds); when ``relative`` the total runtime of each
    bar is also drawn as a dot on a right-hand log axis.

    Each bar is labelled with its group; benchmark names label the clusters.
    """
    nonempty = {g: ds for g, ds in groups.items() if ds}
    group_components = _group_components(nonempty)
    group_components = {g: bk for g, bk in group_components.items() if bk}
    group_names = list(group_components)
    n_groups = len(group_names)

    col_label = {}
    for ds in nonempty.values():
        for d in ds:
            if time_components(d) is not None:
                col_label.setdefault(bench_key(d), bench_label(d))
    bench_keys = sorted(col_label, key=lambda k: col_label[k])
    n_bench = len(bench_keys)

    runtimes = [
        t
        for bk in group_components.values()
        for c in bk.values()
        if (t := sum(c.values())) > 0
    ]
    max_runtime = max(runtimes, default=1.0)
    min_runtime = min(runtimes, default=0.1)
    rt_floor = min_runtime / 2
    used = {
        cn
        for bk in group_components.values()
        for c in bk.values()
        for cn, _ in TIME_COMPONENTS
        if c[cn] > 0
    }

    cluster_w = 0.86
    slot_w = cluster_w / n_groups
    bar_w = slot_w * 0.82
    centers = np.arange(n_bench)

    fig, ax = plt.subplots(figsize=(max(12, n_bench * max(n_groups, 4) * 0.42), 5.5))
    ax2 = ax.twinx() if relative else None

    bar_x, bar_lab = [], []  # per-bar tick positions / group labels
    for gi, gname in enumerate(group_names):
        by_key = group_components[gname]
        xs = centers - cluster_w / 2 + (gi + 0.5) * slot_w
        bottom = np.zeros(n_bench)
        for cname, color in TIME_COMPONENTS:
            if cname not in used:
                continue
            vals = np.array(
                [
                    (
                        by_key[k][cname] / (sum(by_key[k].values()) or 1.0)
                        if relative
                        else by_key[k][cname]
                    )
                    if k in by_key
                    else 0.0
                    for k in bench_keys
                ]
            )
            ax.bar(xs, vals, bottom=bottom, width=bar_w, color=color)
            bottom += vals
        if relative:
            rt = np.array(
                [sum(by_key[k].values()) if k in by_key else np.nan for k in bench_keys]
            )
            present = ~np.isnan(rt)
            ax2.plot(
                xs[present],
                rt[present],
                "o",
                color=RUNTIME_COLOR,
                markersize=4,
                alpha=0.85,
            )
        for ci, k in enumerate(bench_keys):
            if k in by_key:
                bar_x.append(xs[ci])
                bar_lab.append(gname)

    # Per-bar group labels are drawn as text (not minor ticks) because matplotlib
    # hides a minor label that coincides with a major tick -- which is exactly
    # the centre bar of every cluster. Benchmark labels are padded major ticks
    # sitting below, kept multi-line so they stay narrow and don't overlap.
    for xb, lab in zip(bar_x, bar_lab):
        ax.text(
            xb,
            -0.012,
            lab,
            rotation=90,
            ha="center",
            va="top",
            fontsize=6,
            transform=ax.get_xaxis_transform(),
        )
    pad = 14 + int(5.0 * max((len(g) for g in group_names), default=4))
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [col_label[k] for k in bench_keys],
        fontsize=7,
        fontweight="bold",
        linespacing=0.9,
    )
    ax.tick_params(axis="x", which="major", length=0, pad=pad)
    for c in centers[:-1]:
        ax.axvline(c + 0.5, color="0.85", lw=0.8)
    ax.set_xlim(-0.5, n_bench - 0.5)

    if relative:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("fraction of runtime")
        ax2.set_yscale("log")
        ax2.set_ylim(rt_floor, max_runtime * 1.3)
        ax2.set_ylabel("runtime (s, log)")
    else:
        ax.set_ylabel("runtime (s)")

    handles = [Patch(color=c, label=n) for n, c in TIME_COMPONENTS if n in used]
    if relative:
        from matplotlib.lines import Line2D

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=RUNTIME_COLOR,
                markeredgecolor="none",
                label="runtime (s)",
            )
        )
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=len(handles),
        fontsize=8,
    )
    ax.set_title(
        title or "Runtime breakdown — clusters: benchmark, bars: group",
        pad=24,
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


def plot_round_breakdown(d, relative=False, title=None, figsize=None):
    """Stacked per-round runtime-breakdown bars for a single experiment.

    One bar per learning round, stacked into the ROUND_COMPONENTS phases (seconds,
    or fraction of the round's runtime when ``relative``). Each bar is annotated
    with the hypothesis size ``|H|`` it was built from and which check found the
    counterexample (``fn`` / ``fp`` / ``samp``; ``✓`` for the final round that
    found none). ``d`` is a single experiment entry (as produced by
    ``load_experiment_data``).
    """
    rounds = round_components(d)
    if rounds is None:
        raise ValueError(
            f"{d.get('json_path')}: run has no per-round timing data "
            "(needs the L#box learning path with round_timings / rounds)"
        )

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
            f"{i + 1}\n|H|={r['_size']}" if r["_size"] is not None else f"{i + 1}"
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


def group_summary(groups):
    """Per-group success / runtime summary as a DataFrame.

    Counts how many runs finished vs failed and summarises the runtime of the
    finished ones — useful for ranking which group "works best".
    """
    rows = []
    for label, ds in groups.items():
        finished = [d for d in ds if d["results"] is not None]
        rts = [runtime(d) for d in finished]
        n = len(ds)
        rows.append(
            {
                "group": label,
                "runs": n,
                "finished": len(finished),
                "failed": n - len(finished),
                "success %": round(100 * len(finished) / n, 1) if n else None,
                "median (s)": round(float(np.median(rts)), 2) if rts else None,
                "total (s)": round(sum(rts), 1) if rts else None,
                "max (s)": round(max(rts), 1) if rts else None,
            }
        )
    return pd.DataFrame(rows)


def runtime_pivot(groups):
    """Benchmark x group table of runtimes (NaN where the run did not finish).

    Lets you read across a row to see which group solved a benchmark fastest,
    and spot which groups fail (NaN) on which benchmarks.
    """
    labels = {}
    records: dict = {}
    for label, ds in groups.items():
        for d in ds:
            k = bench_key(d)
            labels[k] = bench_label(d).replace("\n", " ")
            records.setdefault(k, {})[label] = runtime(d)  # None if unfinished
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index = [labels[k] for k in df.index]
    cols = [g for g in groups if g in df.columns]
    return df[cols].sort_index()


def find_runs(groups, group=None, benchmark=None, links=False):
    """Locate runs for inspection and return their json/log paths.

    Filters use the same coordinates as ``runtime_pivot``: ``group`` is an exact
    column label and ``benchmark`` is a substring matched against the row label.
    Returns a DataFrame (one row per matching run) with the runtime, whether it
    finished, any error, and the ``json`` / ``log`` paths. With ``links=True``
    the paths are rendered as clickable file links (a Styler) instead of text.

    Examples
    --------
    >>> find_runs(groups, group="bisection")            # all bisection runs
    >>> find_runs(groups, benchmark="airport")          # all airport runs
    >>> find_runs(groups, "bisection", "airportA-3")    # one specific cell
    """
    rows = []
    for glabel, ds in groups.items():
        if group is not None and glabel != group:
            continue
        for d in ds:
            blabel = bench_label(d).replace("\n", " ")
            if benchmark is not None and benchmark not in blabel:
                continue
            try:
                rt = runtime(d)
            except Exception:
                rt = None  # keep the row so the offending json can be opened
            err = d.get("error")
            jp, lp = d.get("json_path"), d.get("log_path")
            rows.append(
                {
                    "group": glabel,
                    "benchmark": blabel,
                    "runtime": round(rt, 2) if rt is not None else None,
                    "finished": d["results"] is not None,
                    "error": (str(err)[:50] if err else None),
                    "json": os.path.abspath(jp) if jp else None,
                    "log": os.path.abspath(lp) if lp else None,
                }
            )
    df = pd.DataFrame(rows)
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

    Use ``style_method_table`` to render it with the best (smallest) value per
    metric highlighted and dividers between methods.
    """
    metric_label = METHOD_METRICS
    groups = {
        g: ds for g, ds in groups.items() if any(d["results"] is not None for d in ds)
    }

    bench_details: dict = {}  # bench key -> detail columns
    bench_metrics: dict = {}  # bench key -> per-method metric columns
    bench_index: dict = {}  # bench key -> (name, file, horizon) row index
    for label, ds in groups.items():
        seen = set()
        for d in ds:
            k = bench_key(d)
            if k not in bench_details:
                e = d["experiment"]
                threshold = e.get("threshold")
                mc = d.get("mc") or {}
                bench_details[k] = {
                    ("benchmark", "λ"): float(threshold)
                    if isinstance(threshold, Fraction)
                    else threshold,
                    ("benchmark", "|S|"): mc.get("mc_states"),
                    ("benchmark", "|T|"): mc.get("mc_transitions"),
                }
                file = str(e.get("file") or "").split("/")[-1]
                bench_index[k] = (e.get("name"), file, e.get("horizon"))
            if k in seen:  # one entry per benchmark per method (first wins)
                continue
            seen.add(k)
            rec = bench_metrics.setdefault(k, {})
            # On failure every metric cell holds the reason instead of a number.
            reason = failure_reason(d)
            for m in metrics:
                rec[(label, metric_label[m])] = (
                    reason if reason is not None else _metric_value(d, m)
                )

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
        # Cells mix numbers with reason strings (and NaN for absent runs); keep
        # the reason text, blank out NaN, and number-format everything else.
        def f(v):
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

    def _row_styles(row):
        out = pd.Series("", index=row.index)
        for col in row.index:
            if col[0] in methods and isinstance(row[col], str):
                out[col] = reason_style
        for metric in metrics:
            cells = [(m, metric) for m in methods if (m, metric) in row.index]
            nums = [
                (c, row[c])
                for c in cells
                if isinstance(row[c], (int, float))
                and not (isinstance(row[c], float) and np.isnan(row[c]))
            ]
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
