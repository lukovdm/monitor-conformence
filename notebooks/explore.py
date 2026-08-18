import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # --- Interactive experiment explorer (marimo) -------------------------------
    # A reactive front-end over the existing `tover.analysis` helpers. Change a
    # dropdown/slider and every dependent cell re-runs automatically (no stale
    # state, no re-running cells by hand). Run with:
    #
    #     uv run marimo edit notebooks/explore.py      # edit / explore
    #     uv run marimo run  notebooks/explore.py      # read-only app view
    #
    # Nothing here re-implements analysis logic; it only *selects* inputs and
    # renders the same `group_summary` / `method_table` / `plot_*` / `find_runs`
    # outputs the notebooks already use.
    import marimo as mo
    import os
    import sys

    # Anchor on the repo root (the dir holding pyproject.toml) regardless of the
    # cwd marimo was launched from, so `import tover...` and `out/` both resolve.
    ROOT = os.getcwd()
    while ROOT != "/" and not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        ROOT = os.path.dirname(ROOT)
    return ROOT, mo, os, sys


@app.cell
def _(mo):
    # Refresh buttons. Each click re-runs the downstream cell that reads its
    # ``.value`` (marimo button values are click counters — referencing them is
    # how a cell subscribes to the click).
    #   • "↻ folder list" → re-lists ``out/`` (pick up newly-created experiments).
    #   • "↻ reload data" → re-reads JSON for the currently selected folder
    #     (useful while an experiment is still writing files).
    # Refreshing the folder list re-creates the exp_dd dropdown and resets it to
    # its default; pick again afterwards if needed.
    refresh_folders = mo.ui.button(label="↻ folder list", kind="neutral")
    refresh_data = mo.ui.button(label="↻ reload data", kind="neutral")
    mo.hstack([refresh_folders, refresh_data], justify="start", gap=1)
    return refresh_data, refresh_folders


@app.cell
def _(ROOT, sys):
    # Force a non-interactive backend *before* importing the analysis module
    # (which imports pyplot at import time): the plot helpers call plt.show(),
    # which is a no-op under Agg — we capture plt.gcf() instead and let marimo
    # render the figure.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    import pandas as pd

    from tover.analysis.load_data import (
        load_experiment_data,
        clean_data,
        add_symbol_color,
    )
    import importlib
    import tover.analysis.breakdown as _breakdown
    importlib.reload(_breakdown)
    import tover.analysis.plots as _plots
    importlib.reload(_plots)

    from tover.analysis.breakdown import (
        group_summary,
        method_table,
        style_method_table,
        plot_runtime_breakdown,
        plotly_runtime_breakdown,
        plot_round_breakdown,
        plotly_round_breakdown,
        plot_round_breakdown_compare,
        plot_tree_growth,
        plotly_tree_growth,
        cq_seqs_summary,
        find_runs,
        bench_row,
        bench_label,
        time_components,
        TIME_COMPONENTS,
    )
    from tover.analysis.plots import compare_runtimes, plotly_compare_runtimes


    return (
        TIME_COMPONENTS,
        add_symbol_color,
        bench_label,
        bench_row,
        clean_data,
        load_experiment_data,
        method_table,
        pd,
        plotly_compare_runtimes,
        plotly_round_breakdown,
        plotly_runtime_breakdown,
        plotly_tree_growth,
        time_components,
    )


@app.cell
def _(ROOT, mo, os, refresh_folders):
    # 1) Pick which experiment folder to explore. Any ``out/<dir>`` that has a
    #    ``json/`` subdir is offered; defaults to the newest compare_cq_amount run.
    #    Re-runs when "↻ folder list" is clicked (button value = click counter).
    _ = refresh_folders.value
    _out_dir = os.path.join(ROOT, "out")
    _folders = sorted(
        f
        for f in os.listdir(_out_dir)
        if os.path.isdir(os.path.join(_out_dir, f, "json"))
    )
    _default = next(
        (f for f in reversed(_folders) if "exp" in f),
        _folders[-1] if _folders else None,
    )
    exp_dd = mo.ui.dropdown(_folders, value=_default, label="Experiment folder")
    mo.md(f"### Experiment\n{exp_dd}")
    return (exp_dd,)


@app.cell
def _(
    ROOT,
    add_symbol_color,
    clean_data,
    exp_dd,
    load_experiment_data,
    mo,
    os,
    refresh_data,
):
    # 2) Load the selected experiment (reactive: re-runs on dropdown change or
    #    when "↻ reload data" is clicked).
    _ = refresh_data.value
    PATH = os.path.join(ROOT, "out", exp_dd.value)
    data = load_experiment_data(PATH)
    clean_data(data)
    add_symbol_color(data)
    mo.md(f"**{len(data)}** entries loaded from `{exp_dd.value}`")
    return (data,)


@app.cell
def _(data, mo):
    # 3) Choose the dimension whose values become the comparison columns/groups.
    #    Only dimensions that actually vary in this experiment are offered; the
    #    accessor for each handles where the field lives in the experiment dict.


    def _method_variant(d):
        # "lsharp" / "lstar" alone lumps lsharp-with-dc/rl together with plain
        # lsharp. Combine the flags so each variant is its own group:
        # lstar, lsharp, lsharp+dc+rl, ...
        e = d["experiment"]
        m = e.get("learning_method")
        if m is None:
            return None
        parts = [m]
        if e.get("use_dont_care"):
            parts.append("dc")
        if e.get("use_refrence_language"):
            parts.append("rl")
        return "+".join(parts)


    GROUPERS = {
        "max_seqs": lambda d: (d["experiment"].get("random_eq_method") or {}).get(
            "max_seqs"
        ),
        "learning_method": lambda d: d["experiment"].get("learning_method"),
        "method+dc/rl": _method_variant,
        "conditional_method": lambda d: d["experiment"].get("conditional_method"),
        "use_dont_care": lambda d: d["experiment"].get("use_dont_care"),
        "use_refrence_language": lambda d: d["experiment"].get("use_refrence_language"),
        "name": lambda d: d["experiment"].get("name"),
    }
    _varying = [k for k, fn in GROUPERS.items() if len({fn(d) for d in data}) > 1]
    _distinct = {k: len({GROUPERS[k](d) for d in data}) for k in _varying}
    # ``name`` identifies the benchmark — already the table's *rows* — so grouping by
    # it pools every method into one cell and the mean/std then mix methods, not
    # seeds. Keep it selectable but never auto-default to it.
    _IDENTITY_GROUPERS = {"name"}
    # Default: ``max_seqs`` if it varies (per-budget comparisons), otherwise the
    # method/config dimension with the most distinct values — so for
    # compare_lsharp_lstar the default picks ``method+dc/rl`` (3 groups), giving
    # lsharp+dc+rl its own column.
    _default_pool = [k for k in _varying if k not in _IDENTITY_GROUPERS]
    if "max_seqs" in _default_pool:
        _default = "max_seqs"
    elif _default_pool:
        _default = max(_default_pool, key=lambda k: _distinct[k])
    else:
        _default = "name"
    group_dd = mo.ui.dropdown(list(GROUPERS), value=_default, label="Group columns by")
    mo.md(
        f"### Grouping\n{group_dd}  &nbsp; "
        f"(varying here: {', '.join(_varying) or '—'})"
    )
    return GROUPERS, group_dd


@app.cell
def _(GROUPERS, bench_row, data, group_dd, mo):
    # Build {label -> entries} for the chosen dimension, then let the user keep a
    # subset of the resulting groups (columns) to compare.
    def _fmt(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return f"{v:,}"
        return str(v)

    _acc = GROUPERS[group_dd.value]
    _vals = sorted({_acc(d) for d in data if _acc(d) is not None}, key=lambda v: str(v))
    groups = {_fmt(v): [d for d in data if _acc(d) == v] for v in _vals}

    group_sel = mo.ui.multiselect(
        list(groups), value=list(groups), label="Groups to compare"
    )

    import re as _re


    def _config_sig(d):
        # Everything that defines a run *except* the seed; only seeds of one
        # identical config should be averaged together inside a cell.
        return _re.sub(r"seed=[^,)]*,?", "", d["experiment"].get("variant") or "")


    # If the chosen grouping leaves several configs in one (group, benchmark) cell,
    # the table's mean/std mix configs rather than seed repetitions.
    _pooled = []
    for _k, _ds in groups.items():
        _by_bench: dict = {}
        for _d in _ds:
            _by_bench.setdefault(bench_row(_d), set()).add(_config_sig(_d))
        if any(len(s) > 1 for s in _by_bench.values()):
            _pooled.append(_k)
    _warn = (
        f"\n\n> ⚠️ Grouping by **{group_dd.value}** pools several configurations "
        f"per benchmark — mean±std mix configs, not just seeds — in: "
        f"{', '.join(_pooled)}. Choose a finer grouping (e.g. `method+dc/rl`)."
        if _pooled
        else ""
    )
    mo.md(
        f"{group_sel}\n\nSizes: "
        + ", ".join(f"`{k}`={len(v)}" for k, v in groups.items())
        + _warn
    )
    return group_sel, groups


@app.cell
def _(group_sel, groups):
    # The group-selected data, *before* any scatter filtering. ``sel_groups`` /
    # ``flat`` (defined just below the scatter) narrow this down to the runs behind
    # the points selected in the scatter, so every view further down reacts to it.
    base_groups = {k: groups[k] for k in group_sel.value} or groups
    base_flat = [d for ds in base_groups.values() for d in ds]
    return base_flat, base_groups


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Runtime scatter — select to filter everything below

    Compare two methods' per-benchmark runtimes. **Box/lasso-select** points to
    restrict every table and plot below to the selected benchmark/seed runs;
    clear the selection to see all data again.
    """)
    return


@app.cell
def _(base_groups, mo):
    # Pick the two method groups to put on each axis of the runtime scatter.
    _cmp_opts = list(base_groups)
    cmp_x_dd = mo.ui.dropdown(
        _cmp_opts, value=_cmp_opts[0] if _cmp_opts else None, label="x-axis method"
    )
    cmp_y_dd = mo.ui.dropdown(
        _cmp_opts,
        value=_cmp_opts[1] if len(_cmp_opts) > 1 else (_cmp_opts[0] if _cmp_opts else None),
        label="y-axis method",
    )
    mo.hstack([cmp_x_dd, cmp_y_dd], justify="start", gap=1)
    return cmp_x_dd, cmp_y_dd


@app.cell
def _(base_flat, base_groups, cmp_x_dd, cmp_y_dd, mo, plotly_compare_runtimes):
    # Runtime scatter over the full (group-selected) data. ``meta_func`` stashes the
    # flat positions of the paired x/y runs into each point\'s customdata so the
    # filter cell below can map a selection back to runs.
    _pos_of = {id(_d): _i for _i, _d in enumerate(base_flat)}
    _cx, _cy = cmp_x_dd.value, cmp_y_dd.value
    if _cx and _cy and _cx != _cy:
        compare_fig = plotly_compare_runtimes(
            base_groups[_cx],
            base_groups[_cy],
            label1=_cx,
            label2=_cy,
            height=520,
            meta_func=lambda d1, d2: [_pos_of.get(id(d1)), _pos_of.get(id(d2))],
        )
        compare_fig.update_layout(dragmode="select")
        compare_chart = mo.ui.plotly(compare_fig)
    else:
        compare_fig = None
        compare_chart = mo.md("_pick two different methods to compare_")
    compare_chart
    return compare_chart, compare_fig


@app.cell(hide_code=True)
def _(base_flat, base_groups, bench_row, compare_chart, compare_fig):
    # Scatter selection -> global filter. The flat positions stashed in each point\'s
    # customdata identify the runs behind the selected points; keep every run sharing
    # their (benchmark, seed) so all methods stay comparable in the views below.
    plot_sel_ids = set()
    if compare_fig is not None and getattr(compare_chart, "value", None):
        for _pt in compare_chart.value:
            _cn = _pt.get("curveNumber")
            _pn = _pt.get("pointNumber")
            if _pn is None:
                _pn = _pt.get("pointIndex")  # box/lasso selections use pointIndex
            if _cn is None or _pn is None or _cn >= len(compare_fig.data):
                continue
            _cd = getattr(compare_fig.data[_cn], "customdata", None)
            if _cd is not None and _pn < len(_cd):
                for _idx in _cd[_pn][4:6]:  # flat positions of the x-run and y-run
                    if _idx is not None:
                        plot_sel_ids.add(int(_idx))


    def _runkey(d):
        return (*bench_row(d), d["experiment"].get("seed"))


    if plot_sel_ids:
        _keys = {_runkey(base_flat[_i]) for _i in plot_sel_ids}
        sel_groups = {}
        for _k, _ds in base_groups.items():
            _kept = [d for d in _ds if _runkey(d) in _keys]
            if _kept:
                sel_groups[_k] = _kept
    else:
        sel_groups = base_groups
    flat = [d for ds in sel_groups.values() for d in ds]
    return flat, sel_groups


@app.cell(hide_code=True)
def _(bench_row, data, mo):
    # Shared "selected benchmark" state so the runtime-breakdown click event and
    # the bench_dd dropdown can both drive the compare section below. The state
    # is re-created when the experiment (and therefore its benchmark set) changes.
    _benches_all = sorted(
        {bench_row(_d) for _d in data}, key=lambda t: tuple(str(x) for x in t)
    )
    get_selected_bench, set_selected_bench = mo.state(
        _benches_all[0] if _benches_all else None
    )
    return get_selected_bench, set_selected_bench


@app.cell
def _(mo):
    # 5) Full per-benchmark x group metric table (mean ± std over seeds).
    mo.md("""
    ## Metrics per benchmark × group
    """)
    return


@app.cell
def _(mo):
    # Which per-group metrics to include as columns. All four on by default;
    # toggle off "steps" / "EQs" if you want to fit more groups on one screen.
    metric_show = mo.ui.multiselect(
        ["runtime (s)", "steps", "EQs", "|M|"],
        value=["runtime (s)", "steps", "EQs", "|M|"],
        label="Per-group metrics to show",
    )
    metric_show
    return (metric_show,)


@app.cell
def _(method_table, metric_show, mo, pd, sel_groups):
    import math

    # Marimo-native rendering of method_table: a sortable / searchable / selectable
    # table. Highlighting (best mean per metric, muted reason cells) is done via
    # ``style_cell`` so the cell strings stay tight — no ★ prefix bloating widths.
    # Fixed per-column ``column_widths`` keep more groups visible without
    # horizontal scrolling, and ``freeze_columns_left`` pins the row identity
    # while the user does scroll.
    _mt = method_table(sel_groups)
    _methods = [m for m in dict.fromkeys(c[0] for c in _mt.columns) if m != "benchmark"]
    _metrics = list(dict.fromkeys(c[1] for c in _mt.columns if c[0] in _methods))
    _visible_metrics = [m for m in _metrics if m in set(metric_show.value)]
    _number_format = {
        "runtime (s)": "{:.1f}", "λ": "{:.1f}", "steps": "{:,.0f}", "h": "{:.0f}",
        "|S|": "{:.0f}", "|T|": "{:.0f}", "|RL|": "{:.0f}", "|Σ|": "{:.0f}",
        "EQs": "{:.0f}", "|M|": "{:.0f}",
    }
    _metric_short = {"runtime (s)": "time", "steps": "steps", "EQs": "EQs", "|M|": "|M|"}


    def _is_agg(v):
        return hasattr(v, "mean") and hasattr(v, "std") and hasattr(v, "n_total")


    def _numeric(v):
        if _is_agg(v):
            return v.mean
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return v
        return None


    def _fmt_cell(v, spec):
        if _is_agg(v):
            m_ = spec.format(v.mean)
            if v.n_total <= 1:
                return m_
            cnt = f"{v.n}" if v.n == v.n_total else f"{v.n}/{v.n_total}"
            if v.n >= 2:
                return f"{m_}±{spec.format(v.std)} ({cnt})"
            return f"{m_} ({cnt})"
        if isinstance(v, str):
            return v
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return spec.format(v)


    _best_cells: set = set()
    _reason_cells: set = set()
    _rows = []
    for _pos, (_ridx, _row) in enumerate(_mt.iterrows()):
        _name, _file, _params, _h = _ridx
        _rec = {"name": _name, "file": _file, "params": _params, "h": _h}
        _row_id = str(_pos)
        _best_per_metric: dict = {}
        for _met in _visible_metrics:
            _nums = []
            for _m in _methods:
                _c = (_m, _met)
                if _c in _mt.columns:
                    _v = _numeric(_row[_c])
                    if _v is not None:
                        _nums.append((_m, _v))
            if _nums:
                _mn = min(v for _, v in _nums)
                _best_per_metric[_met] = {m for m, v in _nums if v == _mn}
        for _col in _mt.columns:
            _grp, _met = _col
            _spec = _number_format.get(_met, "{}")
            _txt = _fmt_cell(_row[_col], _spec)
            if _grp == "benchmark":
                _rec[_met] = _txt
            else:
                if _met not in _visible_metrics:
                    continue
                _short = _metric_short.get(_met, _met)
                _col_name = f"{_grp} \n {_short}"
                _rec[_col_name] = _txt
                if _grp in _best_per_metric.get(_met, set()):
                    _best_cells.add((_row_id, _col_name))
                if isinstance(_row[_col], str):
                    _reason_cells.add((_row_id, _col_name))
        _rows.append(_rec)

    _df = pd.DataFrame(_rows)


    def _style_cell(rowId, columnName, value):
        if (rowId, columnName) in _best_cells:
            return {"fontWeight": "bold", "backgroundColor": "#00521d", "color": "white"}
        if (rowId, columnName) in _reason_cells:
            return {"color": "#999", "fontStyle": "italic"}
        return {}


    _widths = {
        "name": 100,
        "file": 160,
        "params": 110,
        "h": 35,
        "λ": 45, "|S|": 55, "|T|": 60, "|RL|": 50, "|Σ|": 50,
    }
    _per_metric_width = {"time": 105, "steps": 90, "EQs": 90, "|M|": 70}
    for _c in _df.columns:
        if " · " in _c:
            _widths[_c] = _per_metric_width.get(_c.split(" · ")[-1], 90)

    _numeric_cols = [c for c in _df.columns if c not in ("name", "file", "params")]

    metric_table = mo.ui.table(
        _df,
        selection="single",
        page_size=20,
        label="Metrics per benchmark × group (best mean highlighted)",
        style_cell=_style_cell,
        column_widths=_widths,
        freeze_columns_left=["name", "file", "params", "h"],
        text_justify_columns={c: "right" for c in _numeric_cols},
        max_columns=None,
    )
    metric_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Runtime breakdown — interactive (plotly: hover + click)
    """)
    return


@app.cell
def _(bench_label, bench_row, flat, mo, plotly_runtime_breakdown, sel_groups):
    # Plotly twin of the matplotlib breakdown above. ``runtime_fig`` is kept so
    # the click watcher below can read ``runtime_fig.data[i].customdata`` to find
    # the clicked benchmark; ``bench_label_to_row`` reverses the flattened facet
    # label back to a bench_row tuple.
    runtime_fig = plotly_runtime_breakdown(sel_groups)
    runtime_chart = mo.ui.plotly(runtime_fig)
    bench_label_to_row = {
        bench_label(_d).replace("\n", " · "): bench_row(_d) for _d in flat
    }
    runtime_chart
    return bench_label_to_row, runtime_chart, runtime_fig


@app.cell(hide_code=True)
def _(
    bench_label_to_row,
    get_selected_bench,
    mo,
    runtime_chart,
    runtime_fig,
    set_selected_bench,
):
    # Click watcher: any click on the runtime breakdown chart pushes its benchmark
    # to the shared selected_bench state. Click events surface in
    # ``runtime_chart.value`` as a list of points; we map (curveNumber, pointNumber)
    # back to the trace's customdata, which plotly_runtime_breakdown encodes as
    # [benchmark_label, group, phase, ...].
    _pts = runtime_chart.value or []
    if _pts:
        _p = _pts[0]
        _cn = _p.get("curveNumber")
        _pn = _p.get("pointNumber")
        if _cn is not None and _pn is not None and _cn < len(runtime_fig.data):
            _cd = getattr(runtime_fig.data[_cn], "customdata", None)
            if _cd is not None and _pn < len(_cd):
                _lbl = _cd[_pn][0]
                _row = bench_label_to_row.get(_lbl)
                if _row is not None and _row != get_selected_bench():
                    set_selected_bench(_row)
    mo.md("")
    return


@app.cell
def _(bench_row, data, get_selected_bench, mo, set_selected_bench):
    # 6) Drill into one benchmark across the selected groups. The dropdown is
    #    two-way bound to ``selected_bench`` state so clicking a bar in the runtime
    #    breakdown chart above also updates it.
    _benches = sorted(
        {bench_row(_d) for _d in data}, key=lambda t: tuple(str(x) for x in t)
    )
    _options = {
        f"{n}  |  {f}" + (f"  |  {p}" if p else "") + f"  |  h={h}": (n, f, p, h)
        for (n, f, p, h) in _benches
    }
    _value_to_label = {v: k for k, v in _options.items()}
    _current = get_selected_bench()
    _initial_label = _value_to_label.get(_current) or next(iter(_options), None)
    bench_dd = mo.ui.dropdown(
        _options,
        value=_initial_label,
        label="Benchmark",
        on_change=set_selected_bench,
    )
    mo.md(
        f"## Compare one benchmark across groups\n"
        f"{bench_dd}\n\n"
        f"_click a bar in the runtime breakdown above to drill into that benchmark_"
    )
    return


@app.cell
def _(mo, sel_groups):
    # Which of the selected groups to overlay on the per-round comparison plots
    # below. Defaults to *all* groups currently selected; toggle off to compare a
    # subset (e.g. only the smallest and largest budgets).
    compare_group_sel = mo.ui.multiselect(
        list(sel_groups),
        value=list(sel_groups),
        label="Groups to compare on this benchmark",
    )
    compare_group_sel
    return (compare_group_sel,)


@app.cell
def _(bench_row, compare_group_sel, get_selected_bench, mo, sel_groups):
    # Pick the latest-seed run (with a log) for the chosen benchmark in each
    # group selected for comparison.
    target = get_selected_bench()
    _methods = [m for m in compare_group_sel.value if m in sel_groups]


    def _pick(method):
        cands = [
            d
            for d in sel_groups[method]
            if bench_row(d) == target and d.get("log_path")
        ]
        if not cands:
            return None
        return max(cands, key=lambda d: d["experiment"].get("seed") or 0)


    runs = {m: r for m in _methods if (r := _pick(m)) is not None}

    _n_rounds = max(
        (len((r["results"] or {}).get("rounds") or []) for r in runs.values()),
        default=0,
    )
    round_slider = mo.ui.range_slider(
        start=0,
        stop=max(_n_rounds, 1),
        value=[0, max(_n_rounds, 1)],
        label="Round range",
        full_width=True,
    )
    _msg = (
        f"Comparing **{len(runs)}** group(s) on `{target}` — "
        + ", ".join(
            f"{m} (seed {r['experiment'].get('seed')})" for m, r in runs.items()
        )
        if runs
        else f"No run with a log for `{target}` in the selected groups."
    )
    mo.md(f"{_msg}\n\n{round_slider if runs else ''}")
    return round_slider, runs


@app.cell
def _(mo, plotly_round_breakdown, round_slider, runs):
    # Per-round wall-time phase split, side-by-side bars per group via the
    # unified plotly_round_breakdown (multi-run branch). Not every run carries
    # per-round timing (only the L#box path does), so degrade gracefully.
    if not runs:
        _out = mo.md("_select a benchmark present in the chosen groups_")
    else:
        try:
            _out = plotly_round_breakdown(
                runs, round_range=tuple(round_slider.value)
            )
        except Exception as _e:
            _out = mo.md(f"_no per-round timing data for this benchmark: {_e}_")
    _out
    return


@app.cell
def _(mo, plotly_tree_growth, runs):
    # Observation-tree growth & reference coverage, one plotly figure per compared run.
    def _tree(r):
        try:
            return plotly_tree_growth(r)
        except Exception as _ex:
            return mo.md(f"_no tree-growth data: {_ex}_")


    mo.vstack(
        [mo.vstack([mo.md(f"**{m}**"), _tree(r)]) for m, r in runs.items()]
    ) if runs else mo.md("")
    return


@app.cell
def _(mo):
    # 7) Clickable drill-down: every seed-run in the current (scatter-filtered)
    #    selection as a row. Tick rows to inspect their method, paths, total time and
    #    per-round breakdowns below.
    mo.md("""
    ## Inspect individual seed-runs (tick rows)

    Every seed-run in the current selection. Tick one or more rows to see their
    details below.
    """)
    return


@app.cell
def _(bench_row, mo, pd, sel_groups):
    # Seed-runs in the current selection. ``id`` indexes into ``flat`` (the detail
    # cell reads ``flat[int(id)]``). Tick rows to drill into their details below.
    _rows = []
    _i = 0
    for _g, _ds in sel_groups.items():
        for _d in _ds:
            _e = _d["experiment"]
            _n, _f, _p, _h = bench_row(_d)
            _res = _d.get("results")
            _rows.append(
                {
                    "id": _i,
                    "group": _g,
                    "name": _n,
                    "file": _f,
                    "params": _p,
                    "h": _h,
                    "seed": _e.get("seed"),
                    "ok": _res is not None,
                    "time": (_res or {}).get("time"),
                    "error": _d.get("error"),
                    "variant": _e.get("variant"),
                }
            )
            _i += 1

    runs_table = mo.ui.table(
        pd.DataFrame(_rows),
        selection="multi",
        label="Seed-runs (failures have ok=False)",
        page_size=15,
        freeze_columns_left=["id", "group", "name"],
        column_widths={
            "id": 40, "group": 110, "name": 100, "file": 150, "params": 110, "h": 35,
            "seed": 45, "ok": 40, "time": 65, "error": 70, "variant": 260,
        },
        wrapped_columns=["variant"],
    )
    runs_table
    return (runs_table,)


@app.cell
def _(
    TIME_COMPONENTS,
    flat,
    mo,
    os,
    plotly_round_breakdown,
    plotly_tree_growth,
    runs_table,
    time_components,
):
    # Details for the ticked seed-runs: method + paths, a total runtime breakdown,
    # and the per-round breakdown and tree-growth plots.
    _sel = runs_table.value
    _ids = list(_sel["id"]) if _sel is not None and len(_sel) else []


    def _method_label(e):
        parts = [e.get("learning_method") or "?"]
        if e.get("use_dont_care"):
            parts.append("dc")
        if e.get("use_refrence_language"):
            parts.append("rl")
        return "+".join(parts)


    def _round_brk(d):
        return plotly_round_breakdown({f"seed {d['experiment'].get('seed')}": d})


    def _total_brk(d):
        import plotly.graph_objects as _go
        from matplotlib import colors as _mcolors

        comp = time_components(d)
        if not comp:
            return mo.md("_no timing breakdown available_")
        _fig = _go.Figure()
        for _cn, _cc in TIME_COMPONENTS:
            _v = comp.get(_cn, 0.0) or 0.0
            if _v <= 0:
                continue
            _fig.add_trace(
                _go.Bar(
                    x=[_v],
                    y=[""],
                    orientation="h",
                    name=_cn,
                    marker_color=_mcolors.to_hex(_cc),
                    hovertemplate=f"{_cn}: %{{x:.2f}} s<extra></extra>",
                )
            )
        _fig.update_layout(
            barmode="stack",
            height=170,
            margin=dict(l=10, r=10, t=40, b=30),
            xaxis_title="seconds",
            template="plotly_white",
            legend=dict(orientation="h"),
            title=f"total {sum(v or 0 for v in comp.values()):.2f} s",
        )
        return _fig


    if not _ids:
        _detail = mo.md("_tick one or more rows above to see their details_")
    else:
        _blocks = []
        for _pos, _i in enumerate(_ids):
            _d = flat[int(_i)]
            _e = _d["experiment"]
            _res = _d.get("results") or {}
            _dot = _res.get("dot_file")
            _dot_line = f"- monitor dot: `{os.path.abspath(_dot)}`\n" if _dot else ""
            _parts = [
                mo.md(
                    f"### {_pos + 1}. **{_e.get('name')}** · method `{_method_label(_e)}` · "
                    f"`{_e.get('file')}` · h={_e.get('horizon')} · seed={_e.get('seed')}\n\n"
                    f"- variant: `{_e.get('variant')}`\n"
                    f"- json: `{_d.get('json_path')}`\n"
                    f"- log: `{_d.get('log_path')}`\n"
                    f"{_dot_line}"
                    f"- error: {_d.get('error')}"
                )
            ]
            for _label, _fn in (
                ("Total time breakdown", _total_brk),
                ("Round breakdown", _round_brk),
                ("Tree growth", plotly_tree_growth),
            ):
                try:
                    _fig = _fn(_d)
                    _parts.append(mo.vstack([mo.md(f"**{_label}**"), _fig]))
                except Exception as _ex:
                    _parts.append(mo.md(f"_{_label}: not available ({_ex})_"))
            _blocks.append(mo.vstack(_parts))
            if _pos < len(_ids) - 1:
                _blocks.append(mo.md("---"))
        _detail = mo.vstack(_blocks)
    _detail
    return


if __name__ == "__main__":
    app.run()
