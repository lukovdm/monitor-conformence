import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Any, Literal

import stormpy
from paynt.family.family import Family
from stormpy import (
    ExpressionManager,
    Rational,
    SparseDtmc,
    SparseExactDtmc,
    SparseExactMdp,
    SparseMdp,
    export_to_drn,
    model_checking,
    parse_properties,
)

from tover.core.synthesis import ConditionalMethod, Verifier
from tover.core.transformations import stormpy_unroll
from tover.models.algorithms import complement_monitor
from tover.utils.logger import logger


@dataclass
class VerifyStats:
    product_time: float = 0.0
    paynt_time: float = 0.0
    iterations: int = 0
    value: float | None = None
    double_check_time: float | None = None

    def __iadd__(self, other: "VerifyStats") -> "VerifyStats":
        self.product_time += other.product_time
        self.paynt_time += other.paynt_time
        self.iterations += other.iterations
        if other.double_check_time is not None:
            self.double_check_time = (
                self.double_check_time or 0.0
            ) + other.double_check_time
        return self


default_option = {
    "good_spec": 'P>0.9 [ "good" ]',
    "good_label": "good",
    "relative_error": 0.1,
    "use_risk": False,
    "paynt_strategy": "ar",
    "conditional_method": ConditionalMethod.REJECTION,
}

VerifyResult = tuple[
    tuple[float, list[str], Family] | None,
    Verifier,
    VerifyStats,
]


def false_positive(
    mc: SparseDtmc | SparseExactDtmc,
    mon: SparseMdp | SparseExactMdp,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options: dict | None = None,
) -> VerifyResult:
    """Find a trace that is a false positive: accepted by the monitor but bad."""
    return _rate("fp", mc, mon, horizon, expr_manager, threshold, options)


def false_negative(
    mc: SparseDtmc | SparseExactDtmc,
    mon: SparseMdp | SparseExactMdp,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options: dict | None = None,
) -> VerifyResult:
    """Find a trace that is a false negative: rejected by the monitor but good."""
    return _rate("fn", mc, mon, horizon, expr_manager, threshold, options)


def true_positive(
    mc: SparseDtmc | SparseExactDtmc,
    mon: SparseMdp | SparseExactMdp,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options: dict | None = None,
) -> VerifyResult:
    """Find a trace that is a true positive: accepted by the monitor and good."""
    return _rate("tp", mc, mon, horizon, expr_manager, threshold, options)


def true_negative(
    mc: SparseDtmc | SparseExactDtmc,
    mon: SparseMdp | SparseExactMdp,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options: dict | None = None,
) -> VerifyResult:
    """Find a trace that is a true negative: rejected by the monitor and bad."""
    return _rate("tn", mc, mon, horizon, expr_manager, threshold, options)


def _rate(
    kind: Literal["fp", "fn", "tp", "tn"],
    mc: SparseDtmc | SparseExactDtmc,
    mon: SparseMdp | SparseExactMdp,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options: dict[str, Any] | None = None,
) -> VerifyResult:
    """Parameterised core for all four verification functions.

    kind determines:
      - which goal label to maximise toward ("goal" or "stop")
      - whether to complement the monitor (fn/tn use the complement)
      - the threshold direction and the default result when no CEX is found
    """
    if options is None:
        options = {}

    conditional_spec = (
        ' || F "condition"' if options.get("conditional_method") != "rejection" else ""
    )

    use_goal_label = kind in ("fn", "tp")
    use_complement = kind in ("fn", "tn")
    # fp/tp: threshold is 1-threshold (we need PAYNT to find strategies above that)
    # fn/tn: threshold is the raw threshold
    threshold_is_inverted = kind in ("fp", "tp")
    # When no CEX is found and threshold is None, return the best-case default
    default_when_no_cex = 0 if kind in ("fn", "tn") else 1

    label = "goal" if use_goal_label else "stop"

    if threshold is None:
        paynt_spec = f'Pmax=? [F "{label}"{conditional_spec}]'
    elif threshold_is_inverted:
        paynt_spec = f'P>{1 - threshold} [F "{label}"{conditional_spec}]'
    else:
        paynt_spec = f'P>{threshold} [F "{label}"{conditional_spec}]'

    if use_complement:
        mon = complement_monitor(mon, "accepting")

    cex, model, stats = _verify_helper(
        mc, mon, paynt_spec, horizon, expr_manager, default_option | options
    )

    if cex is None and threshold is None:
        return (default_when_no_cex, [], None), model, stats  # type: ignore[return-value]

    return cex, model, stats


def _verify_helper(
    mc: SparseDtmc | SparseExactDtmc,
    mon: SparseMdp | SparseExactMdp,
    paynt_spec: str,
    horizon: int,
    expr_manager: ExpressionManager,
    options: dict | None = None,
) -> VerifyResult:
    if options is None:
        options = {}

    stats = VerifyStats()

    logger.info("Building model")

    product_start = time()
    mon_unrolled = stormpy_unroll(mon, horizon)
    logger.debug("Unrolling done")

    model = Verifier(
        mc,
        mon_unrolled,
        expr_manager,
        options["good_label"],
        options["paynt_strategy"],
        options.get("export_benchmarks", False),
        options.get("conditional_method", "rejection"),
    )
    if options["use_risk"]:
        model.set_risk(options["good_spec"])
        logger.debug("Apply risk done")
    else:
        model.apply_spec(options["good_spec"])
        logger.debug("Apply spec done")

    model.create_product()
    stats.product_time = time() - product_start
    logger.debug("creating product done")

    if "model_path" in options and options["export_benchmarks"]:
        os.makedirs(options["model_path"], exist_ok=True)
        timestamp = time()
        path = (
            f"{options['model_path']}/pomdp-tnull-itnull"
            f"-st{len(model.pomdp.states)}"
            f"-{paynt_spec.replace('/', ' div ')}"
            f"-{options.get('hash', 'nohash')}-{timestamp}.drn"
        )
        export_to_drn(model.pomdp, path)

    logger.debug("Finding specified trace")
    paynt_start = time()
    assignment, value, iterations = model.check_paynt_prop(
        paynt_spec, options["relative_error"]
    )
    stats.paynt_time = time() - paynt_start
    stats.iterations = iterations

    if "model_path" in options and options["export_benchmarks"]:
        try:
            os.rename(
                path,
                f"{options['model_path']}/pomdp"
                f"-t{stats.paynt_time:.3f}"
                f"-it{stats.iterations}"
                f"-st{len(model.pomdp.states)}"
                f"-{paynt_spec.replace('/', ' div ')}"
                f"-{options.get('hash', 'nohash')}-{timestamp}.drn",
            )
        except Exception:
            logger.error(f"Could not rename benchmark file: {traceback.format_exc()}")

    if assignment is None:
        logger.info("no counter example during verification")
        return None, model, stats
    else:
        stats.value = value

    trace = model.trace_of_assignment(assignment)
    logger.info(f"Found trace: {trace}")

    double_check_time = time()
    induced_mc = model.created_induced_mc(assignment)

    env = stormpy.Environment()
    env.solver_environment.set_linear_equation_solver_type(
        stormpy.EquationSolverType.eigen
    )
    env.solver_environment.minmax_solver_environment.method = (
        stormpy.MinMaxMethod.policy_iteration
    )
    env.solver_environment.native_solver_environment.precision = Rational(str(1e-6))
    env.solver_environment.minmax_solver_environment.precision = Rational(str(1e-6))

    conditional_prop = (
        ' || F "condition"' if options.get("conditional_method") != "rejection" else ""
    )

    result_goal: float = model_checking(
        induced_mc,
        parse_properties(f'Pmax=? [F "goal"{conditional_prop}]')[0],
        environment=env,
    ).at(induced_mc.initial_states[0])
    result_stop: float = model_checking(
        induced_mc,
        parse_properties(f'Pmax=? [F "stop"{conditional_prop}]')[0],
        environment=env,
    ).at(induced_mc.initial_states[0])
    logger.info(f"Goal probability counterexample: {result_goal}")

    if (mc.is_exact and result_stop + result_goal != 1) or (
        not mc.is_exact and abs(1 - (result_stop + result_goal)) > 0.05
    ):
        raise Exception(
            f"Inconsistent scheduler found during maximisation, this should not happen. "
            f"(Cond) prob stop: {result_stop}, (cond) prob goal: {result_goal}, "
            f"together: {result_stop + result_goal}."
        )

    if value:
        if "goal" in paynt_spec:
            diff = abs(value - result_goal)
        else:
            diff = abs(value - result_stop)

        if (mc.is_exact and diff != 0) or (not mc.is_exact and diff > 0.05):
            logger.warning(
                f"paynt value and checking value differ: {value} vs "
                f"goal:{result_goal} or stop:{result_stop}"
            )
            if "model_path" in options:
                os.makedirs(options["model_path"], exist_ok=True)
                export_to_drn(
                    induced_mc,
                    f"{options['model_path']}/induced-chck-{datetime.now()}.drn",
                )

        if "filtering" in options:
            logger.info("Checking results using filtering")
            res_filtering = options["filtering"].steps(trace)
            result_goal = res_filtering
            logger.info(f"Filtering result: {res_filtering}")
            if "stop" in paynt_spec:
                value = 1 - value

            diff_filtering = abs(res_filtering - value)
            if (mc.is_exact and diff_filtering != 0) or (
                not mc.is_exact and diff_filtering > 0.05
            ):
                logger.warning(
                    f"Paynt value and filtering value differ: {value} vs {res_filtering}"
                )
                if "model_path" in options:
                    os.makedirs(options["model_path"], exist_ok=True)
                    export_to_drn(
                        induced_mc,
                        f"{options['model_path']}/induced-filter-{datetime.now()}.drn",
                    )

    stats.double_check_time = time() - double_check_time

    return (result_goal, trace, assignment), model, stats
