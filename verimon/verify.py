from copy import deepcopy
import os
from time import time

from paynt.family.family import Family

import stormpy
from stormpy import (
    model_checking,
    parse_properties,
    SparseDtmc,
    ExpressionManager,
    export_to_drn,
    Rational,
)
from stormvogel.mapping import stormvogel_to_stormpy
from stormvogel.model import Model

from verimon.algs import complement_model
from verimon.generator import Verifier
from verimon.logger import logger
from verimon.transformations import (
    prune_monitor,
    simulator_unroll,
)

default_option = {
    "good_spec": 'P>0.9 [ "good" ]',
    "good_label": "good",
    "relative_error": 0.1,
    "use_risk": False,
    "paynt_strategy": "cegis",
}


def false_positive(
    mc: SparseDtmc,
    mon: Model,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options=None,
):

    if options is None:
        options = {}

    if threshold is None:
        paynt_spec = 'Pmax=? [F "stop"]'
    else:
        paynt_spec = f'P>{1 - threshold} [F "stop"]'

    res = _verify_helper(
        mc, mon, paynt_spec, horizon, expr_manager, default_option | options
    )
    if res[0] is None and threshold is None:
        return 1, res[1], res[2], res[3], res[4]
    elif res is None:
        return res
    return res


def false_negative(
    mc: SparseDtmc,
    mon: Model,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options=None,
):

    if options is None:
        options = {}

    if threshold is None:
        paynt_spec = 'Pmax=? [F "goal"]'
    else:
        paynt_spec = f'P>{threshold} [F "goal"]'

    mon_c = deepcopy(mon)
    complement_model(mon_c, "accepting")

    res = _verify_helper(
        mc, mon_c, paynt_spec, horizon, expr_manager, default_option | options
    )
    if res is None and threshold is None:
        return 0, res[1], res[2], res[3], res[4]
    elif res is None:
        return res
    return res


def true_positive(
    mc: SparseDtmc,
    mon: Model,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options=None,
):

    if options is None:
        options = {}

    if threshold is None:
        paynt_spec = 'Pmax=? [F "goal"]'
    else:
        paynt_spec = f'P>{1 - threshold} [F "goal"]'

    res = _verify_helper(
        mc, mon, paynt_spec, horizon, expr_manager, default_option | options
    )
    if res is None and threshold is None:
        return 1, res[1], res[2], res[3], res[4]
    elif res is None:
        return res
    return res


def true_negative(
    mc: SparseDtmc,
    mon: Model,
    horizon: int,
    expr_manager: ExpressionManager,
    threshold: float | None = None,
    options=None,
):
    if options is None:
        options = {}

    if threshold is None:
        paynt_spec = 'Pmax=? [F "stop"]'
    else:
        paynt_spec = f'P>{threshold} [F "stop"]'

    mon_c = deepcopy(mon)
    complement_model(mon_c, "accepting")

    res = _verify_helper(
        mc, mon_c, paynt_spec, horizon, expr_manager, default_option | options
    )
    if res is None and threshold is None:
        return 0, res[1], res[2], res[3], res[4]
    elif res is None:
        return res
    return res


def _verify_helper(
    mc: SparseDtmc,
    mon_cycl: Model,
    paynt_spec: str,
    horizon: int,
    expr_manager: ExpressionManager,
    options=None,
) -> tuple[float | None, list[str] | None, Family | None, Verifier, dict[str, float]]:
    if options is None:
        options = {}

    stats = {"product_time": 0.0, "paynt_time": 0.0}

    logger.debug("Building model")

    mon_unroll = simulator_unroll(mon_cycl, horizon)
    logger.debug("Unrolling done")

    # try:
    #     prune_monitor(mon_unroll)
    # except RuntimeError as e:
    #     raise Exception(
    #         "Monitor horizon probably not deep enough, no accepting states in monitor",
    #         e,
    #     )

    # logger.debug("Pruning done")

    mon = stormvogel_to_stormpy(mon_unroll)
    # mon = bisim_minimise_monitor(mon) Bisimulation minimisation removes choice labeling, whichi is essential for us
    model = Verifier(
        mc, mon, expr_manager, options["good_label"], options["paynt_strategy"]
    )
    if options["use_risk"]:
        model.set_risk(options["good_spec"])
        logger.debug("Apply risk done")
    else:
        model.apply_spec(options["good_spec"])
        logger.debug("Apply spec done")

    product_start = time()
    model.create_product()
    stats["product_time"] = time() - product_start
    logger.debug("creating product done")

    if "model_path" in options:
        os.makedirs(options["model_path"], exist_ok=True)
        path = f"{options['model_path']}/pomdp-null-{len(model.pomdp.states)}-{paynt_spec}.drn"
        export_to_drn(
            model.pomdp,
            path,
        )

    logger.debug("Finding specified trace")
    paynt_start = time()
    res = model.check_paynt_prop(paynt_spec, options["relative_error"])
    stats["paynt_time"] = time() - paynt_start

    if "model_path" in options:
        try:
            os.rename(
                path,
                f"{options['model_path']}/pomdp-{stats['paynt_time']:.3f}-{len(model.pomdp.states)}-{paynt_spec}.drn",
            )
        except Exception as e:
            logger.error(f"Could not rename file: {e}")

    if res is None:
        logger.info("no counter example during verification")
        return None, None, None, model, stats
    else:
        assignment, value = res
        stats["value"] = value

    trace = model.trace_of_assignment(assignment)
    logger.info(f"Found trace: {trace}")

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

    result_goal: float = model_checking(
        induced_mc, parse_properties('Pmax=? [F "goal"]')[0], environment=env
    ).at(induced_mc.initial_states[0])
    result_stop: float = model_checking(
        induced_mc, parse_properties('Pmax=? [F "stop"]')[0], environment=env
    ).at(induced_mc.initial_states[0])
    logger.info(f"Goal probability counterexample: {result_goal}")

    if abs(1 - (result_stop + result_goal)) > 0.05:
        raise Exception(
            "Inconsistent scheduler found during maximisation, this should not happen",
            result_stop,
            result_goal,
        )

    if value:
        if "goal" in paynt_spec:
            diff = abs(value - result_goal)
        else:
            diff = abs(value - result_stop)

        if diff > 0.05:
            if "model_path" in options:
                os.makedirs(options["model_path"], exist_ok=True)
                export_to_drn(
                    induced_mc,
                    f"{options['model_path']}/induced-chck-{value}-{diff}.drn",
                )
            logger.warning(
                f"paynt value and checking value differ: {value} vs goal:{result_goal} or stop:{result_stop}"
            )

        if "filtering" in options:
            logger.info("Checking results using filtering")
            res_filtering = options["filtering"].steps(trace)
            result_goal = res_filtering
            logger.info(f"Filtering result: {res_filtering}")
            if "stop" in paynt_spec:
                value = 1 - value

            diff_filtering = abs(res_filtering - value)
            if diff_filtering > 0.05:
                if "model_path" in options:
                    os.makedirs(options["model_path"], exist_ok=True)
                    export_to_drn(
                        induced_mc,
                        f"{options['model_path']}/induced-filter-{value}-{diff_filtering}.drn",
                    )
                logger.warning(
                    f"Paynt value and filtering value differ: {value} vs {res_filtering}"
                )

    return result_goal, trace, assignment, model, stats
