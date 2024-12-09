from copy import deepcopy
from time import time
from typing import Literal

from paynt.family.family import Family

from stormpy import model_checking, parse_properties, SparseDtmc, ExpressionManager
from stormvogel.mapping import stormvogel_to_stormpy, stormpy_to_stormvogel
from stormvogel.model import Model
from stormvogel.show import show

from verimon import loaders
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
    "paynt_strategy": "ar",
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
    if res is None and threshold is None:
        return 1, None, None, None, None
    elif res is None:
        return None
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
        return 0, None, None, None, None
    elif res is None:
        return None
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
        return 1, None, None, None, None
    elif res is None:
        return None
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
        return 0, None, None, None, None
    elif res is None:
        return None
    return res


def _verify_helper(
    mc: SparseDtmc,
    mon_cycl: Model,
    paynt_spec: str,
    horizon: int,
    expr_manager: ExpressionManager,
    options=None,
) -> None | tuple[float, list[str], Family, Verifier, dict[str, float]]:
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

    logger.debug("Finding specified trace")
    paynt_start = time()
    assignment = model.check_paynt_prop(paynt_spec, options["relative_error"])
    stats["paynt_time"] = time() - paynt_start

    if assignment is None:
        logger.info("no counter example during verification")
        return None

    trace = model.trace_of_assignment(assignment)
    logger.info(f"Found trace: {trace}")

    induced_mc = model.created_induced_mc(assignment)
    # with open(f"models/inducedmc-{paynt_spec}.dot", "w") as f:
    #     f.write(induced_mc.to_dot())
    result_goal: float = model_checking(
        induced_mc, parse_properties('Pmax=? [F "goal"]')[0]
    ).at(induced_mc.initial_states[0])
    result_stop: float = model_checking(
        induced_mc, parse_properties('Pmax=? [F "stop"]')[0]
    ).at(induced_mc.initial_states[0])
    logger.info(f"Goal probability counterexample: {result_goal}")

    if abs(1 - (result_stop + result_goal)) > 0.05:
        raise Exception(
            "Inconsistent scheduler found during maximisation, this should not happen",
            result_stop,
            result_goal,
        )

    return result_goal, trace, assignment, model, stats
