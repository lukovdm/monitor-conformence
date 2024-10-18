from copy import deepcopy

from stormpy import model_checking, parse_properties
from stormvogel.model import Model
from stormvogel.show import show

from verimon.MDP_product import MC_MON_Product
from verimon.algs import complement_model
from verimon.unrolling import prune_monitor, simulator_unroll
from verimon.utils import logger

default_option = {
    "good_spec": 'P>0.9 [ "good" ]',
    "good_label": "good",
    "relative_error": 0.1,
    "show_monitor": False,
}


def false_positive(
    mc: Model,
    mon: Model,
    gb: Model,
    horizon: int,
    threshold: float | None = None,
    options=None,
):

    if options is None:
        options = {}

    if threshold is None:
        paynt_spec = 'Pmax=? [F "stop"]'
    else:
        paynt_spec = f'P>{threshold} [F "stop"]'

    return _verify_helper(mc, mon, gb, paynt_spec, horizon, default_option | options)


def false_negative(
    mc: Model,
    mon: Model,
    gb: Model,
    horizon: int,
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

    return _verify_helper(mc, mon_c, gb, paynt_spec, horizon, default_option | options)


def true_positive(
    mc: Model,
    mon: Model,
    gb: Model,
    horizon: int,
    threshold: float | None = None,
    options=None,
):

    if options is None:
        options = {}

    if threshold is None:
        paynt_spec = 'Pmax=? [F "goal"]'
    else:
        paynt_spec = f'P>{threshold} [F "goal"]'

    return _verify_helper(mc, mon, gb, paynt_spec, horizon, default_option | options)


def true_negative(
    mc: Model,
    mon: Model,
    gb: Model,
    horizon: int,
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

    return _verify_helper(mc, mon_c, gb, paynt_spec, horizon, default_option | options)


def _verify_helper(
    mc: Model,
    mon_cycl: Model,
    gb: Model,
    paynt_spec: str,
    horizon: int,
    options=None,
):
    if options is None:
        options = {}

    logger.debug("Building model")
    mon = simulator_unroll(mon_cycl, horizon)
    logger.debug("Unrolling done")
    prune_monitor(mon)
    if options["show_monitor"]:
        show(mon)
    logger.debug("Pruning done")
    model = MC_MON_Product(mc, mon, gb, options["good_label"])
    model.apply_spec(options["good_spec"])
    logger.debug("Apply spec done")
    model.create_product(use_step_label=True)
    logger.debug("creating product done")

    logger.debug("Creating trace")
    assignment = model.check_paynt_prop(paynt_spec, options["relative_error"])

    if assignment is None:
        logger.info("no counter example during verification")
        return None

    induced_mc = model.created_induced_mc(assignment)

    trace = model.trace_of_assignment(assignment)
    logger.info(f"Found trace: {trace}")

    result_goal: float = model_checking(
        induced_mc, parse_properties('Pmax=? [F "goal"]')[0]
    ).at(induced_mc.initial_states[0])
    result_stop: float = model_checking(
        induced_mc, parse_properties('Pmax=? [F "stop"]')[0]
    ).at(induced_mc.initial_states[0])
    logger.info(f"Goal probability counterexample: {result_goal}")

    if abs(1 - (result_stop + result_goal)) > 0.01:
        raise Exception(
            "Inconsistent scheduler found during maximisation, this should not happen",
            result_stop,
            result_goal,
        )

    return result_goal, trace, assignment, model
