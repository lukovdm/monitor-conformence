from math import sqrt
from random import randrange

from stormpy import (
    PrismProgram,
    PrismConstant,
    parse_prism_program,
    BuilderOptions,
    build_sparse_model_with_options,
    build_sparse_exact_model_with_options,
    SparseDtmc,
    SparseExactDtmc,
    ExpressionManager,
)
from stormvogel.mapping import stormpy_to_stormvogel
from stormvogel.model import Model, new_mdp

# Path to the uncontrollable Snakes & Ladders PRISM template
SNL_MC_PATH = "benchmarks/models/snake_ladder/mc_u_nxn.pm"


def random_snl_board(n: int) -> tuple[int, dict[int, int], dict[int, int]]:
    """Generate a random Snakes & Ladders board with n squares."""

    def random_ladder(n):
        source = randrange(1, n - int(sqrt(n)))
        dest = randrange(source, int(min(n, source + n / 2)))
        return source, dest

    def random_snake(n):
        source = randrange(int(sqrt(n)) + 1, n)
        dest = randrange(1, source)
        return source, dest

    ladders = {1: 1}
    snakes = {1: 1}
    while not set(ladders.keys()).isdisjoint(snakes.keys()):
        ladders = dict(random_ladder(n) for _ in range(int(sqrt(n))))
        snakes = dict(random_snake(n) for _ in range(int(sqrt(n))))

    return n, ladders, snakes


def load_snl(path: str, n: int, ladders: dict[int, int], snakes: dict[int, int]) -> Model:
    snl_prism: PrismProgram = parse_prism_program(path)
    if not snl_prism.has_undefined_constants:
        raise Exception("Model is already fully defined")
    snl_prism = _define_snl_constants(snl_prism, n, ladders, snakes)
    return _load_prism(snl_prism)


def load_snl_stormpy(
    path: str, n: int, ladders: dict[int, int], snakes: dict[int, int], use_exact=False
) -> tuple[SparseDtmc | SparseExactDtmc, ExpressionManager]:
    snl_prism: PrismProgram = parse_prism_program(path)
    if not snl_prism.has_undefined_constants:
        raise Exception("Model is already fully defined")
    prism = _define_snl_constants(snl_prism, n, ladders, snakes)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()

    if use_exact:
        return build_sparse_exact_model_with_options(prism, options), snl_prism.expression_manager
    else:
        return build_sparse_model_with_options(prism, options), snl_prism.expression_manager


def load_defined_snl(path: str) -> tuple[Model, int, dict[int, int], dict[int, int]]:
    mc_prism = parse_prism_program(path)
    n, ladders, snakes = _get_sl_prism_consts(mc_prism)
    mc = _load_prism(mc_prism)
    return mc, n, ladders, snakes


def _define_snl_constants(
    snl_prism: PrismProgram, n: int, ladders: dict[int, int], snakes: dict[int, int]
):
    ladders_list = list(ladders.items())
    snakes_list = list(snakes.items())
    mapping = {}
    const: PrismConstant
    for const in snl_prism.constants:
        if const.defined:
            continue
        elif const.name == "n":
            value = snl_prism.expression_manager.create_integer(n)
        elif const.name.startswith("l"):
            if int(const.name[1:-1]) > len(ladders_list):
                value = snl_prism.expression_manager.create_integer(-1)
            else:
                value = snl_prism.expression_manager.create_integer(
                    int(ladders_list[int(const.name[1:-1]) - 1][0 if const.name[2] == "s" else 1])
                )
        elif const.name.startswith("s"):
            if int(const.name[1:-1]) > len(snakes_list):
                value = snl_prism.expression_manager.create_integer(-1)
            else:
                value = snl_prism.expression_manager.create_integer(
                    int(snakes_list[int(const.name[1:-1]) - 1][0 if const.name[2] == "s" else 1])
                )
        else:
            continue
        mapping[const.expression_variable] = value
    print(", ".join([f"{var.name}={val}" for var, val in mapping.items()]))
    return snl_prism.define_constants(mapping)


def _load_prism(prism: PrismProgram, add_valuation=True) -> Model:
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    options.set_build_observation_valuations()
    model_stormpy = build_sparse_model_with_options(prism, options)
    model = stormpy_to_stormvogel(model_stormpy)
    if model is None:
        raise Exception("Could not build model")
    if add_valuation:
        _add_valuation_to_sv_labels(model_stormpy, model)
    return model


def _add_valuation_to_sv_labels(spy, sv: Model):
    for s in spy.states:
        sv.states[s.id].add_label(s.valuations)


def _get_sl_prism_consts(model) -> tuple[int, dict[int, int], dict[int, int]]:
    n = next(c.definition.evaluate_as_int() for c in model.constants if c.name == "n")
    ladders_list = [[0, 0] for _ in range(n)]
    snakes_list = [[0, 0] for _ in range(n)]

    for c in model.constants:
        if c.name.startswith("l"):
            ladders_list[int(c.name[1])][0 if c.name[2] == "s" else 1] = (
                c.definition.evaluate_as_int()
            )
        elif c.name.startswith("s"):
            snakes_list[int(c.name[1])][0 if c.name[2] == "s" else 1] = (
                c.definition.evaluate_as_int()
            )

    return n, dict(ladders_list), dict(snakes_list)
