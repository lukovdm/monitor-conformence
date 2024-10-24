from math import sqrt
from random import randrange

from stormpy import (
    PrismProgram,
    PrismConstant,
    preprocess_symbolic_input,
    parse_prism_program,
    BuilderOptions,
    build_sparse_model_with_options,
    SparseDtmc,
    SparseMdp,
)
from stormvogel.mapping import stormpy_to_stormvogel
from stormvogel.model import Model, EmptyAction, Branch, ModelType, new_mdp


def random_snl_board(n: int):
    def random_ladder(n):
        source = randrange(1, n - int(sqrt(n)))
        dest = randrange(source, int(min(n, source + n / 2)))
        return source, dest

    def random_snake(n):
        source = randrange(int(sqrt(n)) + 1, n)
        dest = randrange(1, source)
        return source, dest

    ladders = dict(random_ladder(n) for _ in range(int(sqrt(n))))
    snakes = dict(random_snake(n) for _ in range(int(sqrt(n))))
    return n, ladders, snakes


def load_snl(path: str, n: int, ladders: dict[int, int], snakes: dict[int, int]):
    snl_prism: PrismProgram = parse_prism_program(path)
    if not snl_prism.has_undefined_constants:
        raise Exception("Model is already fully defined")
    snl_prism = _define_snl_constants(snl_prism, n, ladders, snakes)
    return _load_prism(snl_prism)


def load_snl_stormpy(
    path: str, n: int, ladders: dict[int, int], snakes: dict[int, int]
):
    snl_prism: PrismProgram = parse_prism_program(path)
    if not snl_prism.has_undefined_constants:
        raise Exception("Model is already fully defined")
    prism = _define_snl_constants(snl_prism, n, ladders, snakes)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    return build_sparse_model_with_options(prism, options)


def load_defined_snl(path: str) -> tuple[Model, int, dict[int, int], dict[int, int]]:
    mc_prism = parse_prism_program(path)
    n, ladders, snakes = _get_sl_prism_consts(mc_prism)
    mc = _load_prism(mc_prism)
    return mc, n, ladders, snakes


def load_dfa(path: str) -> Model:
    dfa_prism = parse_prism_program(path)
    return _load_prism(dfa_prism)


def load_dfa_stormpy(path: str) -> SparseMdp:
    dfa_prism = parse_prism_program(path)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    return build_sparse_model_with_options(dfa_prism, options)


def pomdp_to_mc(path: str, constants: str = "") -> tuple[set[int], Model]:
    prism = parse_prism_program(path)
    prism, _ = preprocess_symbolic_input(prism, [], constants)
    pomdp = _load_prism(prism)
    possible_actions = set()
    for state in pomdp.states.values():
        possible_actions.add("obs" + str(state.get_observation().observation))
        state.add_label("obs" + str(state.get_observation().observation))

    for transition in pomdp.transitions.values():
        mc_branch = []
        nr_actions = len(transition.transition)
        for b in transition.transition.values():
            mc_branch += [(p / nr_actions, s) for p, s in b.branch]
        transition.transition.clear()
        transition.transition[EmptyAction] = Branch(mc_branch)

    pomdp.type = ModelType.DTMC
    return possible_actions, pomdp


def gen_monitor(action_strs: list[str], horizon: int, accepting_after=1):
    mon = new_mdp("Monitor")
    mon.get_initial_state().add_label("[step=0]")
    actions = [mon.new_action(a, frozenset([a])) for a in action_strs]
    for i in range(1, horizon):
        s = mon.new_state([f"[step={i}]"])
        if i >= accepting_after:
            s.add_label("accepting")
    mon.states[len(mon.states) - 1].add_label("horizon")

    for i, state in mon.states.items():
        if "horizon" in state.labels:
            state.set_transitions([(a, state) for a in actions])
        else:
            state.set_transitions([(a, mon.states[i + 1]) for a in actions])

    return mon


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
                    ladders_list[int(const.name[1:-1]) - 1][
                        0 if const.name[2] == "s" else 1
                    ]
                )
        elif const.name.startswith("s"):
            if int(const.name[1:-1]) > len(snakes_list):
                value = snl_prism.expression_manager.create_integer(-1)
            else:
                value = snl_prism.expression_manager.create_integer(
                    snakes_list[int(const.name[1:-1]) - 1][
                        0 if const.name[2] == "s" else 1
                    ]
                )
        else:
            continue
        mapping[const.expression_variable] = value
    return snl_prism.define_constants(mapping)


def _load_prism(prism: PrismProgram) -> Model:
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    model_stormpy = build_sparse_model_with_options(prism, options)
    model = stormpy_to_stormvogel(model_stormpy)
    if model is None:
        raise Exception("Could not build model")

    _add_valuation_to_sv_labels(model_stormpy, model)

    return model


def _add_valuation_to_sv_labels(spy: SparseDtmc | SparseMdp, sv: Model):
    for s in spy.states:
        sv.states[s.id].add_label(s.valuations)


def _get_sl_prism_consts(model) -> tuple[int, dict[int, int], dict[int, int]]:
    n = next(c.definition.evaluate_as_int() for c in model.constants if c.name == "n")
    ladders_list = [[0, 0] for _ in range(n)]
    snakes_list = [[0, 0] for _ in range(n)]

    for c in model.constants:
        if c.name.startswith("l"):
            ladders_list[int(c.name[1])][
                0 if c.name[2] == "s" else 1
            ] = c.definition.evaluate_as_int()
        elif c.name.startswith("s"):
            snakes_list[int(c.name[1])][
                0 if c.name[2] == "s" else 1
            ] = c.definition.evaluate_as_int()

    return n, dict(ladders_list), dict(snakes_list)
