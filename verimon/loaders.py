from math import sqrt
from random import randrange
from typing import Any

from aalpy import Dfa
from numpy import matrix
from stormpy import (
    PrismProgram,
    PrismConstant,
    preprocess_symbolic_input,
    parse_prism_program,
    BuilderOptions,
    build_sparse_model_with_options,
    build_sparse_exact_model_with_options,
    SparseDtmc,
    SparseExactDtmc,
    SparseMdp,
    SparseExactMdp,
    ExpressionManager,
    DirectEncodingParserOptions,
    build_model_from_drn,
    ExactSparseMatrixBuilder,
    SparseMatrixBuilder,
    StateLabeling,
    ChoiceLabeling,
    Rational,
    SparseExactModelComponents,
    SparseModelComponents,
)
import stormvogel
from stormvogel.mapping import stormpy_to_stormvogel, stormvogel_to_stormpy
from stormvogel.model import Model, EmptyAction, Branch, ModelType, new_mdp, State

from verimon.logger import logger
from verimon.utils import compact_json_str


def random_snl_board(n: int):
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


def load_snl(path: str, n: int, ladders: dict[int, int], snakes: dict[int, int]):
    snl_prism: PrismProgram = parse_prism_program(path)
    if not snl_prism.has_undefined_constants:
        raise Exception("Model is already fully defined")
    snl_prism = _define_snl_constants(snl_prism, n, ladders, snakes)
    return _load_prism(snl_prism)


def load_snl_stormpy(
    path: str, n: int, ladders: dict[int, int], snakes: dict[int, int], use_exact=False
) -> tuple[SparseDtmc, ExpressionManager]:
    snl_prism: PrismProgram = parse_prism_program(path)
    if not snl_prism.has_undefined_constants:
        raise Exception("Model is already fully defined")
    prism = _define_snl_constants(snl_prism, n, ladders, snakes)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()

    if use_exact:
        return (
            build_sparse_exact_model_with_options(prism, options),
            snl_prism.expression_manager,
        )
    else:
        return (
            build_sparse_model_with_options(prism, options),
            snl_prism.expression_manager,
        )


def load_defined_snl(path: str) -> tuple[Model, int, dict[int, int], dict[int, int]]:
    mc_prism = parse_prism_program(path)
    n, ladders, snakes = _get_sl_prism_consts(mc_prism)
    mc = _load_prism(mc_prism)
    return mc, n, ladders, snakes


def load_dfa(path: str) -> Model:
    dfa_prism = parse_prism_program(path)
    return _load_prism(dfa_prism)


def load_dfa_drn(path: str) -> Model:
    opts = DirectEncodingParserOptions()
    opts.build_choice_labels = True
    dfa_storm = build_model_from_drn(path, opts)
    dfa = stormpy_to_stormvogel(dfa_storm)
    if dfa is None:
        raise Exception("Could not build model")
    return dfa


def load_dfa_stormpy(path: str) -> SparseMdp:
    dfa_prism = parse_prism_program(path)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    return build_sparse_model_with_options(dfa_prism, options)


def load_dfa_stormpy_exact(path: str) -> SparseMdp:
    dfa_prism = parse_prism_program(path)
    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    return build_sparse_exact_model_with_options(dfa_prism, options)


def pomdp_to_stormpy_mc(
    path: str, constants: str, use_exact: bool
) -> tuple[str, set[str], SparseDtmc | SparseExactDtmc, ExpressionManager]:
    prism = parse_prism_program(path)
    symb, _ = preprocess_symbolic_input(prism, [], constants)
    symb = symb.as_prism_program()

    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    options.set_build_observation_valuations()
    if use_exact:
        pomdp_stormpy = build_sparse_exact_model_with_options(symb, options)
    else:
        pomdp_stormpy = build_sparse_model_with_options(symb, options)

    state_labeling = StateLabeling(len(pomdp_stormpy.states))
    for label in pomdp_stormpy.labeling.get_labels():
        state_labeling.add_label(label)

    # Get observation classes and and them as labels
    observation_classes: set[str] = set()
    initial_observation = compact_json_str(
        str(
            pomdp_stormpy.observation_valuations.get_json(
                pomdp_stormpy.get_observation(pomdp_stormpy.initial_states[0])
            )
        )
    )

    for state in pomdp_stormpy.states:
        obs_string = compact_json_str(
            str(
                pomdp_stormpy.observation_valuations.get_json(
                    pomdp_stormpy.get_observation(state.id)
                )
            )
        )
        observation_classes.add(obs_string)

    for obs in observation_classes:
        state_labeling.add_label(obs)

    # Create transition matrix
    if use_exact:
        builder = ExactSparseMatrixBuilder(0, 0, 0, False, False)
    else:
        builder = SparseMatrixBuilder(0, 0, 0, False, False)
    for s in pomdp_stormpy.states:
        # Set labels
        for label in s.labels:
            state_labeling.add_label_to_state(label, s.id)

        state_labeling.add_label_to_state(
            compact_json_str(
                str(
                    pomdp_stormpy.observation_valuations.get_json(
                        pomdp_stormpy.get_observation(s.id)
                    )
                )
            ),
            s.id,
        )

        # Set transition
        amount_of_actions = len(s.actions)
        new_row_dict: dict[int, Any] = {}
        for action in s.actions:
            for transition in action.transitions:
                dest_s = transition.column
                if dest_s in new_row_dict:
                    new_row_dict[dest_s] += transition.value() / amount_of_actions
                else:
                    new_row_dict[dest_s] = transition.value() / amount_of_actions

        for new_dest_s, value in sorted(new_row_dict.items()):
            builder.add_next_value(s.id, new_dest_s, value)

    matrix = builder.build(overridden_column_count=len(pomdp_stormpy.states))

    if use_exact:
        components = SparseExactModelComponents(matrix, state_labeling)
        return (
            initial_observation,
            observation_classes,
            SparseExactDtmc(components),
            prism.expression_manager,
        )
    else:
        components = SparseModelComponents(matrix, state_labeling)
        return (
            initial_observation,
            observation_classes,
            SparseDtmc(components),
            prism.expression_manager,
        )


def pomdp_to_mc(
    path: str, constants: str
) -> tuple[str, set[str], SparseDtmc, ExpressionManager]:
    prism = parse_prism_program(path)
    symb, _ = preprocess_symbolic_input(prism, [], constants)
    symb = symb.as_prism_program()

    options = BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    options.set_build_observation_valuations()
    model_stormpy = build_sparse_model_with_options(symb, options)
    model = stormpy_to_stormvogel(model_stormpy)
    logger.debug(
        f"Finished loading original pomdp with {model_stormpy.nr_observations} observations"
    )

    observation_classes: set[str] = set()
    initial_observation = compact_json_str(
        str(
            model_stormpy.observation_valuations.get_json(
                model.get_initial_state().observation.get_observation()
            )
        )
    )

    for state in model.states.values():
        obs_string = compact_json_str(
            str(
                model_stormpy.observation_valuations.get_json(
                    state.observation.get_observation()
                )
            )
        )
        observation_classes.add(obs_string)
        state.add_label(obs_string)

    logger.debug("Finished assigning labels to states")
    for transition in model.transitions.values():
        mc_trans_prob: dict[int, float] = {}
        nr_actions = len(transition.transition)
        for b in transition.transition.values():
            for p, s in b.branch:
                if not type(p) is float:
                    continue
                if s.id in mc_trans_prob:
                    mc_trans_prob[s.id] += p / nr_actions
                else:
                    mc_trans_prob[s.id] = p / nr_actions
        transition.transition.clear()
        transition.transition[EmptyAction] = Branch(
            [(p, model.states[s]) for s, p in mc_trans_prob.items()]
        )
    logger.debug("Finished creating new transitions")

    model.type = ModelType.DTMC
    dtmc = stormvogel_to_stormpy(model)
    logger.debug("transformed POMDP to stormpy DTMC")
    return initial_observation, observation_classes, dtmc, prism.expression_manager


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
                    int(
                        ladders_list[int(const.name[1:-1]) - 1][
                            0 if const.name[2] == "s" else 1
                        ]
                    )
                )
        elif const.name.startswith("s"):
            if int(const.name[1:-1]) > len(snakes_list):
                value = snl_prism.expression_manager.create_integer(-1)
            else:
                value = snl_prism.expression_manager.create_integer(
                    int(
                        snakes_list[int(const.name[1:-1]) - 1][
                            0 if const.name[2] == "s" else 1
                        ]
                    )
                )
        else:
            continue
        mapping[const.expression_variable] = value
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


def _add_valuation_to_sv_labels(spy: SparseDtmc | SparseMdp, sv: Model):
    for s in spy.states:
        sv.states[s.id].add_label(s.valuations)
        # sv.states[s.id].add_label(str(s.id))


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


def aalpy_dfa_to_stormvogel(dfa_a: Dfa):
    dfa_sv = new_mdp()

    action_mapping = {}
    for act in dfa_a.get_input_alphabet():
        action_mapping[act] = dfa_sv.new_action(act, frozenset({act}))

    state_mapping = {dfa_a.initial_state: dfa_sv.get_initial_state()}
    dfa_sv.get_initial_state().add_label(dfa_a.initial_state.state_id)
    if dfa_a.initial_state.is_accepting:
        dfa_sv.get_initial_state().add_label("accepting")

    for s in sorted(dfa_a.states, key=lambda q: q.state_id):
        if s == dfa_a.initial_state:
            continue

        s_sv = dfa_sv.new_state(s.state_id)
        state_mapping[s] = s_sv
        if s.is_accepting:
            s_sv.add_label("accepting")

    for s in sorted(dfa_a.states, key=lambda q: q.state_id):
        for act, dest_s in s.transitions.items():
            state_mapping[s].add_transitions(
                [(action_mapping[act], state_mapping[dest_s])]
            )

    return dfa_sv


def aalpy_dfa_to_stormpy(dfa_a: Dfa, use_exact: bool):
    state_labeling = StateLabeling(len(dfa_a.states))
    state_labeling.add_label("accepting")
    state_labeling.add_label("init")

    choice_labeling = ChoiceLabeling(sum([len(s.transitions) for s in dfa_a.states]))

    for label in dfa_a.get_input_alphabet():
        choice_labeling.add_label(label)

    state_mapping = {
        dfa_s.state_id: i
        for i, dfa_s in enumerate(sorted(dfa_a.states, key=lambda q: q.state_id))
    }

    state_labeling.add_label_to_state(
        "init", state_mapping[dfa_a.initial_state.state_id]
    )

    if use_exact:
        builder = ExactSparseMatrixBuilder(0, 0, 0, False, True)
    else:
        builder = SparseMatrixBuilder(0, 0, 0, False, True)
    current_row = 0
    for s in sorted(dfa_a.states, key=lambda q: q.state_id):
        if s.is_accepting:
            state_labeling.add_label_to_state("accepting", state_mapping[s.state_id])

        builder.new_row_group(current_row)
        for act, dest_s in s.transitions.items():
            builder.add_next_value(
                current_row,
                state_mapping[dest_s.state_id],
                Rational(1.0) if use_exact else 1.0,
            )
            choice_labeling.add_label_to_choice(act, current_row)
            current_row += 1

    matrix = builder.build(overridden_column_count=len(dfa_a.states))

    if use_exact:
        components = SparseExactModelComponents(matrix, state_labeling)
        components.choice_labeling = choice_labeling
        return SparseExactMdp(components)
    else:
        components = SparseModelComponents(matrix, state_labeling)
        components.choice_labeling = choice_labeling
        return SparseMdp(components)
