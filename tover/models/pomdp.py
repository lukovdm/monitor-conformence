from typing import Any

from stormpy import (
    preprocess_symbolic_input,
    parse_prism_program,
    BuilderOptions,
    build_sparse_model_with_options,
    build_sparse_exact_model_with_options,
    SparseDtmc,
    SparseExactDtmc,
    ExpressionManager,
    ExactSparseMatrixBuilder,
    SparseMatrixBuilder,
    StateLabeling,
    Rational,
    SparseExactModelComponents,
    SparseModelComponents,
)
from stormvogel.mapping import stormpy_to_stormvogel, stormvogel_to_stormpy
from stormvogel.model import Model, EmptyAction, Branch, ModelType, new_mdp

from tover.utils.helpers import compact_json_str
from tover.utils.logger import logger


def pomdp_to_stormpy_mc(
    path: str, constants: str, use_exact: bool
) -> tuple[str, set[str], SparseDtmc | SparseExactDtmc, ExpressionManager]:
    """Load a POMDP from a PRISM file and convert it to a labelled Stormpy DTMC.

    Observation valuations are encoded as state labels so the belief tracker
    and the learning alphabet can work with plain strings.
    """
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

    if use_exact:
        builder = ExactSparseMatrixBuilder(0, 0, 0, False, False)
    else:
        builder = SparseMatrixBuilder(0, 0, 0, False, False)

    for s in pomdp_stormpy.states:
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
        return initial_observation, observation_classes, SparseExactDtmc(components), prism.expression_manager
    else:
        components = SparseModelComponents(matrix, state_labeling)
        return initial_observation, observation_classes, SparseDtmc(components), prism.expression_manager


def pomdp_to_mc(
    path: str, constants: str
) -> tuple[str, set[str], SparseDtmc, ExpressionManager]:
    """Load a POMDP and convert it to a Stormvogel-then-Stormpy DTMC (averaging over actions)."""
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
