from typing import Any

from stormpy import (
    BuilderOptions,
    ExactSparseMatrixBuilder,
    ExpressionManager,
    SparseDtmc,
    SparseExactDtmc,
    SparseExactModelComponents,
    SparseMatrixBuilder,
    SparseModelComponents,
    StateLabeling,
    build_sparse_exact_model_with_options,
    build_sparse_model_with_options,
    parse_prism_program,
    preprocess_symbolic_input,
)

from tover.utils.helpers import compact_json_str


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
