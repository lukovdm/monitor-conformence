import stormpy
from stormvogel.mapping import stormpy_to_stormvogel
from stormvogel.model import Model


def load_mc(path: str) -> tuple[Model, int, dict[int, int], dict[int, int]]:
    mc_prism = stormpy.parse_prism_program(path)  # type: ignore
    n, ladders, snakes = get_sl_prism_consts(mc_prism)
    options = stormpy.BuilderOptions()  # type: ignore
    options.set_build_all_labels()
    options.set_build_state_valuations()
    mcs = stormpy.build_sparse_model_with_options(mc_prism, options)  # type: ignore
    mc = stormpy_to_stormvogel(mcs)
    if mc is None:
        raise Exception("Could not build model")

    for s in mcs.states:
        mc.states[s.id].add_label(s.valuations)

    return mc, n, ladders, snakes


def load_dfa(path: str) -> Model:
    dfa_prism = stormpy.parse_prism_program(path)  # type: ignore
    options = stormpy.BuilderOptions()  # type: ignore
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_state_valuations()
    dfas = stormpy.build_sparse_model_with_options(dfa_prism, options)  # type: ignore
    dfa = stormpy_to_stormvogel(dfas)
    if dfa is None:
        raise Exception("Could not build model")

    for s in dfas.states:
        dfa.states[s.id].add_label(s.valuations)

    return dfa


def get_sl_prism_consts(model) -> tuple[int, dict[int, int], dict[int, int]]:
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
