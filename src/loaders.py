import stormpy
from stormvogel.model import Model
from stormvogel.mapping import stormpy_to_stormvogel


def load_mc(path: str) -> Model:
    mc_prism = stormpy.parse_prism_program(path)  # type: ignore
    options = stormpy.BuilderOptions()  # type: ignore
    options.set_build_all_labels()
    options.set_build_state_valuations()
    mcs = stormpy.build_sparse_model_with_options(mc_prism, options)  # type: ignore
    mc = stormpy_to_stormvogel(mcs)
    if mc is None:
        raise Exception("Could not build model")

    for s in mcs.states:
        mc.states[s.id].add_label(s.valuations)

    return mc


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

    return dfa
