import stormvogel.model


def product_mc(
    mc: stormvogel.model.Model, dfa: stormvogel.model.Model
) -> stormvogel.model.Model:
    if mc.type != stormvogel.model.ModelType.DTMC:
        raise Exception("Wrong model type for the MC, should be DTMC", mc.type)

    if dfa.type != stormvogel.model.ModelType.MDP:
        raise Exception("Wrong model type for the DFA, should be MDP", dfa.type)

    prod_mc = stormvogel.model.new_dtmc("Product")

    init_state = prod_mc.get_initial_state()
    states: dict[tuple[int, int], stormvogel.model.State] = {}

    for dfa_id, dfa_state in dfa.states.items():
        for mc_id, mc_state in mc.states.items():
            labels = [f"l{dfa_id}", f"s{mc_id}"] + list(
                filter(lambda l: l != "init", mc_state.labels + dfa_state.labels)
            )

            if "init" in dfa_state.labels and "init" in mc_state.labels:
                state = init_state
                for l in labels:
                    state.add_label(l)
            else:
                state = prod_mc.new_state(labels)

            states[(dfa_id, mc_id)] = state

    for dfa_id, dfa_trans in dfa.transitions.items():
        for mc_id, mc_trans in mc.transitions.items():
            state = states[(dfa_id, mc_id)]
            mc_branch = mc_trans.transition[stormvogel.model.EmptyAction]

            mc_transs = [
                (p, s)
                for (p, s) in mc_branch.branch
                if not set(
                    map(lambda a: list(a.labels)[0], dfa_trans.transition.keys())
                ).isdisjoint(s.labels)
            ]
            total_prob = sum([float(p) for (p, _) in mc_transs])

            new_trans = []
            for action, branch in dfa_trans.transition.items():
                if branch.branch[0][0] != 1:
                    raise Exception(
                        f"DFA state {dfa_id} has probabilistic transitions on {action}, {dfa_trans}"
                    )
                dfa_dest = branch.branch[0][1]
                action_label = list(action.labels)[0]

                new_trans += [
                    (float(p) / total_prob, states[(dfa_dest.id, s.id)])
                    for (p, s) in mc_transs
                    if action_label in s.labels
                ]

            state.set_transitions(new_trans)
    return prod_mc
