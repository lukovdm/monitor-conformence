import stormpy
import stormvogel.model

def gen_id(dfa_id, mc_id, mc_len):
    return mc_len * dfa_id + mc_id

mc_prism = stormpy.parse_prism_program("tests/mc1.pm")
mc = stormpy.build_model(mc_prism)
print(mc)

dfa_prism = stormpy.parse_prism_program("tests/dfa1.pm")
dfa = stormpy.build_model(dfa_prism)
print(dfa)

builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)

for dfa_state in dfa.states:
    for mc_state in mc.states:
        id = gen_id(dfa_state.id, mc_state.id, len(mc.states))
        builder.new_row_group(id)

        # dfa id, mc id, new id, mc labels, the transitions the mc takes, the transitions the dfa takes.
        print((dfa_state.id, mc_state.id), id, mc_state.labels, list(mc_state.actions)[0].transitions, [(a.id, a.origins, str(a.transitions)) for a in dfa_state.actions])
        for mc_trans in mc_state.actions[0].transitions:
            dfa_trans, action_id = [(list(a.transitions)[0], a.id) for a in dfa_state.actions if dfa_prism.get_action_name(a.id+1) in mc.labels_state(mc_trans.column)][0]
            
            # id of associated dfa transition location, id of mc transtition location, id of gen location, label of transition, label of location state in mc 
            print((dfa_trans.column, mc_trans.column), gen_id(dfa_trans.column, mc_trans.column, len(mc.states)), dfa.labels_state(dfa_trans.column), mc.labels_state(mc_trans.column))
            # builder.add_next_value(action_id, gen_id(dfa_trans.column, mc_trans.column, len(mc.states)), mc_trans.value())
