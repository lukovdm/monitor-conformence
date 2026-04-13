from aalpy.utils import load_automaton_from_file

from tover.lsharp.IcyDrivingSUL import IcyDrivingSUL
from tover.lsharp.monitor_lsharp import run_monitor_lsharp
from tover.lsharp.monitor_wp_method import (
    MonitorRandomWpMethodEqOracle,
    MonitorWpMethodEqOracle,
)

alphabet = ["icy", "dry"]
sul = IcyDrivingSUL()

# reference should be input enabled, all inputs that are not enabled or exceed the horizon should go to the sink state
reference = load_automaton_from_file("reference.dot", automaton_type="dfa")
eq_oracle = MonitorWpMethodEqOracle(alphabet, sul, reference, depth=2)
# eq_oracle = MonitorRandomWpMethodEqOracle(alphabet, sul, reference, min_length=1, expected_length=5, max_seqs=100)

learned_dfa, info = run_monitor_lsharp(
    alphabet,
    reference,
    sul,
    eq_oracle,
    return_data=True,
    solver_timeout=3600,
    replace_basis=False,
    use_compatibility=False,
)

print(learned_dfa)
print(info)
