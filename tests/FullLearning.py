# %%
from stormpy import export_to_drn
import sys

sys.path.append("..")

from verimon.logger import setup_logging

setup_logging()

# %%
from verimon import loaders


mc_sl_u_nxn = "../tests/snake_ladder/mc_u_nxn.pm"

# n, ladders, snakes = loaders.random_snl_board(5**2)
n, ladders, snakes = (
    25,
    {17: 19, 9: 15, 8: 15, 6: 10, 14: 21},
    {23: 8, 22: 20, 8: 1, 18: 2, 12: 2},
)
print(n, ladders, snakes)

# Random snakes and ladders
mc = loaders.load_snl_stormpy(mc_sl_u_nxn, n, ladders, snakes)

milton_snakes = {
    98: 76,
    95: 75,
    93: 73,
    87: 24,
    64: 60,
    62: 19,
    55: 53,
    49: 11,
    47: 26,
    16: 6,
}
milton_ladders = {1: 38, 4: 14, 9: 31, 28: 64, 40: 42, 36: 44, 51: 67, 71: 91, 80: 100}
# mc = loaders.load_snl_stormpy(mc_sl_u_nxn, n := 10**2, ladders:=milton_ladders, snakes:=milton_snakes)

# %%
from stormvogel.mapping import stormpy_to_stormvogel, stormvogel_to_stormpy
from stormvogel.show import show
import stormvogel

stormvogel.communication_server.enable_server = False

mc_sv = stormpy_to_stormvogel(mc)
loaders._add_valuation_to_sv_labels(mc, mc_sv)

# %%
from verimon.draw import animate_player_movement
import math
from IPython.display import HTML

player_path = [(0, [])]

goal_squares = [
    next(int(l[5:-1]) for l in state.labels if l.startswith("[pos"))
    for state in mc_sv.states.values()
    if "good" in state.labels
]

animation = animate_player_movement(
    int(math.sqrt(n)), snakes, ladders, goal_squares, player_path
)
animation.save("pre.gif")

# %%
from aalpy import run_Lstar
from verimon.MonitorLearning import FilteringSUL, VerimonEqOracle

setup_logging()


threshold = 0.4
slack = 0.2
horizon = 12
relative_error = 0.1
spec = 'P>0.5 [F<3 "good" ]'

alphabet = ["init", "normal", "snake", "ladder"]

filtering_sul = FilteringSUL(mc, "init", alphabet, spec, threshold, horizon)
eq_oracle = VerimonEqOracle(
    alphabet, filtering_sul, mc, threshold, slack, horizon, spec, "good", relative_error
)
learned_monitor = run_Lstar(
    alphabet,
    filtering_sul,
    eq_oracle,
    automaton_type="dfa",
    print_level=2,
)

# %%
from verimon.MonitorLearning import aalpy_dfa_to_stormvogel
from verimon.transformations import simulator_unroll, prune_monitor
from verimon.algs import complement_model

mon_cycl = aalpy_dfa_to_stormvogel(learned_monitor)
# show(mon_cycl)
# complement_model(mon_cycl, "accepting")
mon = simulator_unroll(mon_cycl, horizon)
prune_monitor(mon)
print(len(mon.states))
# %%
from verimon.MonitorLearning import aalpy_dfa_to_stormvogel
from verimon.verify import *

mon_cycl = aalpy_dfa_to_stormvogel(learned_monitor)
export_to_drn(stormvogel_to_stormpy(mon_cycl), "monitor.drn")
result_goal, trace, assignment, product = false_positive(
    mc, mon_cycl, horizon, options={"good_spec": spec}
)

# %%
from verimon.draw import animate_player_movement
import math
from IPython.display import HTML

# player_path = [(0, [])]
poss = product.simulate_paynt_assignment(assignment, 100000)
player_path = poss

goal_squares = [
    int(str(state.valuations)[5:-1])
    for state in product.mc.states
    if "good" in state.labels
]

animation = animate_player_movement(
    int(math.sqrt(n)), snakes, ladders, goal_squares, player_path
)
animation.save("steps.gif")
# %%
