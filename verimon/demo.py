from stormvogel.mapping import stormvogel_to_stormpy

from verimon import loaders
from verimon.generator import Verifier
from verimon.logger import setup_logging
from verimon.transformations import simulator_unroll

setup_logging()

mon_sl_nxn = "../tests/snake_ladder/mon_nxn.pm"
mc_sl_u_nxn = "../tests/snake_ladder/mc_u_nxn.pm"

mon = loaders.load_dfa(mon_sl_nxn)
mon = simulator_unroll(mon, 12)
mon = stormvogel_to_stormpy(mon)


# n, ladders, snakes = random_snl_board(5**2)
#
# # Random snakes and ladders
# mc = loaders.load_snl_stormpy(mc_sl_u_nxn, n, ladders, snakes)

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
mc = loaders.load_snl_stormpy(
    mc_sl_u_nxn, n := 10**2, ladders := milton_ladders, snakes := milton_snakes
)

pomdp = Verifier(mc, mon, "good")
pomdp.create_product()
print("created product")

print(pomdp.monitor_verifier.get_product())
print(pomdp.monitor_verifier.observation_map)

assignment = pomdp.check_paynt_prop('Pmax=? [ (F "goal") ]')
print(pomdp.trace_of_assignment(assignment))
