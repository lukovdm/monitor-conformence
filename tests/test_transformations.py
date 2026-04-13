from math import sqrt
from weakref import ref

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tover.core.transformations import language_of_hmm
from tover.models.pomdp import pomdp_to_stormpy_mc
from tover.models.snakes import load_snl_stormpy, random_snl_board
from tover.utils.draw import COLORS, draw_board


def test_reference_language_pomdp():
    initial_observation, observations, mc, expr_manager = pomdp_to_stormpy_mc(
        "experiments/premise/airportA-3.nm", "DMAX=3,PMAX=3", False
    )
    alphabet = list(observations)

    refrence_lang = language_of_hmm(mc, alphabet)
    print(refrence_lang)
    refrence_lang.visualize()


def test_reference_language_snl():
    n, ladders, snakes = random_snl_board(2**2)
    mc, expr_manager = load_snl_stormpy(
        "experiments/snake_ladder/mc_u_nxn.pm", n, ladders, snakes
    )
    alphabet = ["ladder", "snake", "normal", "good"]

    fig, ax = plt.subplots(figsize=(6, 6))
    width = int(sqrt(n))
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)
    ax.set_facecolor(COLORS["background"])
    ax.set_xticks(range(width + 1), ["" for _ in range(width + 1)])
    ax.xaxis.set_ticks_position("none")
    ax.set_yticks(range(width + 1), ["" for _ in range(width + 1)])
    ax.yaxis.set_ticks_position("none")
    ax.grid(True, color=COLORS["grid"], linestyle="-", linewidth=1)
    ax.set_aspect("equal")
    draw_board(ax, width, snakes, ladders, [n])
    plt.savefig("snl_board.png")
    print("Saved SNL board to snl_board.png")

    refrence_lang = language_of_hmm(mc, alphabet)
    print(refrence_lang)
    refrence_lang.visualize(file_type="png")


if __name__ == "__main__":
    # test_reference_language_pomdp()
    test_reference_language_snl()
