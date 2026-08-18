"""Seed the L# observation tree from a most-probable-path belief search.

L# starts from an empty observation tree and pays a membership query for every symbol it
explores. The belief most-probable-path search in Storm already walks the belief space of the
same model looking for the most probable observation trace that drives the risk over the
threshold, and every belief it settles on the way corresponds to an observation trace whose
risk it has already computed. Those traces are exactly the shape the observation tree wants,
so they can be inserted up front for free rather than re-queried one symbol at a time.

The search stops at the first belief whose risk reaches the threshold, so the seed set is the
by-product of goal-seeking rather than an exhaustive enumeration: it is biased towards the
most probable traces, which is the part of the language a monitor cares about most.

Crucially the outputs are derived from :meth:`FilteringSUL.output_for_risk`, using the risk
vector the SUL itself handed to its tracker. Seeded outputs that disagreed with the SUL by even
one symbol would teach L# a wrong automaton.

That derivation is not quite enough on its own. The search and the tracker normalise a belief
at different points, so in floating point their risks can differ by an ulp or so; for a belief
sitting essentially on a decision boundary the two can land on opposite sides. Such sequences
are detected and relabelled by querying the SUL directly -- see :func:`_near_boundary`.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Any, Literal

from stormpy import Rational
from stormpy.pomdp import _pomdp

from tover.core.sul import storm_one
from tover.utils.logger import logger

if TYPE_CHECKING:
    from tover.core.sul import FilteringSUL

SeedSequence = tuple[list[str], list[bool | Literal["unknown"]]]


class SeedingStats:
    """What the seeding search cost and produced."""

    def __init__(self) -> None:
        self.enabled = False
        self.found_risky_belief = False
        self.truncated = False
        self.timed_out = False
        self.expansions = 0
        self.settled_beliefs = 0
        # Distinct seed words actually handed to the tree, after horizon truncation collapses
        # traces that share a prefix. Counting before the collapse overstated this badly:
        # airportA-3 at horizon 6 reports 329 settled beliefs but only 80 distinct words.
        self.seed_sequences = 0
        self.seed_symbols = 0
        # How many settled traces collapsed into an already-seen word.
        self.duplicate_sequences = 0
        # Sequences whose risk sat close enough to a decision boundary that the search's own
        # risk could disagree with the tracker's; these were relabelled by the SUL.
        self.boundary_sequences = 0
        self.boundary_sul_steps = 0
        self.goal_trace_length = 0
        self.goal_log_probability = float("-inf")
        self.search_seconds = 0.0
        self.total_seconds = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _boundary_margin(sul: "FilteringSUL") -> float | None:
    """How close to a decision boundary counts as "the search and the tracker might disagree".

    Only floating point needs this. In exact arithmetic both normalisation orders compute the
    same rational, so there is nothing to guard against and no seed needs re-querying.
    """
    return None if sul.mc.is_exact else 1e-12


def _near_boundary(sul: "FilteringSUL", risk, margin: float) -> bool:
    """Whether `risk` sits close enough to a threshold that its label is not trustworthy."""
    bounds = sul.threshold if isinstance(sul.threshold, tuple) else (sul.threshold,)
    return any(abs(float(risk) - float(bound)) <= margin for bound in bounds)


def compute_seed_sequences(
    sul: "FilteringSUL",
    horizon: int | None = None,
    max_expansions: int = 100000,
    timeout_ms: int = 0,
) -> tuple[list[SeedSequence], SeedingStats]:
    """Run the belief search and turn everything it explored into ``(word, outputs)`` pairs.

    :param sul: The SUL the monitor is learning against. Its POMDP, risk vector, observation
        classes and threshold are all reused, so the seeds describe the same system.
    :param horizon: Longest seed word to insert. Pass the monitor's horizon, not the SUL's --
        with ``use_horizon_in_filtering=False`` the SUL has none, and unbounded seeds would
        assert labels for words outside the reference language, which `execute_query` never
        creates. Defaults to the SUL's horizon.
    :param max_expansions: Safety budget; the reachable belief space can be infinite.
    :param timeout_ms: Wall-clock budget for the search, 0 for none.
    :returns: The seed sequences and a :class:`SeedingStats`.
    """
    stats = SeedingStats()
    stats.enabled = True
    started = time()

    # With don't cares the threshold is an interval; "risky" means what the SUL would answer
    # True to, i.e. the upper end.
    threshold = sul.threshold[1] if isinstance(sul.threshold, tuple) else sul.threshold
    if sul.mc.is_exact and not isinstance(threshold, Rational):
        # The SUL compares exact risks against this Python float, so convert it to the rational
        # the float actually denotes rather than to the decimal it prints as -- otherwise a
        # belief sitting exactly on the threshold could be classified differently here.
        threshold = Rational(float(threshold))

    explore = (
        _pomdp.explore_belief_tracesExact
        if sul.mc.is_exact
        else _pomdp.explore_belief_tracesDouble
    )
    risk_values = (
        sul.risk_values
        if sul.mc.is_exact
        else [float(value) for value in sul.risk_values]
    )

    result = explore(sul.pomdp, risk_values, threshold, max_expansions, timeout_ms)

    stats.found_risky_belief = result.found
    stats.truncated = result.truncated
    stats.timed_out = result.timed_out
    stats.expansions = result.expansions
    stats.settled_beliefs = len(result.traces)
    stats.goal_trace_length = len(result.goal_trace)
    stats.goal_log_probability = result.goal_log_probability
    stats.search_seconds = result.search_seconds

    # Past the horizon the SUL stops tracking and boxes every answer, so a seed that ran on
    # would assert outputs the SUL never produces. Truncate instead -- which makes traces that
    # share a prefix collapse onto the same word, so deduplicate before counting.
    horizon = sul.horizon if horizon is None else horizon
    by_word: dict[tuple[str, ...], list] = {}
    boundary_words: set[tuple[str, ...]] = set()
    margin = _boundary_margin(sul)
    for trace in result.traces:
        length = len(trace.observations)
        if horizon is not None:
            length = min(length, horizon)
        if length == 0:
            continue  # the initial belief; the tree root already carries its output
        word = tuple(sul.observation_classes[o] for o in trace.observations[:length])
        if word in by_word:
            stats.duplicate_sequences += 1
            continue
        risks = list(trace.risks[:length])
        by_word[word] = [sul.output_for_risk(risk) for risk in risks]
        if margin is not None and any(_near_boundary(sul, r, margin) for r in risks):
            boundary_words.add(word)

    # The search and the tracker compute a belief's risk with the normalisation in a different
    # order, so in floating point they can disagree by an ulp or so. That is harmless except
    # for a belief sitting essentially on a decision boundary, where the two can land on
    # different sides and the seed would teach L# a label the SUL contradicts. Rather than
    # trust the derived label there, ask the SUL. Borderline beliefs are rare, so this costs
    # a handful of queries.
    for word in boundary_words:
        sul.pre()
        by_word[word] = [sul.step(symbol) for symbol in word]
        sul.post()
        stats.boundary_sul_steps += len(word)
    stats.boundary_sequences = len(boundary_words)

    seeds: list[SeedSequence] = [(list(word), outputs) for word, outputs in by_word.items()]
    stats.seed_sequences = len(seeds)
    stats.seed_symbols = sum(len(word) for word, _ in seeds)
    stats.total_seconds = time() - started

    logger.info(
        f"Seeding: {stats.seed_sequences} distinct sequences / {stats.seed_symbols} symbols "
        f"from {stats.settled_beliefs} settled beliefs "
        f"({stats.duplicate_sequences} collapsed by horizon truncation, "
        f"{stats.boundary_sequences} relabelled by the SUL) in {stats.total_seconds:.3f}s "
        f"(risky belief found: {stats.found_risky_belief}, "
        f"goal trace length {stats.goal_trace_length}, "
        f"truncated: {stats.truncated}, timed out: {stats.timed_out})"
    )
    if not result.found:
        logger.warning(
            "The belief search did not reach the risk threshold, so the seeds cover only the "
            "part of the belief space it managed to explore."
        )
    return seeds, stats
