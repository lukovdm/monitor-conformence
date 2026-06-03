"""Unit tests for tover.lsharp.apartness.

These are Storm-free: they build small observation-tree fragments out of
``MooreNode`` objects by hand and check the apartness primitives directly.

They act as a behaviour guard for the planned optimizations:
  * B2 - iterating unordered node pairs in the find_hypothesis apartness loop
    must yield the exact same set of apart pairs as the current ordered loop.
  * B1 - caching apartness must return identical booleans, and relies on
    apartness being monotonic (once apart, always apart).
"""

from itertools import combinations

from tover.lsharp.apartness import Apartness
from tover.lsharp.monitor_observation_tree import MonitorObservationTree
from tover.lsharp.moore_node import MooreNode


class FakeTree:
    """Minimal stand-in for MonitorObservationTree.

    Apartness only reads ``automaton_type`` and ``alphabet`` from the tree for
    the apart/witness primitives (and ``use_compatibility`` for the
    incompatibility delegation path), plus the static ``get_transfer_sequence``.
    """

    get_transfer_sequence = staticmethod(MonitorObservationTree.get_transfer_sequence)

    def __init__(self, alphabet, use_compatibility=False):
        self.automaton_type = "dfa"
        self.alphabet = alphabet
        self.use_compatibility = use_compatibility


def build_tree(seqs_outputs):
    """Build a tree from a {sequence_tuple: output} mapping.

    The mapping must contain every prefix (including the empty tuple).
    Returns (root, {sequence: node}).
    """
    MooreNode._id_counter = 0
    root = MooreNode()
    root.set_output(seqs_outputs.get((), None))
    nodes = {(): root}
    for seq in sorted(seqs_outputs, key=len):
        if seq == ():
            continue
        parent = nodes[seq[:-1]]
        node = parent.extend_and_get(seq[-1], seqs_outputs[seq])
        node.set_output(seqs_outputs[seq])
        nodes[seq] = node
    return root, nodes


def test_states_are_apart_conflicting_successor():
    tree = FakeTree(["x", "y"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,
            ("b", "x"): False,
        }
    )
    p, q = nodes[("a",)], nodes[("b",)]
    assert Apartness.states_are_apart(p, q, tree) is True
    witness = Apartness.compute_witness(p, q, tree)
    assert witness == ["x"]


def test_states_not_apart_when_outputs_match():
    tree = FakeTree(["x", "y"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,
            ("b", "x"): True,
        }
    )
    p, q = nodes[("a",)], nodes[("b",)]
    assert Apartness.states_are_apart(p, q, tree) is False
    assert Apartness.compute_witness(p, q, tree) is None


def test_unknown_output_does_not_cause_apartness():
    tree = FakeTree(["x"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): "unknown",
            ("b",): None,
            ("b", "x"): True,
        }
    )
    p, q = nodes[("a",)], nodes[("b",)]
    assert Apartness.states_are_apart(p, q, tree) is False


def test_apartness_requires_both_successors_defined():
    tree = FakeTree(["x"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,  # no ("b","x") successor
        }
    )
    p, q = nodes[("a",)], nodes[("b",)]
    assert Apartness.states_are_apart(p, q, tree) is False


def test_get_distinguishing_sequences():
    tree = FakeTree(["x", "y"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("a", "y"): True,
            ("b",): None,
            ("b", "x"): False,
            ("b", "y"): True,
        }
    )
    group = [nodes[("a",)], nodes[("b",)]]
    seqs = list(Apartness.get_distinguishing_sequences(group, tree))
    assert ["x"] in seqs
    assert ["y"] not in seqs


def test_states_are_incompatible_delegates_to_apart_without_compatibility():
    tree = FakeTree(["x"], use_compatibility=False)
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,
            ("b", "x"): False,
        }
    )
    p, q = nodes[("a",)], nodes[("b",)]
    assert Apartness.states_are_incompatible(p, q, tree) is True


def _apart_pairs_ordered(node_list, tree):
    """Mimic the current ordered double loop in find_hypothesis."""
    pairs = set()
    for n1 in node_list:
        for n2 in node_list:
            if n1 is n2:
                continue
            if Apartness.states_are_apart(n1, n2, tree):
                pairs.add(frozenset((n1.id, n2.id)))
    return pairs


def _apart_pairs_unordered(node_list, tree):
    """The planned B2 unordered loop."""
    pairs = set()
    for i, n1 in enumerate(node_list):
        for n2 in node_list[i + 1 :]:
            if Apartness.states_are_apart(n1, n2, tree):
                pairs.add(frozenset((n1.id, n2.id)))
    return pairs


def test_b2_unordered_loop_yields_same_apart_pairs():
    """B2 guard: ordered vs unordered pair iteration is equivalent."""
    tree = FakeTree(["x", "y"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,
            ("b", "x"): False,
            ("c",): None,
            ("c", "x"): True,
            ("d",): None,
            ("d", "x"): "unknown",
        }
    )
    node_list = [nodes[s] for s in (("a",), ("b",), ("c",), ("d",))]
    assert _apart_pairs_ordered(node_list, tree) == _apart_pairs_unordered(
        node_list, tree
    )


def test_b1_apartness_is_monotonic_under_growth():
    """B1 relies on: once two nodes are apart, extending the tree keeps them
    apart. Verify a previously-apart pair stays apart after adding successors."""
    tree = FakeTree(["x", "y"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,
            ("b", "x"): False,
        }
    )
    p, q = nodes[("a",)], nodes[("b",)]
    assert Apartness.states_are_apart(p, q, tree) is True

    # Grow the tree with extra (compatible) observations on a fresh input.
    p.extend_and_get("y", "unknown").set_output("unknown")
    q.extend_and_get("y", "unknown").set_output("unknown")
    assert Apartness.states_are_apart(p, q, tree) is True


def test_b1_cache_matches_uncached_and_only_caches_true():
    """B1 guard: a tree with an apart cache returns identical booleans to one
    without, and the cache only ever stores apart (True) pairs."""
    plain = FakeTree(["x"])  # no _apart_cache attribute -> caching disabled
    cached = FakeTree(["x"])
    cached._apart_cache = set()

    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,
            ("b", "x"): False,  # apart from a
            ("c",): None,
            ("c", "x"): True,  # compatible with a
        }
    )
    a, b, c = nodes[("a",)], nodes[("b",)], nodes[("c",)]

    pairs = [(a, b), (b, a), (a, c), (c, a), (a, b)]  # repeats + both orderings
    for n1, n2 in pairs:
        assert Apartness.states_are_apart(n1, n2, plain) == Apartness.states_are_apart(
            n1, n2, cached
        )

    # Only the apart pair {a, b} should be cached; the compatible {a, c} not.
    assert frozenset((a.id, b.id)) in cached._apart_cache
    assert frozenset((a.id, c.id)) not in cached._apart_cache


def test_b1_cached_result_is_used():
    """A cached pair returns True even if recomputation would not (proves the
    cache is consulted)."""
    cached = FakeTree(["x"])
    cached._apart_cache = set()
    _, nodes = build_tree(
        {(): None, ("a",): None, ("a", "x"): True, ("b",): None, ("b", "x"): True}
    )
    a, b = nodes[("a",)], nodes[("b",)]
    assert Apartness.states_are_apart(a, b, cached) is False  # not actually apart
    cached._apart_cache.add(frozenset((a.id, b.id)))  # pretend it was found apart
    assert Apartness.states_are_apart(a, b, cached) is True  # served from cache


def test_b2_pairwise_symmetry_of_apartness():
    """Apartness is symmetric, which is why the unordered loop is sound."""
    tree = FakeTree(["x"])
    _, nodes = build_tree(
        {
            (): None,
            ("a",): None,
            ("a", "x"): True,
            ("b",): None,
            ("b", "x"): False,
        }
    )
    for n1, n2 in combinations([nodes[("a",)], nodes[("b",)]], 2):
        assert Apartness.states_are_apart(n1, n2, tree) == Apartness.states_are_apart(
            n2, n1, tree
        )
