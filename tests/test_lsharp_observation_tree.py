"""Tests for tover.lsharp.monitor_observation_tree and the L# learning loop.

Storm-free: uses a hand-built aalpy Dfa reference plus a small custom SUL, in
the spirit of tover/lsharp/Example.py + IcyDrivingSUL.py.

The golden end-to-end test is the strongest behaviour guard for the planned
optimizations (A1-A3, B1, B2): it asserts the *language* learned by
``run_monitor_lsharp`` is unchanged, not the exact automaton structure.

Note: ``use_compatibility=True`` exercises a separate ``Apartness.merge`` code
path which currently crashes on simple synthetic references (a pre-existing bug
unrelated to these optimizations), so the end-to-end test pins the
``use_compatibility=False`` configuration that Example.py uses. The merge-free
apartness primitives that the optimizations touch are covered directly in
tests/test_lsharp_apartness.py.
"""

import random
from itertools import product

import pytest
from aalpy.automata import Dfa, DfaState
from aalpy.base.SUL import SUL

from tover.lsharp.monitor_lsharp import run_monitor_lsharp
from tover.lsharp.monitor_observation_tree import MonitorObservationTree
from tover.lsharp.monitor_wp_method import MonitorWpMethodEqOracle

ALPHABET = ["a", "b"]
HORIZON = 4


def build_reference(alphabet, horizon):
    """A reference DFA that marks every word of length <= horizon as defined
    (accepting) and routes anything longer into a rejecting sink.

    ``defined_in_reference`` reads ``compute_output_seq`` (one output per
    consumed input), so the state reached after k inputs must be accepting for
    1 <= k <= horizon and rejecting beyond.
    """
    states = [DfaState(f"q{i}") for i in range(horizon + 1)]
    sink = DfaState("sink")
    sink.is_accepting = False
    for letter in alphabet:
        sink.transitions[letter] = sink
    for i, st in enumerate(states):
        st.is_accepting = True
        for letter in alphabet:
            st.transitions[letter] = states[i + 1] if i < horizon else sink
    return Dfa(states[0], states + [sink])


def _ends_in_b(prefix):
    return len(prefix) > 0 and prefix[-1] == "b"


class EndsInBSUL(SUL):
    """Monitor that accepts (True) exactly the words ending in 'b'."""

    def query(self, word):
        self.pre()
        out = [_ends_in_b(word[:i]) for i in range(1, len(word) + 1)] or [
            _ends_in_b(())
        ]
        self.post()
        self.num_queries += 1
        self.num_steps += len(word)
        return out

    def pre(self):
        pass

    def post(self):
        pass

    def step(self, letter=None):
        raise RuntimeError("Only use query")


class UnknownZoneSUL(SUL):
    """Like EndsInBSUL but returns 'unknown' (don't-care) for words containing
    'aa', to exercise the unknown-output handling."""

    def query(self, word):
        self.pre()
        out = []
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            if any(prefix[j] == "a" and prefix[j + 1] == "a" for j in range(i - 1)):
                out.append("unknown")
            else:
                out.append(_ends_in_b(prefix))
        if not out:
            out = [_ends_in_b(())]
        self.post()
        self.num_queries += 1
        self.num_steps += len(word)
        return out

    def pre(self):
        pass

    def post(self):
        pass

    def step(self, letter=None):
        raise RuntimeError("Only use query")


def _assert_language_matches(learned, sul, alphabet, horizon):
    """Every word up to horizon: learned output matches SUL on non-unknown
    positions."""
    for length in range(horizon + 1):
        for word in product(alphabet, repeat=length):
            word = list(word)
            sul_out = sul.query(word)
            hyp_out = learned.compute_output_seq(learned.initial_state, word)
            for s, h in zip(sul_out, hyp_out):
                if s != "unknown":
                    assert s == h, f"mismatch on {word}: sul={s} hyp={h}"


@pytest.mark.parametrize("replace_basis", [False, True])
def test_golden_ends_in_b(replace_basis):
    random.seed(0)
    reference = build_reference(ALPHABET, HORIZON)
    sul = EndsInBSUL()
    oracle = MonitorWpMethodEqOracle(ALPHABET, sul, reference, depth=2)

    learned, info = run_monitor_lsharp(
        ALPHABET,
        reference,
        sul,
        oracle,
        solver_timeout=60,
        learning_timeout=120,
        replace_basis=replace_basis,
        use_compatibility=False,
    )

    _assert_language_matches(learned, sul, ALPHABET, HORIZON)
    assert info["learning_rounds"] >= 1


@pytest.mark.parametrize("replace_basis", [False, True])
def test_golden_with_unknown_zone(replace_basis):
    random.seed(1)
    reference = build_reference(ALPHABET, HORIZON)
    sul = UnknownZoneSUL()
    oracle = MonitorWpMethodEqOracle(ALPHABET, sul, reference, depth=2)

    learned, _ = run_monitor_lsharp(
        ALPHABET,
        reference,
        sul,
        oracle,
        solver_timeout=60,
        learning_timeout=120,
        replace_basis=replace_basis,
        use_compatibility=False,
    )

    _assert_language_matches(learned, sul, ALPHABET, HORIZON)


@pytest.mark.parametrize("replace_basis", [False, True])
def test_classic_path_without_dont_cares(replace_basis):
    """With use_dont_care=False the learner must use the classic L# construction
    (no SMT solver) and still learn the correct language."""
    random.seed(2)
    reference = build_reference(ALPHABET, HORIZON)
    sul = EndsInBSUL()
    oracle = MonitorWpMethodEqOracle(ALPHABET, sul, reference, depth=2)

    learned, info = run_monitor_lsharp(
        ALPHABET,
        reference,
        sul,
        oracle,
        solver_timeout=60,
        learning_timeout=120,
        replace_basis=replace_basis,
        use_compatibility=False,
        use_dont_care=False,
    )

    _assert_language_matches(learned, sul, ALPHABET, HORIZON)
    # The classic path never invokes the SMT solver.
    assert info["smt_time"] == 0


@pytest.mark.parametrize("use_dont_care", [True, False])
def test_reference_language_optional(use_dont_care):
    """run_monitor_lsharp must work without a reference language (reference=None):
    every queried sequence is treated as defined."""
    random.seed(3)
    reference = build_reference(ALPHABET, HORIZON)
    sul = EndsInBSUL()
    # The tree gets no reference; the eq oracle keeps a reference only to bound
    # counterexample search to the horizon for this assertion.
    oracle = MonitorWpMethodEqOracle(ALPHABET, sul, reference, depth=2)

    learned, _ = run_monitor_lsharp(
        ALPHABET,
        None,
        sul,
        oracle,
        solver_timeout=60,
        learning_timeout=120,
        replace_basis=False,
        use_compatibility=False,
        use_dont_care=use_dont_care,
    )

    _assert_language_matches(learned, sul, ALPHABET, HORIZON)


def _make_tree():
    sul = EndsInBSUL()
    reference = build_reference(ALPHABET, HORIZON)
    return MonitorObservationTree(
        ALPHABET,
        reference,
        sul,
        solver_timeout=60,
        replace_basis=False,
        use_compatibility=False,
    )


def test_insert_and_get_successor():
    tree = _make_tree()
    tree.insert_observation_sequence(["a", "b"], [False, True])
    node = tree.get_successor(["a", "b"])
    assert node is not None
    assert node.output is True
    assert tree.get_successor(["a"]).output is False
    assert tree.get_successor(["a", "a"]) is None  # never inserted


def test_get_access_and_transfer_sequence():
    tree = _make_tree()
    tree.insert_observation_sequence(["a", "b", "a"], [False, True, False])
    node = tree.get_successor(["a", "b", "a"])
    assert tree.get_access_sequence(node) == ["a", "b", "a"]

    mid = tree.get_successor(["a"])
    assert tree.get_transfer_sequence(mid, node) == ["b", "a"]
    assert tree.get_transfer_sequence(tree.root, node) == ["a", "b", "a"]


def test_count_informative_nodes_excludes_unknown():
    tree = _make_tree()
    # root (False) + a (False) + ab (True) = 3 informative nodes
    tree.insert_observation_sequence(["a", "b"], [False, True])
    assert tree.count_informative_nodes() == 3
    # adding an "unknown" leaf must not increase the informative count
    tree.insert_observation_sequence(["a", "b", "a"], [False, True, "unknown"])
    assert tree.count_informative_nodes() == 3
