import random
from itertools import chain
from typing import final

from aalpy.automata import Dfa
from aalpy.base import SUL, Oracle
from aalpy.oracles.WpMethodEqOracle import (
    first_phase_it,
    second_phase_it,
    state_characterization_set,
)


def reference_filter(seq, reference):
    """
    Truncates a test sequence based on whether the next input is accepting in the reference model
    """
    seq_filtered = []
    reference.reset_to_initial()
    for letter in seq:
        out_exp = reference.step(letter)
        if not out_exp:
            return list(seq_filtered)
        seq_filtered.append(letter)
    return list(seq)


@final
class MonitorWpMethodEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle and takes a reference model into account.
    Whenever an input sequence is not enabled or exceeds the horizon as indicates by the reference model, we truncate it
    to it's defined prefix
    """

    def __init__(
        self, alphabet: list[str], sul: SUL, reference: Dfa[str], depth: int = 2
    ):
        super().__init__(alphabet, sul)
        self.depth = depth + 1
        self.reference = reference
        self.cache = set()

    def find_cex(self, hypothesis: Dfa[str]) -> list[str] | None:
        if len(hypothesis.states) == 1:
            hypothesis.characterization_set = [
                (a,) for a in hypothesis.get_input_alphabet()
            ]

        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        transition_cover = [
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        ]
        random.shuffle(transition_cover)

        state_cover = [state.prefix for state in hypothesis.states]
        random.shuffle(state_cover)

        difference = set(transition_cover).difference(set(state_cover))

        # first phase State Cover * Middle * Characterization Set
        first_phase = first_phase_it(
            self.alphabet, state_cover, self.depth, hypothesis.characterization_set
        )

        # second phase (Transition Cover - State Cover) * Middle * Characterization Set
        # of the state that the prefix leads to
        second_phase = second_phase_it(
            hypothesis, self.alphabet, difference, self.depth
        )
        test_suite = chain(first_phase, second_phase)

        for seq in test_suite:
            hypothesis.reset_to_initial()
            seq = reference_filter(seq, self.reference)

            if tuple(seq) not in self.cache:
                # self.reset_hyp_and_sul(hypothesis)
                #     for ind, letter in enumerate(seq):
                #         out_hyp = hypothesis.step(letter)
                #         out_sul = self.sul.step(letter)
                #         self.num_steps += 1

                #         if out_hyp != out_sul and sul_o != 'unknown':
                #             self.sul.post()
                #             return seq[: ind + 1]
                # self.sul.post()

                out_hyp = hypothesis.compute_output_seq(hypothesis.initial_state, seq)
                out_sul = self.sul.query(seq)
                for sul_o, hyp_o in zip(out_sul, out_hyp):
                    if sul_o != hyp_o and sul_o != "unknown":
                        return seq
                self.cache.add(tuple(seq))
        return None


@final
class MonitorRandomWpMethodEqOracle(Oracle):
    """
    Implements the Random Wp-Method as described in "Complementing Model
    Learning with Mutation-Based Fuzzing" by Rick Smetsers, Joshua Moerman,
    Mark Janssen, Sicco Verwer.
        1) sample uniformly from the states for a prefix
        2) sample geometrically a random word
        3) sample a word from the set of suffixes / state identifiers
    Additionally, it takes a reference model into account.
    Whenever an input sequence is not enabled or exceeds the horizon as indicates by the reference model, we truncate it
    to it's defined prefix
    """

    def __init__(
        self,
        alphabet: list[str],
        sul: SUL,
        reference: Dfa[str],
        min_length: int = 1,
        expected_length: int = 5,
        max_seqs: int = 5000,
    ):
        super().__init__(alphabet, sul)
        self.reference = reference
        self.min_length = min_length
        self.expected_length = expected_length
        self.max_seqs = max_seqs

    def find_cex(self, hypothesis: Dfa[str]):
        hypothesis.characterization_set = hypothesis.compute_characterization_set()
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = [
                (a,) for a in hypothesis.get_input_alphabet()
            ]

        state_mapping = {
            s: state_characterization_set(hypothesis, self.alphabet, s)
            for s in hypothesis.states
        }

        tries = self.max_seqs
        while tries > 0:
            tries -= 1
            state = random.choice(hypothesis.states)
            _ = self.reference.execute_sequence(
                self.reference.initial_state, state.prefix
            )
            reference_state = self.reference.current_state
            input = list(state.prefix) if state.prefix is not None else []
            limit = self.min_length
            while limit > 0 or random.random() > 1 / (self.expected_length + 1):
                alp = [
                    i
                    for i in self.alphabet
                    if i in reference_state.transitions and reference_state.is_accepting
                ]
                if len(alp) == 0:
                    break
                letter = random.choice(alp)
                reference_state = reference_state.transitions[letter]
                input.append(letter)
                limit -= 1
            if random.random() > 0.5:
                # global suffix with characterization_set
                input += random.choice(hypothesis.characterization_set)
            else:
                # local suffix
                _ = hypothesis.execute_sequence(hypothesis.initial_state, input)
                if state_mapping[hypothesis.current_state]:
                    input += random.choice(state_mapping[hypothesis.current_state])
                else:
                    continue

            seq = input

            out_ref = self.reference.compute_output_seq(
                self.reference.initial_state, seq
            )
            if False in out_ref:
                idx = out_ref.index(False)
                seq = seq[:idx]

            # self.reset_hyp_and_sul(hypothesis)
            #     for ind, letter in enumerate(seq):
            #         out_hyp = hypothesis.step(letter)
            #         out_sul = self.sul.step(letter)
            #         self.num_steps += 1

            #         if out_hyp != out_sul and sul_o != 'unknown':
            #             self.sul.post()
            #             return seq[: ind + 1]
            # self.sul.post()
            out_sul = self.sul.query(seq)
            out_hyp = hypothesis.compute_output_seq(hypothesis.initial_state, seq)
            for sul_o, hyp_o in zip(out_sul, out_hyp):
                if sul_o != hyp_o and sul_o != "unknown":
                    return seq
        return None
