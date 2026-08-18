from enum import Enum
import itertools
from math import floor
import time
import random
from collections import deque

from aalpy import SUL
from aalpy.automata import Dfa, DfaState
from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import GE, LT, Bool, Function, Int, Or, Solver, Symbol
from pysmt.typing import BOOL, INT, FunctionType

from tover.lsharp.monitor.apartness import Apartness
from tover.lsharp.monitor.moore_node import MooreNode
from tover.utils.logger import logger


class SMTBehaviour(Enum):
    EXPO_BACKOFF = "expo_backoff"
    SEQUENTIAL = "sequential"


class MonitorObservationTree:
    def __init__(
        self,
        alphabet,
        reference,
        sul,
        solver_timeout,
        replace_basis,
        use_compatibility,
        integrate_testing,  # NEW
        depth,  # NEW
        full_testing,  # NEW
        test_per_frontier=5,  # NEW
        use_dont_care=True,
        smt_behaviour=SMTBehaviour.EXPO_BACKOFF,
    ):
        # ``reference`` may be None, in which case no reference language is used
        # and every queried sequence is treated as defined.
        """
        Initializes the observation tree with a root node.
        """
        self.automaton_type = "dfa"
        self.solver_timeout = solver_timeout * 1000
        self.replace_basis = replace_basis
        self.use_compatibility = use_compatibility
        # When don't cares are disabled the observation tree is fully defined
        # (the SUL never returns "unknown"), so the hypothesis can be built with
        # the classic L# construction instead of the SMT solver.
        self.use_dont_care = use_dont_care
        self.smt_behaviour = smt_behaviour

        self.integrate_testing = integrate_testing
        self.depth = depth
        self.full_testing = full_testing
        self.test_per_frontier = test_per_frontier

        # Logger information
        self.smt_time = 0
        # Number of SMT solver invocations (one per find_hypothesis call).
        self.smt_queries = 0
        MooreNode._id_counter = 0

        # Initialize tree
        self.alphabet = alphabet
        self.reference: Dfa = reference
        self.sul: SUL = sul
        self.outputAlphabet = [True, False, "unknown"]
        self.states_list = []

        self.root = MooreNode()
        # assuming querying an empty list returns a singleton list
        self.root.set_output(self.sul.query([])[0])

        self.size = 1
        self.basis = [self.root]
        self.frontier_to_basis_dict = dict()

        # Persistent positive cache of apart node pairs (frozenset of node ids).
        # Apartness is monotonic here: outputs only go None -> known (never flip,
        # since set_output is always fed the deterministic SUL result), so once
        # two nodes are apart they stay apart for the rest of the run. Only
        # positive results are cached; negatives can change as the tree grows.
        self.apart_cache = set()

    def insert_observation_sequence(self, inputs, outputs):
        """
        Insert an observation into the tree using a sequence of inputs and their corresponding outputs.
        """
        node = self.root
        for inp, output in zip(inputs, outputs):
            node = node.extend_and_get(inp, output)
            node.set_output(output)
            if node not in self.frontier_to_basis_dict and node not in self.basis:
                self.frontier_to_basis_dict[node] = set(self.basis)

    def get_successor(self, inputs, start_node=None):
        """
        Retrieve the node corresponding to the given input sequence
        """
        if start_node is None:
            node = self.root
        else:
            node = start_node
        for input_val in inputs:
            successor_node = node.get_successor(input_val)
            if successor_node is None:
                return None
            node = successor_node
        return node

    @staticmethod
    def get_transfer_sequence(start_node, end_node):
        """
        Get the sequence of inputs that moves from the start node to the end node.
        """
        transfer_sequence = []
        node = end_node

        while node != start_node:
            if node.parent is None:
                raise ValueError("End node is not reachable from the start node.")
            transfer_sequence.append(node.input_to_parent)
            node = node.parent

        transfer_sequence.reverse()
        return transfer_sequence

    def get_access_sequence(self, target_node):
        """
        Get the sequence of inputs that moves from the root node to the target node.
        """
        transfer_sequence = []
        node = target_node

        while node != self.root:
            if node.parent is None:
                raise ValueError("Target node is not reachable from the root.")
            transfer_sequence.append(node.input_to_parent)
            node = node.parent

        transfer_sequence.reverse()
        return transfer_sequence

    def get_size(self):
        """
        Get the number of nodes in the observation tree.
        """
        return self.root.id_counter

    @staticmethod
    def is_known(node):
        """
        Check if the output of a node is known.
        """
        return node.output is not None and node.output != "unknown"

    def count_informative_nodes(self):
        """
        counts how many nodes have informative information
        """
        queue = deque()
        queue.append(self.root)
        count = 0
        while queue:
            node = queue.popleft()
            if node.output != "unknown":
                count += 1
            for successor in node.successors.values():
                queue.append(successor)
        return count

    def reference_explored_depth(self):
        """Largest depth ``d`` such that every reference-defined word of length
        ``<= d`` is present as a node in the observation tree (i.e. has been
        queried). Returns ``None`` when no reference language is used.

        Walks the reference DFA and the observation tree in lockstep: a word is
        reference-defined as long as its transitions stay in accepting (non-dead)
        reference states, matching ``defined_in_reference``. Naturally caps at the
        horizon, since the reference has no defined continuations past it.
        """
        if self.reference is None:
            return None
        # Pairs (reference_state, tree_node) for words of length `depth` that are
        # both reference-defined and already present in the tree.
        frontier = [(self.reference.initial_state, self.root)]
        depth = 0
        while frontier:
            next_frontier = []
            for ref_state, node in frontier:
                for letter in self.alphabet:
                    ref_next = ref_state.transitions.get(letter)
                    if ref_next is None or not ref_next.is_accepting:
                        continue  # not defined in the reference -> don't require it
                    child = node.get_successor(letter)
                    if child is None:
                        return depth  # a defined word of length depth+1 is missing
                    next_frontier.append((ref_next, child))
            depth += 1
            frontier = next_frontier
        # Frontier ran dry without a missing node: the whole reference language is
        # explored. The last level carrying defined words was depth-1 (the loop
        # incremented once more past it), so that is the deepest covered depth
        # (= the reference's max length / horizon).
        return depth - 1

    def update_basis_candidates(self, frontier_node):
        """
        Update the basis candidates for a specific frontier node.
        """
        candidates = self.frontier_to_basis_dict[frontier_node]
        new_candidates = {
            node
            for node in candidates
            if not Apartness.states_are_incompatible(frontier_node, node, self)
        }
        self.frontier_to_basis_dict[frontier_node] = new_candidates

    def update_frontier_to_basis_dict(self):
        """
        Update the basis candidates for all frontier nodes.
        """
        self.update_frontier_to_basis_dict_dfs(self.root)

    def update_frontier_to_basis_dict_dfs(self, node):
        if node not in self.basis:
            self.update_basis_candidates(node)
        for successor in node.successors.values():
            if (
                successor not in self.basis
                and len(self.frontier_to_basis_dict[successor]) == 0
            ):
                continue
            self.update_frontier_to_basis_dict_dfs(successor)

    def promote_node_to_basis(self):
        """
        If an isolated frontier node is found, reset the queue and restart from the basis plus the isolated node.
        """
        queue = deque([self.root])
        while queue:
            iso_frontier_node = queue.popleft()
            for successor in iso_frontier_node.successors.values():
                queue.append(successor)
            if iso_frontier_node in self.basis:
                continue
            basis_list = self.frontier_to_basis_dict[iso_frontier_node]
            if not basis_list:
                self.basis.append(iso_frontier_node)
                logger.debug(
                    f"Added {self.get_access_sequence(iso_frontier_node)} to basis"
                )
                # Update the candidates
                del self.frontier_to_basis_dict[iso_frontier_node]
                for candidates in self.frontier_to_basis_dict.values():
                    candidates.add(iso_frontier_node)
                logger.debug(f"Increasing basis size to {len(self.basis)}")
                self.size = max(self.size, len(self.basis))
                return True

        if not self.replace_basis:
            return False

        queue = deque([self.root])
        while queue:
            iso_frontier_node = queue.popleft()
            for successor in iso_frontier_node.successors.values():
                queue.append(successor)
            if iso_frontier_node in self.basis:
                continue
            basis_list = self.frontier_to_basis_dict[iso_frontier_node]
            if len(basis_list) == 1:
                candidate = list(basis_list)[0]
                if len(self.get_access_sequence(candidate)) <= len(
                    self.get_access_sequence(iso_frontier_node)
                ):
                    continue
                self.basis.remove(candidate)
                self.basis.append(iso_frontier_node)
                # Update the candidates
                del self.frontier_to_basis_dict[iso_frontier_node]
                for candidates in self.frontier_to_basis_dict.values():
                    if candidate in candidates:
                        candidates.remove(candidate)
                    candidates.add(iso_frontier_node)
                self.frontier_to_basis_dict[candidate] = {iso_frontier_node}
                return True
        return False

    def make_frontiers_identified(self):
        """
        Loop over all frontier nodes to identify them
        """
        extended = False
        witness_cache = dict()
        for basis_node in self.basis:
            for letter in self.alphabet:
                # not defined if rejecting in reference
                if frontier_node := basis_node.get_successor(letter):
                    if frontier_node in self.basis:
                        continue
                    while self.identify_frontier(
                        frontier_node, witness_cache=witness_cache
                    ):
                        extended = True
                        self.update_basis_candidates(frontier_node)
        return extended

    def identify_frontier(self, frontier_node, witness_cache=None):
        """
        Identify a specific frontier node
        """
        if len(self.frontier_to_basis_dict[frontier_node]) == 0:
            return False

        inputs_to_frontier = self.get_transfer_sequence(self.root, frontier_node)

        witnesses = self._get_witnesses_bfs(frontier_node, witness_cache=witness_cache)
        for witness_seq in witnesses:
            inputs = inputs_to_frontier + witness_seq
            extended = self.execute_query(inputs)
            if extended:
                witness_cache = dict()  # Clear the cache since the tree has changed and there might be more witnesses
                return True
        return False

    def _get_witnesses_bfs(self, frontier_node, witness_cache=None):
        """
        Specifically identify frontier nodes using separating sequences
        """
        basis_candidates = frozenset(self.frontier_to_basis_dict.get(frontier_node, []))
        if witness_cache is not None and basis_candidates in witness_cache:
            witnesses = witness_cache[basis_candidates]
        else:
            witnesses = list(
                Apartness.get_distinguishing_sequences(basis_candidates, self)
            )
            if witness_cache is not None:
                witness_cache[basis_candidates] = witnesses

        for witness_seq in witnesses:
            leads_to_node = self.get_successor(witness_seq, start_node=frontier_node)
            if leads_to_node is None or leads_to_node.output is None:
                yield witness_seq

    def construct_hypothesis_states(self, output_mapping=None):
        """
        Construct the hypothesis states from the basis
        """
        self.states_list = [DfaState(f"s{i}") for i in range(self.size)]
        for i, dfa_state in enumerate(self.states_list):
            dfa_state.is_accepting = output_mapping[i]

    def construct_hypothesis_transitions(self, transition_mapping=None):
        """
        Construct the hypothesis transitions using the transition_mapping and output_mapping.
        """
        for i, dfa_state in enumerate(self.states_list):
            for j, letter in enumerate(self.alphabet):
                dfa_state.transitions[letter] = self.states_list[
                    transition_mapping[i][j]
                ]

    def construct_hypothesis(self, transition_mapping=None, output_mapping=None):
        """
        Constructs the hypothesis DFA from the transition and output mappings.
        """
        logger.debug(f"Constructing hypothesis")
        self.construct_hypothesis_states(output_mapping=output_mapping)
        self.construct_hypothesis_transitions(transition_mapping=transition_mapping)

        hypothesis = Dfa(self.states_list[0], self.states_list)
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = hypothesis.compute_characterization_set(raise_warning=False)

        return hypothesis

    def find_hypothesis(self):
        """
        Find a hypothesis consistent with the observation tree, using the pySMT solver.
        There are 2 free functions: "out" and "m" and 1 bound function "delta".
        """
        logger.debug(
            f"Trying to build hypothesis of size {self.size} "
            f"Basis size: {len(self.basis)}, Frontier size: {len(self.frontier_to_basis_dict)}"
        )
        start_smt_time = time.time()
        self.smt_queries += 1

        # or another backend supported by pySMT
        s = Solver(name="z3", solver_options={"timeout": self.solver_timeout})

        # Function declarations
        delta = Symbol("delta", FunctionType(INT, [INT, INT]))  # δ: int × int → int
        dfa_output = Symbol(
            "dfa_output", FunctionType(BOOL, [INT])
        )  # dfa_output: int → bool
        states_mapping = Symbol(
            "states_mapping", FunctionType(INT, [INT])
        )  # states_mapping: int → int

        # Precompute the position of each alphabet symbol once (avoids repeated
        # O(|alphabet|) scans inside the BFS below).
        alphabet_index = {letter: i for i, letter in enumerate(self.alphabet)}

        # Flatten the tree to a list of nodes, recording each node's index so we
        # never need an O(n) nodes.index() scan later.
        queue = deque([self.root])
        nodes = [self.root]
        node_to_index = {self.root: 0}

        while queue:
            node = queue.popleft()
            idx = node_to_index[node]
            for letter, successor in node.successors.items():
                # Check if successor can reach a known node
                queue.append(successor)
                node_to_index[successor] = len(nodes)
                s.add_assertion(
                    Function(states_mapping, [Int(len(nodes))]).Equals(
                        Function(
                            delta,
                            [
                                Function(states_mapping, [Int(idx)]),
                                Int(alphabet_index[letter]),
                            ],
                        )
                    )
                )
                nodes.append(successor)

        basis_index = {node: i for i, node in enumerate(self.basis)}

        # Basis nodes map to different states
        for node, i in basis_index.items():
            s.add_assertion(
                Function(states_mapping, [Int(node_to_index[node])]).Equals(Int(i))
            )

        # Force known outputs
        for i, node in enumerate(nodes):
            if self.is_known(node):
                val = Bool(node.output)
                s.add_assertion(
                    Function(dfa_output, [Function(states_mapping, [Int(i)])]).Iff(val)
                )

        # Frontier nodes map to the same state as one of their candidates or to a new state
        for node, candidates in self.frontier_to_basis_dict.items():
            if node not in node_to_index:
                continue
            node_idx = node_to_index[node]
            s.add_assertion(
                Or(
                    [
                        Function(states_mapping, [Int(node_idx)]).Equals(
                            Int(basis_index[c])
                        )
                        for c in candidates
                    ]
                    + [
                        Function(states_mapping, [Int(node_idx)]).Equals(Int(i))
                        for i in range(len(self.basis), self.size)
                    ]
                )
            )

        # Correct delta
        for i in range(self.size):
            for j in range(len(self.alphabet)):
                d_ij = Function(delta, [Int(i), Int(j)])
                s.add_assertion(GE(d_ij, Int(0)))
                s.add_assertion(LT(d_ij, Int(self.size)))

        # Appart nodes cannot be merged. Apartness is symmetric, so iterate over
        # unordered pairs (i < j): a single NotEquals per apart pair is enough.
        if self.use_compatibility:
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1 :]:
                    if Apartness.states_are_apart(node1, node2, self):
                        s.add_assertion(
                            Function(
                                states_mapping, [Int(node_to_index[node1])]
                            ).NotEquals(
                                Function(states_mapping, [Int(node_to_index[node2])])
                            )
                        )

        try:
            logger.debug("Solving...")
            if not s.solve():
                logger.debug("UNSAT")
                logger.debug(f"No hypothesis of size {self.size} exists")
                self.smt_time += time.time() - start_smt_time
                return None, None
            else:
                logger.debug("SAT")
                self.smt_time += time.time() - start_smt_time
                model = s.get_model()

                transition_mapping = [
                    [0 for _ in range(len(self.alphabet))] for _ in range(self.size)
                ]
                output_mapping = [False for _ in range(self.size)]

                for i in range(self.size):
                    val = model.get_value(Function(dfa_output, [Int(i)]))
                    output_mapping[i] = str(val) == "True"
                    for j in range(len(self.alphabet)):
                        val = model.get_value(Function(delta, [Int(i), Int(j)]))
                        transition_mapping[i][j] = int(str(val))

                return transition_mapping, output_mapping
        except SolverReturnedUnknownResultError:
            self.smt_time += time.time() - start_smt_time
            logger.debug("TIMEOUT")
            logger.debug(f"Could not find hypothesis of size {self.size}")
            return None, None

    def build_hypothesis(self):
        """
        Builds the hypothesis which will be sent to the SUL and checks consistency
        """
        # if not self.use_dont_care:
        #     # The observation tree is fully defined, so we can use the classic
        #     # L# construction (no SMT solver needed).
        #     return self.build_hypothesis_classic()

        if self.smt_behaviour == SMTBehaviour.EXPO_BACKOFF:
            return self.build_hypothesis_expo_backoff()
        elif self.smt_behaviour == SMTBehaviour.SEQUENTIAL:
            return self.build_hypothesis_sequential()
        else:
            raise ValueError(f"Unknown SMT behaviour: {self.smt_behaviour}")

    def build_hypothesis_sequential(self):
        """
        Builds the hypothesis which will be sent to the SUL and checks consistency
        """
        while True:
            logger.debug(f"Trying to build hypothesis of size {self.size}")
            self.find_adequate_observation_tree()
            transition_mapping, output_mapping = self.find_hypothesis()
            if transition_mapping is not None:
                hypothesis = self.construct_hypothesis(
                    transition_mapping=transition_mapping, output_mapping=output_mapping
                )
                return hypothesis
            else:
                self.size += 1

    def build_hypothesis_expo_backoff(self):
        """
        Builds the hypothesis which will be sent to the SUL and checks consistency
        """

        lower_bound = self.size
        upper_bound = None
        hypothesis = None

        first_iteration = True

        while upper_bound is None or lower_bound < upper_bound:
            self.find_adequate_observation_tree()

            # The size of the basis can have been updated while searching for an adequate observation tree.
            lower_bound = max(lower_bound, len(self.basis))

            if first_iteration:
                self.size = lower_bound
                first_iteration = False
            else:
                self.size = (
                    floor((lower_bound + upper_bound) / 2)
                    if upper_bound is not None
                    else lower_bound * 2
                )
            logger.info(
                f"Trying to find hypothesis of size {self.size} with bounds [{lower_bound}, {upper_bound}]"
            )

            transition_mapping, output_mapping = self.find_hypothesis()
            if transition_mapping is not None:
                hypothesis = self.construct_hypothesis(
                    transition_mapping=transition_mapping, output_mapping=output_mapping
                )
                upper_bound = self.size
            else:
                lower_bound = self.size + 1
                if lower_bound == upper_bound:
                    # When ending on an UNSAT self.size should be incremented such that is equal to the last SAT size.
                    self.size = lower_bound
                    logger.debug(
                        f"End of binary search, ended on UNSAT, setting hyp size to {self.size}"
                    )

        return hypothesis

    def build_hypothesis_classic(self):
        """
        Classic L# hypothesis construction (no SMT), ported from AALpy's
        ObservationTree.build_hypothesis. Usable when the observation tree is
        fully defined (i.e. don't cares are disabled), so that every frontier
        node can be identified with a single basis candidate.
        """
        while True:
            self.find_adequate_observation_tree()
            hypothesis = self.construct_hypothesis_classic()
            counter_example = Apartness.compute_witness_in_tree_and_hypothesis_states(
                self, self.root, hypothesis.initial_state
            )
            if not counter_example:
                self.size = len(self.basis)
                return hypothesis

            self.process_counter_example(counter_example)

    def construct_hypothesis_classic(self):
        """
        Construct the hypothesis DFA directly from the basis and the
        frontier-to-basis mapping, without the SMT solver. Mirrors AALpy's
        ObservationTree.construct_hypothesis.
        """
        states = {node: DfaState(f"s{i}") for i, node in enumerate(self.basis)}
        for node, dfa_state in states.items():
            dfa_state.is_accepting = node.output

        for node, dfa_state in states.items():
            for letter in self.alphabet:
                successor = node.get_successor(letter)
                if successor is None:
                    # Transition is undefined in the reference language; it is
                    # never exercised, so a self-loop keeps the DFA complete.
                    dfa_state.transitions[letter] = dfa_state
                    continue
                if successor not in states:
                    # Frontier node: map it to its (single) basis candidate.
                    candidates = self.frontier_to_basis_dict[successor]
                    successor = next(iter(candidates))
                dfa_state.transitions[letter] = states[successor]

        self.states_list = list(states.values())
        hypothesis = Dfa(states[self.basis[0]], self.states_list)
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = hypothesis.compute_characterization_set(
            raise_warning=False
        )
        return hypothesis

    def defined_in_reference(self, inputs):
        """
        Checks whether all inputs lead to accepting states in the reference model
        If an input does not lead to an accepting state, it must not be enabled or the horizon is exceeded
        We return whether the full sequence is defined (True/False) and the prefix of the inputs up to the last accepting state

        When no reference language is used, every input sequence is considered
        defined (the SUL itself bounds the behaviour, e.g. via its horizon).
        """
        if self.reference is None:
            return True, inputs

        outputs = self.reference.compute_output_seq(
            self.reference.initial_state, inputs
        )
        if False in outputs:  # input not enabled or horizon exceeded
            idx = outputs.index(False)
            return False, inputs[:idx]
        else:
            return True, inputs

    def execute_query(self, inputs):
        """
        Executes an OQ up until the last accepting state in the reference model and inserts the observation
        Returns whether the possible OQ lead to an extension of the observation tree
        """
        defined, defined_inputs = self.defined_in_reference(inputs)
        if defined:
            outputs = self.sul.query(inputs)
            self.insert_observation_sequence(defined_inputs, outputs)
            return True

        if self.get_successor(defined_inputs) is not None:
            # Skipping OQ completely because all required info is in the obs tree
            return False
        elif self.use_dont_care:
            # logger.debug(
            #     f"Posing reduced OQ {defined_inputs[:-1]}, original OQ {inputs}"
            # )
            # CacheSUL.query returns a tuple on a cache hit but a list on a miss,
            # so coerce to list before appending the don't-care output.
            outputs = list(self.sul.query(defined_inputs)) + ["unknown"] # FIXEDBUG: removed [:-1] from defined_inputs
            self.insert_observation_sequence(defined_inputs, outputs)
            return True
        else:
            # Without don't cares we label the whole reference-defined prefix so
            # the observation tree stays fully defined (no "unknown" nodes).
            outputs = self.sul.query(defined_inputs)
            self.insert_observation_sequence(defined_inputs, outputs)
            return True

    def extend_frontier(self):
        """
        Extend the frontier self.size - len(self.basis) steps from the basis
        """
        if self.reference is None:
            length = 2 if self.size - len(self.basis) > 2 else 1
            basis_access = [self.get_access_sequence(node) for node in self.basis]
            for word in itertools.product(self.alphabet, repeat=length):
                word = list(word)
                for access in basis_access:
                    inputs = access + word
                    if self.get_successor(inputs) is None:
                        self.execute_query(inputs)
        else:
            length = self.size - len(self.basis)
            max_words = (
                MooreNode._id_counter * 0.1
            )  # extend the frontier by at most 10% of the number of nodes in the tree to avoid exponential blowup

            bfs_stack = deque([])
            seen_states = set()

            for basis_node in self.basis:
                basis_access_seq = self.get_access_sequence(basis_node)
                basis_error = self.reference.execute_sequence(
                    self.reference.initial_state, basis_access_seq
                )
                if basis_error is False or (
                    isinstance(basis_error, list) and basis_error[-1] is False
                ):
                    continue

                bfs_stack.append((self.reference.current_state, basis_access_seq, 0))
                seen_states.add(self.reference.current_state)

            count = 0
            while bfs_stack:
                if count > max_words:
                    break

                current_state, access_seq, depth = bfs_stack.popleft()
                count += 1

                for letter in self.alphabet:
                    next_state = current_state.transitions.get(letter)
                    if (
                        next_state is None
                        or next_state in seen_states
                        or not next_state.is_accepting
                    ):
                        continue

                    if self.get_successor(access_seq + [letter]) is None:
                        self.execute_query(access_seq + [letter])

                    if depth + 1 < length:
                        bfs_stack.append((next_state, access_seq + [letter], depth + 1))
                        seen_states.add(next_state)

    def update_frontier(self):
        self.update_frontier_to_basis_dict()

    def find_adequate_observation_tree(self):
        """
        Tries to find an observation tree,
        for which each frontier state is identified as much as possible.
        """
        if self.integrate_testing and self.reference:
            self.find_adequate_observation_tree_TESTING()
        else:
            self.find_adequate_observation_tree_TRADITIONAL()

    def find_adequate_observation_tree_TRADITIONAL(self):
        """
        Tries to find an observation tree,
        for which each frontier state is identified as much as possible.
        """
        self.extend_frontier()
        self.update_frontier_to_basis_dict()
        while self.promote_node_to_basis():
            self.extend_frontier()
            self.update_frontier_to_basis_dict()

        logger.debug(f"Extended and promoted frontier.")

        while self.make_frontiers_identified():
            self.update_frontier_to_basis_dict()
            while self.promote_node_to_basis():
                self.extend_frontier()
                self.update_frontier_to_basis_dict()

    def find_adequate_observation_tree_TESTING(self):
        """
        Tries to find an observation tree,
        for which each frontier state is identified as much as possible.
        """
        while True:
            self.explore_frontier()
            self.update_frontier_to_basis_dict()
            while self.promote_node_to_basis():
                self.explore_frontier()
                self.update_frontier_to_basis_dict()

            while self.make_frontiers_identified():
                self.update_frontier_to_basis_dict()
                while self.promote_node_to_basis():
                    self.extend_frontier()
                    self.update_frontier_to_basis_dict()

            old_basis_size = len(self.basis)
            self.testing()
            self.update_frontier_to_basis_dict()
            while self.promote_node_to_basis():
                self.extend_frontier()
                self.update_frontier_to_basis_dict()
            logger.debug(f"Old basis size {old_basis_size} new {len(self.basis)}")
            if len(self.basis) == old_basis_size:  # if no new states are found by testing we stop
                return

    def explore_frontier(self):
        """ 
        Iterates over all basis states and inputs and poses a query to ensure that all frontier states that are allowed according to the reference language are all defined.
        """
        # Compute sepseq that separates first two basis states to get some free identification
        sep_seq = []
        if len(self.basis) > 1:
            sep_seq = Apartness.compute_witness(
                self.basis[0], self.basis[1], self)
        # Loop over basis and inputs, only pose queries when the frontier is not defined and should be defined according to reference
        for basis_state in self.basis:
            basis_access_seq = self.get_access_sequence(basis_state)
            basis_error = self.reference.execute_sequence(
                self.reference.initial_state, basis_access_seq
            )
            if basis_error is False or (isinstance(basis_error, list) and basis_error[-1] is False):
                continue
            for inp in self.alphabet:
                frontier_error = self.reference.execute_sequence(
                    self.reference.initial_state, basis_access_seq + [inp])
                # frontier error must be a list
                if basis_state.get_successor(inp) is None and False not in frontier_error:
                    self.execute_query(basis_access_seq + [inp] + sep_seq)

    def testing(self):
        logger.debug("Start testing phase before hypothesis construction")
        char_list = self.construct_characterization_set()
        if self.full_testing:
            test_suite = self.test_suite_construction(char_list)
            for test in test_suite:
                self.execute_query(test)
        else:
            self.randomized_test_suite(char_list)

    def randomized_test_suite(self, char_list):
        """ 
        Performs self.test_per_frontier tests per frontier state.
        Each test goes to the frontier state, performs a number of steps depending on a geometric distribution with self.depth as expected length and finally a separating sequence.
        """
        for f in list(self.frontier_to_basis_dict.keys()):
            counter = 0
            while counter < self.test_per_frontier:
                test = self.get_access_sequence(f)
                _ = self.reference.execute_sequence(
                    self.reference.initial_state, test)
                reference_state = self.reference.current_state
                limit = 2
                while limit > 0 or random.random() > 1 / (self.depth + 1):
                    alp = [
                        i
                        for i in self.alphabet
                        if i in reference_state.transitions
                        and reference_state.transitions[i].is_accepting
                    ]
                    if len(alp) == 0:
                        break
                    letter = random.choice(alp)
                    reference_state = reference_state.transitions[letter]
                    test.append(letter)
                    limit -= 1
                if len(char_list) > 0:
                    test += random.choice(char_list)
                self.execute_query(test)
                counter += 1

    def construct_characterization_set(self):
        """
        Constructs a set of separating sequences for the basis states
        """
        char_list = []
        for b1 in self.basis:
            for b2 in self.basis:
                if b1 != b2:
                    wit = Apartness.compute_witness(b1, b2, self)
                    if wit not in char_list:
                        char_list.append(wit)
        return [tuple(s) for s in char_list if not any(s != t and t[:len(s)] == s for t in char_list)]

    def test_suite_construction(self, char_list):
        """
        Yields test sequences with depth self.depth
        """
        for f in list(self.frontier_to_basis_dict.keys()):
            yield from self.recursive_test_suite_construction(
                tuple(self.get_access_sequence(f)), self.depth, char_list, [])

    def recursive_test_suite_construction(self, prefix, depth, char_list, sequences):
        """ 
        Constructs tests for the given prefix and calls the function recursively as long as the prefix + inp is defined according to the reference and the depth > 0
        """
        if depth > 0:
            for inp in self.alphabet:
                new_pref = prefix + tuple([inp],)
                if False not in self.reference.execute_sequence(self.reference.initial_state, new_pref):
                    yield new_pref
                else:
                    yield from self.recursive_test_suite_construction(new_pref, depth - 1, char_list, sequences)

        for c in char_list:
            yield prefix + c

    def to_dot(self) -> str:
        """
        Render the observation tree as a DOT (Graphviz) string.

        Node styling:
          Shape (based on output):
            - True output     → doublecircle (accepting)
            - False output    → circle (rejecting)
            - "unknown" output → box
            - None output     → diamond (unqueried)

          Color (based on node type):
            - Basis nodes     → blue
            - Frontier nodes  → green
            - Other nodes     → grey

        Node label includes:
          - Node id and access sequence
          - Output (True/False/unknown/None)

        For frontier nodes, the label also lists the basis candidates
        from frontier_to_basis_dict (shown as basis node indices).

        Edges are labelled with the input symbol that causes the transition.
        """
        basis_set = set(self.basis)
        frontier_set = set(self.frontier_to_basis_dict.keys())

        lines = ["digraph ObservationTree {", "    rankdir=LR;"]

        # BFS to collect all nodes
        queue = deque([self.root])
        visited = set()
        all_nodes = []
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            all_nodes.append(node)
            for successor in node.successors.values():
                queue.append(successor)

        for node in all_nodes:
            access = self.get_access_sequence(node)
            out_str = str(node.output) if node.output is not None else "?"

            # Shape based on output
            if node.output is True:
                shape = "doublecircle"
            elif node.output is False:
                shape = "circle"
            elif node.output == "unknown":
                shape = "box"
            else:  # None
                shape = "diamond"

            # Color based on node type
            if node in basis_set:
                color = "blue"
            elif node in frontier_set:
                color = "green"
            else:
                color = "grey"

            label = f"#{node.id}\\n{out_str}"

            # Add basis candidates for frontier nodes
            if node in frontier_set:
                candidates = self.frontier_to_basis_dict[node]
                label += f"\\n{' '.join(str(cid.id) for cid in candidates)}"

            lines.append(
                f'    n{node.id} [label="{label}", shape={shape}, color={color}];'
            )

        # Edges
        for node in all_nodes:
            for inp, successor in node.successors.items():
                lines.append(f'    n{node.id} -> n{successor.id} [label="{inp}"];')

        lines.append("}")
        return "\n".join(lines)

    def process_counter_example(self, cex_inputs):
        """
        Inserts the counter example into the observation tree and searches for the
        input-output sequence which is different
        """
        # logger.debug(f"Processing counterexample {cex_inputs}")
        self.execute_query(cex_inputs)
        self.update_frontier_to_basis_dict()
        return
