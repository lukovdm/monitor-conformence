import itertools
import time
from collections import deque

from aalpy.automata import Dfa, DfaState
from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import GE, LT, Bool, Function, Int, Or, Solver, Symbol
from pysmt.typing import BOOL, INT, FunctionType

from tover.lsharp.apartness import Apartness
from tover.lsharp.moore_node import MooreNode
from tover.utils.logger import logger


class MonitorObservationTree:
    def __init__(
        self, alphabet, reference, sul, solver_timeout, replace_basis, use_compatibility
    ):
        """
        Initializes the observation tree with a root node.
        """
        self.automaton_type = "dfa"
        self.solver_timeout = solver_timeout * 1000
        self.replace_basis = replace_basis
        self.use_compatibility = use_compatibility

        # Logger information
        self.smt_time = 0
        MooreNode._id_counter = 0

        # Initialize tree
        self.alphabet = alphabet
        self.reference = reference
        self.sul = sul
        self.outputAlphabet = [True, False, "unknown"]
        self.states_list = []

        self.root = MooreNode()
        # assuming querying an empty list returns a singleton list
        self.root.set_output(self.sul.query([])[0])

        self.size = 1
        self.basis = [self.root]
        self.frontier_to_basis_dict = dict()

    def insert_observation_sequence(self, inputs, outputs):
        """
        Insert an observation into the tree using a sequence of inputs and their corresponding outputs.
        """
        node = self.root
        for inp, output in zip(inputs, outputs):
            node = node.extend_and_get(inp, output)
            node.set_output(output)
            if node not in self.frontier_to_basis_dict:
                candidates = {candidate for candidate in self.basis}
                self.frontier_to_basis_dict[node] = candidates

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
                return None
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
                return None
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
            if len(self.frontier_to_basis_dict[node]) == 0:
                return
        for successor in node.successors.values():
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
                for node, candidates in self.frontier_to_basis_dict.items():
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
                candidate = next(iter(self.frontier_to_basis_dict[iso_frontier_node]))
                if len(self.get_access_sequence(candidate)) <= len(
                    self.get_access_sequence(iso_frontier_node)
                ):
                    continue
                self.basis.remove(candidate)
                self.basis.append(iso_frontier_node)
                # Update the candidates
                del self.frontier_to_basis_dict[iso_frontier_node]
                for node, candidates in self.frontier_to_basis_dict.items():
                    if candidate in candidates:
                        candidates.remove(candidate)
                    candidates.add(iso_frontier_node)
                self.frontier_to_basis_dict[candidate] = {node for node in self.basis}
                return True
        return False

    def make_frontiers_identified(self):
        """
        Loop over all frontier nodes to identify them
        """
        extended = False
        for basis_node in self.basis:
            for letter in self.alphabet:
                # not defined if rejecting in reference
                if basis_node.get_successor(letter):
                    frontier_node = basis_node.get_successor(letter)
                    if frontier_node in self.basis:
                        continue
                    while self.identify_frontier(frontier_node):
                        extended = True
                        self.update_basis_candidates(frontier_node)
        return extended

    def identify_frontier(self, frontier_node):
        """
        Identify a specific frontier node
        """
        if len(self.frontier_to_basis_dict[frontier_node]) == 0:
            return False

        inputs_to_frontier = self.get_transfer_sequence(self.root, frontier_node)

        witnesses = self._get_witnesses_bfs(frontier_node)
        for witness_seq in witnesses:
            inputs = inputs_to_frontier + witness_seq
            extended = self.execute_query(inputs)
            if extended:
                return True
        return False

    def _get_witnesses_bfs(self, frontier_node):
        """
        Specifically identify frontier nodes using separating sequences
        """
        basis_candidates = self.frontier_to_basis_dict.get(frontier_node)
        witnesses = Apartness.get_distinguishing_sequences(basis_candidates, self)

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
        self.construct_hypothesis_states(output_mapping=output_mapping)
        self.construct_hypothesis_transitions(transition_mapping=transition_mapping)

        hypothesis = Dfa(self.states_list[0], self.states_list)
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = hypothesis.compute_characterization_set(
            raise_warning=False
        )

        return hypothesis

    def find_hypothesis(self):
        """
        Find a hypothesis consistent with the observation tree, using the pySMT solver.
        There are 2 free functions: "out" and "m" and 1 bound function "delta".
        """
        logger.debug(f"Trying to build hypothesis of size {self.size}")
        logger.debug(
            f"Basis size: {len(self.basis)}, Frontier size: {len(self.frontier_to_basis_dict)}"
        )
        start_smt_time = time.time()

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

        # Flatten the tree to a list of nodes
        queue = deque([self.root])
        nodes = [self.root]

        while queue:
            node = queue.popleft()
            idx = nodes.index(node)
            for letter, successor in node.successors.items():
                # Check if successor can reach a known node
                queue.append(successor)
                s.add_assertion(
                    Function(states_mapping, [Int(len(nodes))]).Equals(
                        Function(
                            delta,
                            [
                                Function(states_mapping, [Int(idx)]),
                                Int(self.alphabet.index(letter)),
                            ],
                        )
                    )
                )
                nodes.append(successor)

        # Basis nodes map to different states
        for i, node in enumerate(self.basis):
            s.add_assertion(
                Function(states_mapping, [Int(nodes.index(node))]).Equals(Int(i))
            )

        # Force known outputs
        for i, node in enumerate(nodes):
            if self.is_known(node):
                val = Bool(node.output)
                s.add_assertion(
                    Function(dfa_output, [Function(states_mapping, [Int(i)])]).Iff(val)
                )

        for node, candidates in self.frontier_to_basis_dict.items():
            if node not in nodes:
                continue
            s.add_assertion(
                Or(
                    [
                        Function(states_mapping, [Int(nodes.index(node))]).Equals(
                            Int(self.basis.index(c))
                        )
                        for c in candidates
                    ]
                    + [
                        Function(states_mapping, [Int(nodes.index(node))]).Equals(
                            Int(i)
                        )
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
        self.find_adequate_observation_tree()
        transition_mapping, output_mapping = self.find_hypothesis()
        if transition_mapping is not None:
            hypothesis = self.construct_hypothesis(
                transition_mapping=transition_mapping, output_mapping=output_mapping
            )
            return hypothesis
        else:
            self.size += 1
            return None

    def defined_in_reference(self, inputs):
        """
        Checks whether all inputs lead to accepting states in the reference model
        If an input does not lead to an accepting state, it must not be enabled or the horizon is exceeded
        We return whether the full sequence is defined (True/False) and the prefix of the inputs up to the last accepting state
        """
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
        else:
            logger.debug(
                f"Posing reduced OQ {defined_inputs[:-1]}, original OQ {inputs}"
            )
            outputs = self.sul.query(defined_inputs[:-1]) + ["unknown"]
            self.insert_observation_sequence(defined_inputs, outputs)
            return True

    def extend_frontier(self):
        """
        Extend the frontier self.size - len(self.basis) steps from the basis
        """
        length = self.size - len(self.basis) + 1  # used to be a 3 here?
        # Loop over words of length 'length'
        for word in itertools.product(self.alphabet, repeat=length):
            for node in self.basis:
                access = self.get_access_sequence(node)
                inputs = access + list(word)
                if self.get_successor(inputs) is None:
                    self.execute_query(inputs)

    def update_frontier(self):
        self.update_frontier_to_basis_dict()

    def find_adequate_observation_tree(self):
        """
        Tries to find an observation tree,
        for which each frontier state is identified as much as possible.
        """
        self.extend_frontier()
        self.update_frontier_to_basis_dict()
        while self.promote_node_to_basis():
            self.extend_frontier()
            self.update_frontier_to_basis_dict()

        while self.make_frontiers_identified():
            self.update_frontier_to_basis_dict()
            while self.promote_node_to_basis():
                self.extend_frontier()
                self.update_frontier_to_basis_dict()

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
                candidate_ids = sorted(
                    self.basis.index(c) for c in candidates if c in basis_set
                )
                label += f"\\n{" ".join(str(cid) for cid in candidate_ids)}"

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
        logger.debug(f"Processing counterexample {cex_inputs}")
        self.execute_query(cex_inputs)
        self.update_frontier_to_basis_dict()
        return
