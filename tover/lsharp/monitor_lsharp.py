import time
from typing import Any

from aalpy.automata import Dfa
from aalpy.base import SUL, Oracle

from tover.lsharp.monitor_observation_tree import MonitorObservationTree
from tover.utils.logger import logger


def run_monitor_lsharp(
    alphabet: list[str],
    reference: Dfa[str],
    sul: SUL,
    eq_oracle: Oracle,
    solver_timeout: int | None = 200,
    replace_basis: bool = True,
    use_compatibility: bool = False,
) -> tuple[Dfa[str], dict[str, Any]]:
    ob_tree = MonitorObservationTree(
        alphabet, reference, sul, solver_timeout, replace_basis, use_compatibility
    )
    start_time = time.time()
    timeout = solver_timeout

    eq_query_time = 0
    learning_rounds = 0
    validity_queries = 0
    hypothesis = None

    while True:
        learning_rounds += 1

        if timeout and time.time() - start_time > timeout:
            logger.warning("Learning timed out after %d seconds.", timeout)
            break

        # Building the hypothesis
        hypothesis = ob_tree.build_hypothesis()

        if hypothesis is None:
            continue

        hypothesis.visualize(
            f"out/test/models/hypothesis_round_{learning_rounds}", file_type="png"
        )

        # Pose Equivalence Query
        eq_query_start = time.time()
        cex = eq_oracle.find_cex(hypothesis)
        eq_query_time += time.time() - eq_query_start
        validity_queries += 1

        if cex is None:
            break

        # Process the counterexample and start a new learning round
        ob_tree.process_counter_example(cex)

    total_time = time.time() - start_time
    smt_time = ob_tree.smt_time
    learning_time = total_time - eq_query_time - smt_time

    info = {
        "learning_rounds": learning_rounds,
        "automaton_size": hypothesis.size if hypothesis else 0,  # time
        "learning_time": learning_time,
        "smt_time": smt_time,
        "eq_oracle_time": eq_query_time,
        "total_time": total_time,
        # learning algorithm
        "queries_learning": sul.num_queries,
        "validity_query": validity_queries,  # tree
        "nodes": ob_tree.get_size(),
        "informative_nodes": ob_tree.count_informative_nodes(),  # system under learning
        "sul_steps": sul.num_steps,
        "cache_saved": sul.num_cached_queries,  # eq_oracle
        "queries_eq_oracle": eq_oracle.num_queries,
        "steps_eq_oracle": eq_oracle.num_steps,
    }

    return hypothesis, info
