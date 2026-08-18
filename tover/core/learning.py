from enum import StrEnum
from time import time
from typing import Any, cast

from aalpy import Dfa, run_Lstar
from tover.lsharp.normal.LSharp import run_Lsharp
from aalpy.base.SUL import CacheSUL
from stormpy import (
    ExpressionManager,
    SparseDtmc,
)

from tover.core.oracles import OracleStats, ToVerEqOracle
from tover.core.sul import FilteringSUL
from tover.core.synthesis import ConditionalMethod
from tover.core.transformations import language_of_hmm, prune_monitor_with_reference
from tover.lsharp.monitor.monitor_lsharp import run_monitor_lsharp
from tover.lsharp.monitor.monitor_observation_tree import SMTBehaviour
from tover.lsharp.monitor.monitor_wp_method import (
    MonitorRandomWpMethodEqOracle, RandomWpMethodEqOracle
)
from tover.core.seeding import compute_seed_sequences
from tover.utils.logger import logger


class LearningMethod(StrEnum):
    LSTAR = "lstar"
    LSHARP = "lsharp"


def run_tover(
    # Core model inputs
    mc: SparseDtmc,
    alphabet: list[str],
    initial_observation: str,
    expression_manager: ExpressionManager,
    # Specification
    spec: str,
    threshold: float,
    # Monitor parameters
    horizon: int,
    fp_slack: float,
    fn_slack: float,
    relative_error: float,
    # Behavior flags
    use_dont_care: bool = False,
    use_horizon_in_filtering: bool = True,
    random_eq_method: dict[str, int] | None = None,
    use_reference_language: bool = True,
    conditional_method: ConditionalMethod = ConditionalMethod.REJECTION,
    learning_method: LearningMethod = LearningMethod.LSHARP,
    integrate_testing: bool = False,
    depth: int | None = 2,
    full_testing: bool = False,
    test_per_frontier: int | None = 5, 
    smt_behaviour: SMTBehaviour = SMTBehaviour.SEQUENTIAL,
    # Seeding the observation tree from the belief most-probable-path search
    seed_from_search: bool = False,
    seed_max_expansions: int = 100000,
    seed_timeout_ms: int = 0,
    # Timeouts
    solver_timeout: int = 200,
    learning_timeout: int | None = 100000,
    # Optional components
    export_benchmarks: bool = False,
    base_dir: str | None = None,
) -> tuple[tuple[Dfa[str], dict[str, Any]], OracleStats]:
    """Run the ToVer L#-based monitor learning algorithm."""
    logger.info(
        f"Running ToVer with spec: {spec}, threshold: {threshold}, "
        f"fp_slack: {fp_slack}, fn_slack: {fn_slack}, relative_error: {relative_error}, "
        f"exact: {mc.is_exact}, "
        f"use_dont_care: {use_dont_care}, use_horizon_in_filtering: {use_horizon_in_filtering}, "
        f"random_eq_method: {random_eq_method}, use_reference_language: {use_reference_language}, "
        f"conditional_method: {conditional_method}, learning_method: {learning_method}, smt_behaviour: {smt_behaviour}, "
        f"integrate_testing: {integrate_testing}, depth {depth}, full_testing {full_testing}, test_per_frontier {test_per_frontier}"
    )
    print(f"Using alphabet of size {len(alphabet)}: {alphabet}")
    sul = FilteringSUL(
        mc,
        initial_observation,
        alphabet,
        spec,
        (threshold - fp_slack, threshold + fn_slack) if use_dont_care else threshold,
        horizon if use_horizon_in_filtering else None,
        use_dont_care,
    )
    sul_cached = CacheSUL(sul)

    # Seed the observation tree from the belief most-probable-path search. Only the monitor L#
    # path owns an observation tree; the other algorithms have nowhere to put the seeds.
    seed_sequences = None
    seeding_stats = None
    if seed_from_search:
        uses_observation_tree = learning_method == LearningMethod.LSHARP and (
            use_reference_language or use_dont_care
        )
        if not uses_observation_tree:
            logger.warning(
                "seed_from_search is set but this configuration does not use the monitor "
                "observation tree (learning_method=%s, use_reference_language=%s, "
                "use_dont_care=%s); seeding will be skipped.",
                learning_method,
                use_reference_language,
                use_dont_care,
            )
        else:
            seed_sequences, seeding_stats = compute_seed_sequences(
                sul,
                horizon=horizon,
                max_expansions=seed_max_expansions,
                timeout_ms=seed_timeout_ms,
            )

    if use_reference_language and learning_method == LearningMethod.LSHARP:
        reference_start = time()
        refrence = language_of_hmm(mc, alphabet, horizon)
        reference_language_time = time() - reference_start
    else:
        refrence = None
        reference_language_time = 0.0

    if random_eq_method is not None:
        if refrence is not None:
            random_eq = MonitorRandomWpMethodEqOracle(
                alphabet, sul, refrence, **random_eq_method
            )
        else:
            random_eq = RandomWpMethodEqOracle(alphabet, sul, **random_eq_method)
    else:
        random_eq = None

    eq_oracle = ToVerEqOracle(
        alphabet,
        sul,
        mc,
        threshold,
        fp_slack,
        fn_slack,
        horizon,
        spec,
        relative_error,
        expression_manager,
        random_eq,
        base_dir,
        export_benchmarks,
        conditional_method,
    )
    eq_oracle.stats.reference_language_time = reference_language_time
    eq_oracle.stats.reference_size = refrence.size if refrence is not None else 0
    if seeding_stats is not None:
        eq_oracle.stats.seeding = seeding_stats.as_dict()

    if learning_method == LearningMethod.LSTAR:
        return (
            cast(
                tuple[Dfa[str], dict[str, Any]],
                run_Lstar(
                    alphabet,
                    sul_cached,
                    eq_oracle,
                    automaton_type="dfa",
                    return_data=True,
                    print_level=2,
                ),
            ),
            eq_oracle.stats,
        )
    elif learning_method == LearningMethod.LSHARP:
        if not use_reference_language and not use_dont_care:
            logger.info(
                "Running L# without reference language and without dont_care. ")
            return (
                cast(
                    tuple[Dfa[str], dict[str, Any]],
                    run_Lsharp(
                        alphabet,
                        sul_cached,
                        eq_oracle,
                        automaton_type="dfa",
                        separation_rule="SepSeq",
                        return_data=True,
                        print_level=2,
                    ),
                ),
                eq_oracle.stats,
            )
        else:
            logger.info(
                "Running L# with reference language and/or dont_care. "
                "This will use the monitor observation tree."
            )
            learned_monitor, info = run_monitor_lsharp(
                alphabet,
                refrence,
                sul_cached,
                eq_oracle,
                solver_timeout=solver_timeout,
                learning_timeout=learning_timeout,
                use_dont_care=use_dont_care,
                smt_behaviour=smt_behaviour,
                integrate_testing=integrate_testing,
                depth=depth,
                full_testing=full_testing,
                test_per_frontier=test_per_frontier,
                seed_sequences=seed_sequences,
            )
            # Prune transitions that are impossible under the reference language,
            # so the saved monitor only keeps behaviour on realisable traces.
            if refrence is not None:
                before = learned_monitor.size
                learned_monitor = prune_monitor_with_reference(
                    learned_monitor, refrence
                )
                logger.info(
                    f"Pruned monitor with reference language: "
                    f"{before} -> {learned_monitor.size} states."
                )
            return (learned_monitor, info), eq_oracle.stats
