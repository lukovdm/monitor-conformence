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
from tover.core.sul import FilteringSUL, ScalarQuerySUL
from tover.core.synthesis import ConditionalMethod
from tover.core.transformations import language_of_hmm, prune_monitor_with_reference
from tover.lsharp.monitor.monitor_lsharp import run_monitor_lsharp
from tover.lsharp.box.LSharpSquare import run_lsharp_square
from tover.lsharp.monitor.monitor_observation_tree import SMTBehaviour
from tover.lsharp.monitor.monitor_wp_method import (
    MonitorRandomWpMethodEqOracle, RandomWpMethodEqOracle
)
from tover.core.seeding import compute_seed_sequences
from tover.utils.logger import logger


class LearningMethod(StrEnum):
    """Which learning algorithm `run_tover` runs, and with it how the SUL answers.

    Don't cares and the reference language used to be free-standing flags, but
    only two of their four combinations were ever meaningful for L#, so they are
    now implied by the method: `LSHARP` learns a fully defined language, while
    `LSHARP_MONITOR` learns with both.
    """

    LSTAR = "lstar"
    # Plain L# (`tover/lsharp/normal`) on a fully defined SUL.
    LSHARP = "lsharp"
    # ToVer's L# (`tover/lsharp/monitor`): don't cares plus the reference language.
    LSHARP_MONITOR = "lsharp_monitor"
    # Baseline: the unmodified L#box of Laumen, Snel and Vaandrager, run on the
    # same filtering SUL and the same PAYNT equivalence oracle as ToVer. Learns
    # with don't cares (its premise) but has no notion of a reference language.
    LSHARP_BOX = "lsharp_box"

    @property
    def use_dont_care(self) -> bool:
        """Whether the SUL reports "unknown" inside the dead zone around the threshold."""
        return self in (LearningMethod.LSHARP_MONITOR, LearningMethod.LSHARP_BOX)

    @property
    def use_reference_language(self) -> bool:
        """Whether learning is restricted to the traces the HMM can actually produce."""
        return self is LearningMethod.LSHARP_MONITOR


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
    use_horizon_in_filtering: bool = True,
    random_eq_method: dict[str, int] | None = None,
    conditional_method: ConditionalMethod = ConditionalMethod.REJECTION,
    learning_method: LearningMethod = LearningMethod.LSHARP_MONITOR,
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
    # Both are implied by the method; see `LearningMethod`. Coerce first, since
    # callers that read the method out of YAML or JSON pass a plain string.
    learning_method = LearningMethod(learning_method)
    use_dont_care = learning_method.use_dont_care
    use_reference_language = learning_method.use_reference_language
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
        if learning_method is not LearningMethod.LSHARP_MONITOR:
            logger.warning(
                "seed_from_search is set but learning_method=%s does not use the monitor "
                "observation tree; seeding will be skipped.",
                learning_method,
            )
        else:
            seed_sequences, seeding_stats = compute_seed_sequences(
                sul,
                horizon=horizon,
                max_expansions=seed_max_expansions,
                timeout_ms=seed_timeout_ms,
            )

    if use_reference_language:
        reference_start = time()
        refrence = language_of_hmm(mc, alphabet, horizon)
        reference_language_time = time() - reference_start
    else:
        refrence = None
        reference_language_time = 0.0

    # The sampling oracle is chosen by method, not by whether a reference language
    # happens to exist: the monitor oracle draws its test sequences from the reference
    # language, so it belongs to the one method that learns against one.
    if random_eq_method is None:
        random_eq = None
    elif learning_method is LearningMethod.LSHARP_MONITOR:
        assert refrence is not None
        random_eq = MonitorRandomWpMethodEqOracle(
            alphabet, sul, refrence, **random_eq_method
        )
    else:
        random_eq = RandomWpMethodEqOracle(alphabet, sul, **random_eq_method)

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
        logger.info("Running plain L#, without reference language or dont_care.")
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
    elif learning_method == LearningMethod.LSHARP_MONITOR:
        logger.info(
            "Running L# with reference language and dont_care. "
            "This will use the monitor observation tree."
        )
        assert refrence is not None
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
        before = learned_monitor.size
        learned_monitor = prune_monitor_with_reference(learned_monitor, refrence)
        logger.info(
            f"Pruned monitor with reference language: "
            f"{before} -> {learned_monitor.size} states."
        )
        return (learned_monitor, info), eq_oracle.stats
    elif learning_method == LearningMethod.LSHARP_BOX:
        # Baseline: upstream L#box, unmodified. It learns with don't cares but has
        # no notion of a reference language.
        logger.info(
            f"Running the L#box baseline. Note that "
            f"upstream uses solver_timeout ({solver_timeout}s) both as the SMT "
            f"solver timeout and as the total learning budget, so learning_timeout "
            f"({learning_timeout}) does not apply here."
        )
        learned_monitor, info = cast(
            tuple[Dfa[str] | None, dict[str, Any]],
            run_lsharp_square(
                alphabet,
                # L#box expects one output per query, not one per input symbol.
                ScalarQuerySUL(sul_cached),
                eq_oracle,
                return_data=True,
                solver_timeout=solver_timeout,
            ),
        )
        if learned_monitor is None:
            # Upstream returns None when the budget runs out on a round whose SMT
            # call came back UNSAT: it drops the previous hypothesis and only keeps
            # the one built this round. Fail loudly instead of handing back None.
            raise RuntimeError(
                f"The L#box baseline hit its {solver_timeout}s budget without "
                f"producing a hypothesis ({info['learning_rounds']} rounds, "
                f"{info['nodes']} tree nodes); raise solver_timeout."
            )
        return (learned_monitor, info), eq_oracle.stats
    else:
        raise ValueError(f"Unknown learning method: {learning_method}")
