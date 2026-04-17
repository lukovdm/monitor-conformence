from enum import StrEnum
from typing import Any, cast

from aalpy import Dfa, run_Lstar
from aalpy.learning_algs.deterministic.LSharp import run_Lsharp
from aalpy.oracles.WpMethodEqOracle import RandomWpMethodEqOracle
from stormpy import (
    ExpressionManager,
    SparseDtmc,
)

from tover.core.oracles import OracleStats, ToVerEqOracle
from tover.core.sul import FilteringSUL
from tover.core.synthesis import ConditionalMethod
from tover.core.transformations import language_of_hmm
from tover.lsharp.monitor_lsharp import run_monitor_lsharp
from tover.lsharp.monitor_wp_method import (
    MonitorRandomWpMethodEqOracle,
)


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
    good_label: str,
    threshold: float,
    # Monitor parameters
    horizon: int,
    fp_slack: float,
    fn_slack: float,
    relative_error: float,
    # Behavior flags
    use_risk: bool = True,
    use_dont_care: bool = True,
    use_horizon_in_filtering: bool = True,
    random_eq_method: dict[str, int] | None = None,
    use_reference_language: bool = True,
    conditional_method: ConditionalMethod = ConditionalMethod.REJECTION,
    learning_method: LearningMethod = LearningMethod.LSHARP,
    # Timeouts
    solver_timeout: int = 200,
    learning_timeout: int | None = 100000,
    # Optional components
    export_benchmarks: bool = False,
    base_dir: str | None = None,
) -> tuple[tuple[Dfa[str], dict[str, Any]], OracleStats]:
    """Run the ToVer L#-based monitor learning algorithm."""
    sul = FilteringSUL(
        mc,
        initial_observation,
        alphabet,
        spec,
        (threshold + fn_slack, threshold - fp_slack) if use_dont_care else threshold,
        horizon if use_horizon_in_filtering else None,
        use_risk,
        use_dont_care,
    )

    if use_reference_language and learning_method == LearningMethod.LSHARP:
        refrence = language_of_hmm(mc, alphabet, horizon)
    else:
        refrence = None

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
        good_label,
        relative_error,
        use_risk,
        expression_manager,
        random_eq,
        base_dir,
        export_benchmarks,
        conditional_method,
    )

    if learning_method == LearningMethod.LSTAR:
        return (
            cast(
                tuple[Dfa[str], dict[str, Any]],
                run_Lstar(
                    alphabet,
                    sul,
                    eq_oracle,
                    automaton_type="dfa",
                    return_data=True,
                    print_level=2,
                ),
            ),
            eq_oracle.stats,
        )
    elif learning_method == LearningMethod.LSHARP:
        if refrence is not None:
            return (
                run_monitor_lsharp(
                    alphabet,
                    refrence,
                    sul,
                    eq_oracle,
                    solver_timeout=solver_timeout,
                    learning_timeout=learning_timeout,
                ),
                eq_oracle.stats,
            )
        else:
            return (
                cast(
                    tuple[Dfa[str], dict[str, Any]],
                    run_Lsharp(
                        alphabet,
                        sul,
                        eq_oracle,
                        automaton_type="dfa",
                        return_data=True,
                        print_level=2,
                    ),
                ),
                eq_oracle.stats,
            )
