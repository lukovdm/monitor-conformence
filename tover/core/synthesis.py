import logging
from enum import StrEnum
from typing import Self

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp
import paynt.synthesizer.synthesizer
import paynt.utils.timer
import paynt.verification.property
import payntbind.synthesis
from numpy._core.numeric import True_
from paynt.family.family import Family
from paynt.parser.prism_parser import PrismParser
from paynt.synthesizer.synthesizer_ar import SynthesizerAR
from stormpy import (
    ConditionalAlgorithmSetting,
    ExpressionManager,
    MinMaxMethod,
    Rational,
    SparseDtmc,
    SparseExactDtmc,
    SparseExactMdp,
    SparseMdp,
    model_checking,
    parse_properties,
)
from stormpy.pomdp import (
    GenerateMonitorVerifierDouble,
    GenerateMonitorVerifierDoubleOptions,
    GenerateMonitorVerifierExact,
    GenerateMonitorVerifierExactOptions,
)
from stormpy.simulator import SimulatorActionMode, SparseSimulator, create_simulator

from tover.utils.helpers import compact_json_str, get_pos, hole_to_observations
from tover.utils.logger import logger


class ConditionalMethod(StrEnum):
    REJECTION = "rejection"
    RESTART = "restart"
    BISECTION = "bisection"
    BISECTION_ADVANCED = "bisection_advanced"
    BISECTION_PT = "bisection_pt"
    BISECTION_ADVANCED_PT = "bisection_advanced_pt"
    POLICY_ITERATION = "policy_iteration"


CONDITIONAL_METHODS = {
    ConditionalMethod.REJECTION: ConditionalAlgorithmSetting.restart,
    ConditionalMethod.RESTART: ConditionalAlgorithmSetting.restart,
    ConditionalMethod.BISECTION: ConditionalAlgorithmSetting.bisection,
    ConditionalMethod.BISECTION_ADVANCED: ConditionalAlgorithmSetting.bisection_advanced,
    ConditionalMethod.BISECTION_PT: ConditionalAlgorithmSetting.bisection_pt,
    ConditionalMethod.BISECTION_ADVANCED_PT: ConditionalAlgorithmSetting.bisection_advanced_pt,
    ConditionalMethod.POLICY_ITERATION: ConditionalAlgorithmSetting.policy_iteration,
}


class Verifier:
    """Orchestrates verification of a monitor against a Markov chain via PAYNT synthesis.

    Creates the product construction (MC × monitor) and delegates property
    checking to PAYNT to find a scheduler (counterexample trace).
    """

    def __init__(
        self: Self,
        mc: SparseDtmc | SparseExactDtmc,
        mon: SparseMdp | SparseExactMdp,
        expr_manager: ExpressionManager,
        good_label: str,
        paynt_strategy: str = "ar",
        export_benchmarks: bool = False,
        conditional_method: ConditionalMethod = ConditionalMethod.REJECTION,
    ) -> None:
        if mc.is_exact ^ mon.is_exact:
            raise ValueError(
                f"MC and MON must have the same exactness: mc={mc.is_exact}, mon={mon.is_exact}"
            )
        self.mc = SparseExactDtmc(mc) if mc.is_exact else SparseDtmc(mc)
        self.mon = SparseExactMdp(mon) if mon.is_exact else SparseMdp(mon)

        self.expr_manager = expr_manager
        self.good_label = good_label
        self.pomdp_quotient = None
        self.pomdp = None
        self.conditional_method = conditional_method

        if mc.is_exact and paynt_strategy != "ar":
            raise ValueError("Only AR strategy is supported for exact models")
        self.paynt_strategy = paynt_strategy
        self.export_benchmarks = export_benchmarks

        self.options = (
            GenerateMonitorVerifierExactOptions()
            if self.mc.is_exact
            else GenerateMonitorVerifierDoubleOptions()
        )
        self.options.good_label = good_label
        self.options.step_prefix = "step="
        self.options.use_risk = True
        self.options.use_rejection_sampling = self.conditional_method == "rejection"

        self._rebuild_generator()

    def _rebuild_generator(self: Self):
        if self.mc.is_exact:
            self.generator = GenerateMonitorVerifierExact(
                self.mc, self.mon, self.expr_manager, self.options
            )
        else:
            self.generator = GenerateMonitorVerifierDouble(
                self.mc, self.mon, self.expr_manager, self.options
            )

    def apply_spec(self: Self, spec: str):
        """Label good states by evaluating spec on the MC, then rebuild the generator."""
        prop = parse_properties(spec)
        result = model_checking(self.mc, prop[0])
        self.mc.labeling.set_states(self.good_label, result.get_truth_values())
        logger.info(
            f"New good states become: {self.mc.labeling.get_states(self.good_label)}"
        )
        self.options.use_risk = False
        self._rebuild_generator()

    def set_risk(self: Self, risk_prop: str):
        """Set the risk function by evaluating risk_prop on the MC."""
        prop = parse_properties(risk_prop)
        result = model_checking(self.mc, prop[0])
        self.generator.set_risk(result.get_values())
        # logger.debug(f"Risk function becomes: {result.get_values()}")

    def create_product(self: Self):
        self.monitor_verifier = self.generator.create_product()
        self.pomdp = self.monitor_verifier.get_product()

    def check_storm_prop(self: Self, str_prop: str):
        prop = parse_properties(str_prop)
        result = model_checking(
            self.pomdp, prop[0], extract_scheduler=True, force_fully_observable=True
        )
        return result.scheduler

    def check_paynt_prop(
        self: Self, str_prop: str, relative_error=0
    ) -> tuple[Family | None, float | Rational | None, int | None]:
        """Run PAYNT synthesis on the product POMDP for the given property.

        Returns (assignment, value, iterations). assignment is None if no
        counterexample was found above the threshold.
        """
        assert self.pomdp is not None, "POMDP product not created yet"

        paynt.cli.setup_logger()
        paynt.utils.timer.GlobalTimer.start()

        formula = PrismParser.parse_property(str_prop)
        prop = paynt.verification.property.construct_property(
            formula, relative_error, self.pomdp.is_exact
        )
        specification = paynt.verification.property.Specification([prop])

        min_max_method = (
            MinMaxMethod.value_iteration
            if "bisection" in self.conditional_method
            else None
        )

        paynt.verification.property.Property.conditional_algorithm = (
            CONDITIONAL_METHODS[self.conditional_method]
        )

        paynt.verification.property.Property.conditional_bisection_optimization = (
            not str_prop.startswith("Pmax")
        )

        paynt.verification.property.Property.initialize(
            self.pomdp.is_exact,
            min_max_method,
        )

        if self.pomdp.is_exact:
            explicit_quotient = payntbind.synthesis.addMissingChoiceLabelsExact(
                self.pomdp
            )
        else:
            explicit_quotient = payntbind.synthesis.addMissingChoiceLabels(self.pomdp)

        if explicit_quotient is None:
            explicit_quotient = self.pomdp

        self.pomdp_quotient = paynt.quotient.pomdp.PomdpQuotient(
            explicit_quotient, specification
        )

        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
            self.pomdp_quotient, self.paynt_strategy
        )  # type: ignore
        assignment = synthesizer.synthesize(print_stats=False, keep_optimum=True)  # type: ignore

        # Clear paynt logger handlers to prevent duplicate logs on subsequent calls
        logging.getLogger().handlers.clear()

        if synthesizer.best_assignment_value == 0:
            logger.info("max probability is 0, thus no counterexample found")
            logger.debug(synthesizer.stat.get_summary())
            return None, None, synthesizer.stat.iterations_mdp

        optimum = (
            synthesizer.quotient.specification.optimality.optimum
            if synthesizer.quotient.specification.optimality
            else None
        )

        if assignment is not None:
            logger.info(f"counterexample found: {assignment} ({optimum})")
            logger.debug(synthesizer.stat.get_summary())
            return assignment, optimum, synthesizer.stat.iterations_mdp
        else:
            logger.info("no counterexamples above threshold")
            logger.debug(synthesizer.stat.get_summary())
            return None, None, synthesizer.stat.iterations_mdp

    def simulate_paynt_assignment(self: Self, assignment: Family, tries=10000):
        simulator: SparseSimulator = create_simulator(self.pomdp)
        sched = hole_to_observations(assignment)

        old_observation, reward, labels = simulator.restart()
        logger.debug(
            f"s{simulator._report_state()}, obs={old_observation}, labels={' '.join(labels)}",
        )
        paths = [[]]
        while len(paths) < tries:
            action = sched.get(old_observation, None)

            simulator.set_action_mode(SimulatorActionMode.GLOBAL_NAMES)
            try:
                current_stormpy_action = simulator.available_actions().index(action)
            except ValueError:
                current_stormpy_action = 0
            simulator.set_action_mode(SimulatorActionMode.INDEX_LEVEL)

            possible_next_states = [
                get_pos(self.pomdp.state_valuations.get_json(t.column))
                for t in self.pomdp.states[simulator._report_state()]
                .actions[current_stormpy_action]
                .transitions
            ]

            observation, reward, labels = simulator.step(current_stormpy_action)
            valuation = self.pomdp.state_valuations.get_json(simulator._report_state())

            paths[-1].append(
                (
                    f"--[{old_observation}, {action}]-->\t"
                    f"s{simulator._report_state()}, val={compact_json_str(str(valuation))}, labels={' '.join(labels)}",
                    get_pos(valuation),
                    possible_next_states,
                )
            )

            if self.pomdp.labeling.has_state_label("init", simulator._report_state()):
                paths.append([])
            if simulator.is_done():
                break
            old_observation = observation

        logger.debug("\n".join([p[0] for p in paths[-1]]))
        logger.debug(f"it took {len(paths)} tries until the goal was reached")

        return [(0, [])] + [
            (pos, next_states) for _, pos, next_states in paths[-1] if pos >= 0
        ]

    def trace_of_assignment(self: Self, assignment: Family) -> list[str]:
        """Extract the counterexample trace from a PAYNT assignment by simulating the monitor."""
        observation_map = self.monitor_verifier.observation_map
        default_action_map = self.monitor_verifier.default_action_map
        sched = hole_to_observations(assignment)

        simulator: SparseSimulator = create_simulator(self.mon)  # type: ignore
        simulator.set_action_mode(SimulatorActionMode.GLOBAL_NAMES)

        prev_state, _, labels = simulator.restart()
        trace = []
        while True:
            step = int([l[5:] for l in labels if l.startswith("step=")][0])
            accepting = "accepting" in labels
            if (step, accepting) not in observation_map:
                logger.warning("trace generation failed")
                raise Exception(
                    f"Observation ({step}, {accepting}) not found in observation map"
                )
            observation = observation_map[(step, accepting)]
            if observation not in sched:
                action = default_action_map[observation]
                logger.info(f"observation {observation} not in sched, taking {action}")
            else:
                action = sched[observation]

            if action == "end":
                break

            trace.append(action)
            state, _, labels = simulator.step(action)
            if state == prev_state:
                logger.warning("loop detected")
                raise Exception(
                    f"Loop detected during trace generation in state {state} while taking action {action}"
                )
            prev_state = state

        return trace

    def created_induced_mc(self, assignment: Family) -> SparseDtmc:
        dtmc = self.pomdp_quotient.build_assignment(assignment)
        return dtmc.model
