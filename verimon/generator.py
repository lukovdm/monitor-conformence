import logging
from typing import Self

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp
import paynt.synthesizer.synthesizer
import paynt.utils.timer
import paynt.verification.property
import payntbind.synthesis
from paynt.family.family import Family
from paynt.parser.prism_parser import PrismParser
from paynt.synthesizer.synthesizer_ar import SynthesizerAR
from stormpy import (
    parse_properties,
    model_checking,
    SparseDtmc,
    SparseMdp,
    ExpressionManager,
)
from stormpy.pomdp import (
    GenerateMonitorVerifierDouble,
    GenerateMonitorVerifierDoubleOptions,
)
from stormpy.simulator import create_simulator, SimulatorActionMode, SparseSimulator

from verimon.logger import logger
from verimon.utils import compact_json_str, get_pos, hole_to_observations


class Verifier:
    def __init__(
        self: Self,
        mc: SparseDtmc,
        mon: SparseMdp,
        expr_manager: ExpressionManager,
        good_label: str,
    ) -> None:
        self.mc = SparseDtmc(mc)
        self.mon = SparseMdp(mon)
        self.expr_manager = expr_manager
        self.good_label = good_label
        self.pomdp_quotient = None
        self.pomdp = None

        self.options = GenerateMonitorVerifierDoubleOptions()
        self.options.good_label = good_label
        self.options.step_prefix = "step="
        self.generator = GenerateMonitorVerifierDouble(
            mc, mon, expr_manager, self.options
        )

    def apply_spec(self: Self, spec: str):
        """
        Add the good label to all states where the probability of spec is above threshold.

        :param spec: A string containing the LTL specification used to determine good states
        """
        prop = parse_properties(spec)
        result = model_checking(self.mc, prop[0])
        self.mc.labeling.set_states(self.good_label, result.get_truth_values())
        logger.info(
            f"New good states become: {self.mc.labeling.get_states(self.good_label)}"
        )
        self.generator = GenerateMonitorVerifierDouble(
            self.mc, self.mon, self.expr_manager, self.options
        )

    def set_risk(self: Self, risk_prop: str):
        self.options.use_risk = True
        self.generator = GenerateMonitorVerifierDouble(
            self.mc, self.mon, self.expr_manager, self.options
        )

        prop = parse_properties(risk_prop)
        result = model_checking(self.mc, prop[0])

        self.generator.set_risk(result.get_values())
        logger.info(f"Risk function becomes: {result.get_values()}")

    def create_product(self: Self):
        with open(f"models/mc-{self.good_label}.dot", "w") as f:
            f.write(self.mc.to_dot())
        self.monitor_verifier = self.generator.create_product()
        self.pomdp = self.monitor_verifier.get_product()
        with open(f"models/pomdp-{self.good_label}.dot", "w") as f:
            f.write(self.pomdp.to_dot())

    def check_storm_prop(self: Self, str_prop: str):
        prop = parse_properties(str_prop)
        result = model_checking(
            self.pomdp, prop[0], extract_scheduler=True, force_fully_observable=True
        )
        scheduler = result.scheduler
        return scheduler

    def check_paynt_prop(
        self: Self, str_prop: str, relative_error=0, return_all=False
    ) -> Family | None:
        paynt.cli.setup_logger()
        paynt.utils.timer.GlobalTimer.start()

        formula = PrismParser.parse_property(str_prop)
        prop = paynt.verification.property.construct_property(formula, relative_error)
        specification = paynt.verification.property.Specification([prop])

        paynt.verification.property.Property.initialize()
        explicit_quotient = payntbind.synthesis.addMissingChoiceLabels(self.pomdp)
        if explicit_quotient is None:
            explicit_quotient = self.pomdp
        self.pomdp_quotient = paynt.quotient.pomdp.PomdpQuotient(
            explicit_quotient, specification
        )

        # synthesize 1-FSC
        synthesizer: SynthesizerAR = (
            paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
                self.pomdp_quotient, "ar"
            )
        )
        assignment = synthesizer.synthesize(
            print_stats=False
        )  # use print_stats=False to remove synthesis summary

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        if assignment is not None:
            logger.info(synthesizer.stat.get_summary())

            return assignment
        else:
            logger.info("counterexample not found")

    def simulate_paynt_assignment(self: Self, assignment: Family, tries=10000):
        simulator: SparseSimulator = create_simulator(self.pomdp)

        sched = hole_to_observations(assignment)

        old_observation, reward, labels = simulator.restart()
        logger.info(
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

        logger.info("\n".join([p[0] for p in paths[-1]]))
        logger.info(f"it took {len(paths)} tries until the goal was reached")

        return [(0, [])] + [
            (pos, next_states) for _, pos, next_states in paths[-1] if pos >= 0
        ]

    def trace_of_assignment(self: Self, assignment: Family):
        observation_map = self.monitor_verifier.observation_map
        default_action_map = self.monitor_verifier.default_action_map

        sched = hole_to_observations(assignment)

        with open("models/mon.dot", "w") as f:
            f.write(self.mon.to_dot())

        simulator: SparseSimulator = create_simulator(self.mon)  # type: ignore
        simulator.set_action_mode(SimulatorActionMode.GLOBAL_NAMES)

        state, _, labels = simulator.restart()
        trace = []
        while True:
            step = int([l[5:] for l in labels if l.startswith("step=")][0])
            accepting = "accepting" in labels
            if (step, accepting) not in observation_map:
                logger.warning("trace generation failed")
                raise Exception()
            observation = observation_map[(step, accepting)]
            if observation not in sched:
                logger.info(
                    f"observation {observation} not in sched {simulator.available_actions()}"
                )
                action = default_action_map[observation]
            else:
                action = sched[observation]

            if action == "end":
                break

            trace.append(action)

            state, _, labels = simulator.step(action)

        return trace

    def created_induced_mc(self, assignment: Family) -> SparseDtmc:
        dtmc = self.pomdp_quotient.build_assignment(assignment)
        return dtmc.model
