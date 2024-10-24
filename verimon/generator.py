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
from stormpy import (
    parse_properties,
    model_checking,
    SparsePomdp,
    SchedulerChoice,
    Scheduler,
    SparseDtmc,
    SparseMdp,
)
from stormpy.pomdp import (
    GenerateMonitorVerifierDouble,
    GenerateMonitorVerifierDoubleOptions,
)
from stormpy.simulator import create_simulator, SimulatorActionMode
from stormpy.utility import ShortestPathsGenerator
from stormvogel.mapping import stormvogel_to_stormpy
from stormvogel.model import (
    ModelType,
)

from verimon.utils import get_pos, logger


class Verifier:
    def __init__(self: Self, mc: SparseDtmc, mon: SparseMdp, good_label: str) -> None:
        self.mc = mc
        self.mon = mon
        self.good_label = good_label

        options = GenerateMonitorVerifierDoubleOptions()
        options.good_label = good_label
        options.step_prefix = "step="
        self.generator = GenerateMonitorVerifierDouble(mc, mon, options)

    def apply_spec(self: Self, spec: str):
        """
        Add the good label to all states where the probability of spec is above threshold.

        :param spec: A string containing the LTL specification used to determine good states
        :param threshold: A float in [0,1] which greater equal compared against the probability of spec at each state
        """
        mc = stormvogel_to_stormpy(self.mc)
        prop = parse_properties(spec)
        result = model_checking(mc, prop[0])
        states = []
        for s_id, s in self.mc.states.items():
            if result.at(s_id):
                # Works since this attribute gets added during the stormvogel to stormpy conversion
                states.append([l for l in s.labels if l.startswith("[")])
                s.add_label(self.good_label)
        logger.info(f"New good states become: {states}")

    def create_product(self: Self):
        self.pomdp = self.generator.create_product()

    def check_storm_prop(self: Self, str_prop: str, simulate=False):
        self.pomdp.type = ModelType.MDP
        storm_mdp = stormvogel_to_stormpy(self.pomdp)
        self.pomdp.type = ModelType.POMDP
        prop = parse_properties(str_prop)
        result = model_checking(storm_mdp, prop[0], extract_scheduler=True)
        simulator = create_simulator(storm_mdp)
        scheduler = result.scheduler

        if simulate:
            print("Result of model checking", result.at(storm_mdp.initial_states[0]))
            for m in range(3):
                state, reward, labels = simulator.restart()
                old_state = state
                print(f"s{state}, labels={' '.join(labels)} -->")
                for n in range(100):
                    chosen_action = storm_mdp.states[old_state].actions[
                        scheduler.get_choice(old_state).get_deterministic_choice()
                    ]
                    state, reward, labels = simulator.step(chosen_action.id)
                    old_state = state
                    print(
                        f"-[{list(chosen_action.labels)[0]}]-> \ts{state}, labels={' '.join(labels)}"
                    )
                    if simulator.is_done():
                        print("Done")
                        break
                print(
                    "---------------------------------------------------------------\n"
                )
        return scheduler

    def check_paynt_prop(
        self: Self, str_prop: str, relative_error=0, return_all=False
    ) -> Family:
        paynt.utils.timer.GlobalTimer.start()

        formula = PrismParser.parse_property(str_prop)
        prop = paynt.verification.property.construct_property(formula, relative_error)
        specification = paynt.verification.property.Specification([prop])

        paynt.verification.property.Property.initialize()
        explicit_quotient = payntbind.synthesis.addMissingChoiceLabels(self.pomdp)
        if explicit_quotient is None:
            explicit_quotient = self.pomdp
        quotient = paynt.quotient.pomdp.PomdpQuotient(explicit_quotient, specification)

        # synthesize 1-FSC
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
            quotient, "ar"
        )
        assignment = synthesizer.synthesize(
            print_stats=False, return_all=return_all
        )  # use print_stats=False to remove synthesis summary
        if assignment is not None:
            logger.info(synthesizer.stat.get_summary())

            return assignment
        else:
            logger.info("counterexample not found")

    def simulate_paynt_assignment(self: Self, assignment: Family, tries=10000):
        simulator = create_simulator(self.pomdp)

        old_observation, reward, labels = simulator.restart()
        print(
            f"s{simulator._report_state()}, labels={' '.join(labels)}",
        )
        paths = [[]]
        while len(paths) < tries:
            if old_observation >= assignment.num_holes:
                a_options = [0]
            else:
                a_options = assignment.hole_options(old_observation)

            possible_next_states = [
                get_pos(b[1].labels)
                for b in list(
                    self.pomdp.transitions[
                        simulator._report_state()
                    ].transition.values()
                )[a_options[0]].branch
            ]

            observation, reward, labels = simulator.step(a_options[0])

            paths[-1].append(
                (
                    f"\t--[{old_observation}, {a_options[0]}:{str(assignment.hole_to_option_labels[old_observation][a_options[0]])}]-->\n"
                    f"s{simulator._report_state()}, labels={' '.join(sorted(labels))}",
                    get_pos(labels),
                    possible_next_states,
                )
            )

            if simulator._report_state() == 0:
                paths.append([])
            if simulator.is_done():
                break
            old_observation = observation

        print("\n" + "\n".join([p[0] for p in paths[-1]]))
        print(f"it took {len(paths)} tries until the goal was reached")

        return [(0, [])] + [
            (pos, next_states) for _, pos, next_states in paths[-1] if pos
        ]

    def trace_of_assignment(self: Self, assignment: Family):
        storm_mon: SparseMdp = stormvogel_to_stormpy(self.mon)
        simulator = create_simulator(storm_mon)
        simulator.set_action_mode(SimulatorActionMode.GLOBAL_NAMES)

        state, _, labels = simulator.restart()
        trace = []
        while True:
            step = int([l[5:] for l in labels if l.startswith("step=")][0])
            accepting = "accepting" in labels
            observation = self.observations[(step, accepting)]
            a_index = assignment.hole_options(observation)[0]
            action = str(assignment.hole_to_option_labels[observation][a_index])
            if action == "end":
                break

            trace.append(action)

            state, _, labels = simulator.step(action)

        return trace

    def find_trace_to_good_state(self: Self):
        self.storm_pomdp: SparsePomdp = stormvogel_to_stormpy(self.pomdp)
        spg = ShortestPathsGenerator(self.storm_pomdp, self.good_label)
        path = spg.get_path_as_list(1)
        trace = []
        for s in path:
            obs_labels = [
                l for l in self.pomdp.states[s].labels if l in self.mon.actions.keys()
            ]
            trace.append(obs_labels[0])

        return trace

    def created_induced_mc(self, assignment: Family) -> SparseDtmc:
        scheduler = self._storm_scheduler_from_paynt_assignment(
            self.storm_pomdp, assignment
        )

        self.pomdp.type = ModelType.MDP
        storm_mdp = stormvogel_to_stormpy(self.pomdp)
        self.pomdp.type = ModelType.POMDP

        induced_mc = storm_mdp.apply_scheduler(scheduler)

        return induced_mc

    def _storm_scheduler_from_paynt_assignment(
        self: Self, storm_pomdp: SparsePomdp, assignment: Family
    ) -> Scheduler:
        sched = []
        for i, obs in enumerate(storm_pomdp.observations):
            if obs >= assignment.num_holes:
                sched.append(0)
            else:
                sched.append(assignment.hole_options(obs)[0])

        scheduler = self.check_storm_prop('Pmax=? [F "init"]')
        for s_id, choice in enumerate(sched):
            scheduler.set_choice(SchedulerChoice(choice), s_id)

        return scheduler
