from copy import deepcopy
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
    export_to_drn,
    parse_properties,
    model_checking,
    SparsePomdp,
    SchedulerChoice,
    Scheduler,
    SparseDtmc,
    SparseMdp,
)
from stormpy.simulator import create_simulator, SimulatorActionMode
from stormpy.utility import ShortestPathsGenerator
from stormvogel.mapping import stormvogel_to_stormpy
from stormvogel.model import (
    Model,
    State,
    ModelType,
    new_pomdp,
    Transition,
    EmptyAction,
    Branch,
    Number,
)
from stormvogel.show import show

from verimon.logger import logger
from verimon.utils import get_pos


class MC_MON_Product:
    def __init__(self: Self, mc: Model, mon: Model, gb: Model, good_label: str) -> None:
        if mc.type != ModelType.DTMC:
            raise Exception("Wrong model type for the MC, should be DTMC", mc.type)
        if mon.type != ModelType.MDP:
            raise Exception("Wrong model type for the monitor, should be MDP", mon.type)
        if gb.type != ModelType.MDP:
            raise Exception(
                "Wrong model type for the Good/Bad DFA, should be MDP", gb.type
            )

        self.mc = deepcopy(mc)
        self.mon = deepcopy(mon)
        self.gb = deepcopy(gb)
        self.good_label = good_label
        self.pomdp = new_pomdp("Product")

    def __gen_observation(self: Self, state: State, dfa_id: int):
        c = len(self.normal_actions)
        for l in state.labels:
            if l in self.normal_actions.keys():
                c = sorted(self.normal_actions.keys()).index(l)
                break

        return c + dfa_id * (len(self.normal_actions) + 1)

    def __create_product_state(self: Self, mon_id: int, gb_id: int, mc_id: int):
        mc_state = self.mc.states[mc_id]
        mon_state = self.mon.states[mon_id]
        gb_state = self.gb.states[gb_id]
        labels = [f"!g{gb_id}", f"!l{mon_id}", f"!s{mc_id}"] + [
            l for l in mc_state.labels if l != "init"
        ]
        if (
            "init" in mon_state.labels
            and "init" in mc_state.labels
            and "init" in gb_state.labels
        ):
            state = self.pomdp.get_initial_state()
            for l in labels:
                state.add_label(l)
        else:
            state = self.pomdp.new_state(labels)

        if self.use_step_label:
            step = int([l[5:] for l in mon_state.labels if l.startswith("step=")][0])
            accepting = "accepting" in mon_state.labels
            state.set_observation(self.observations[(step, accepting)])
        else:
            state.set_observation(mon_id)
        self.states[(mon_id, gb_id, mc_id)] = state

    def __create_product_transition(
        self: Self,
        mon_id: int,
        gb_id: int,
        mc_id: int,
    ):
        mon_trans = self.mon.transitions[mon_id]
        mon_trans_name_dict = {
            list(a.labels)[0]: b for a, b in mon_trans.transition.items()
        }
        gb_trans = self.gb.transitions[gb_id]
        gb_trans_name_dict = {
            list(a.labels)[0]: b for a, b in gb_trans.transition.items()
        }
        mc_trans = self.mc.transitions[mc_id]
        mc_branch = mc_trans.transition[EmptyAction]

        state = self.states[(mon_id, gb_id, mc_id)]

        # Flag to indicate if we need to set the transition or add it
        set_trans = True
        for action in self.pomdp.actions.values():  # type: ignore
            if action == self.end_action:
                if "accepting" in self.mon.get_state_by_id(mon_id).labels:
                    if "happy" in self.gb.get_state_by_id(gb_id).labels:
                        if set_trans:
                            state.set_transitions([(self.end_action, self.goal_state)])
                        else:
                            state.add_transitions([(self.end_action, self.goal_state)])
                    else:
                        if set_trans:
                            state.set_transitions([(self.end_action, self.stop_state)])
                        else:
                            state.add_transitions([(self.end_action, self.stop_state)])
            elif "horizon" in self.mon.get_state_by_id(mon_id).labels:
                # If we have reached the horizon of our monitor we restart the trace
                if self.create_backlinks:
                    if set_trans:
                        state.set_transitions(
                            [(action, self.pomdp.get_initial_state())]
                        )
                        set_trans = False
                    else:
                        state.add_transitions(
                            [(action, self.pomdp.get_initial_state())]
                        )
            elif mon_branch := mon_trans_name_dict.get(action.name):
                # The action exists in the monitor and we add the transition
                if mon_branch.branch[0][0] != 1:
                    raise Exception(
                        f"Monitor state {mon_id} has probabilistic transitions on {action.name}, {mon_trans}"
                    )
                mon_dest = mon_branch.branch[0][1]

                # Create pomdp branch
                pomdp_branch: list[tuple[float, State]] = []
                for prob, mc_new_state in mc_branch.branch:
                    if action.name not in mc_new_state.labels:
                        continue
                    if gb_branch := gb_trans_name_dict.get(
                        "good" if self.good_label in mc_new_state.labels else "bad"
                    ):
                        pomdp_branch.append(
                            (
                                float(prob),
                                self.states[
                                    (
                                        mon_dest.id,
                                        gb_branch.branch[0][1].id,
                                        mc_new_state.id,
                                    )
                                ],
                            )
                        )
                    else:
                        raise Exception(
                            f"Good/Bad DFA should be complete, but {self.gb.get_state_by_id(gb_id)}"
                            f"with transitions {self.gb.get_state_by_id(gb_id).available_actions()}"
                            f"is missing a transition for {mc_new_state.labels}"
                        )

                # Calculate probability in the branch
                total_prob = sum([p for (p, _) in pomdp_branch])

                if total_prob == 0 and self.create_backlinks:
                    # Redirect to initial state if there is a transition in the monitor but not in the mc
                    trans = Transition(
                        {action: Branch([(1, self.pomdp.get_initial_state())])}
                    )
                elif total_prob > 0 and self.create_backlinks:
                    # Add branch with optional transition to the initial state if some of the transitions in this state went to a different label
                    branch: list[tuple[Number, State]] = [
                        (p, s) for (p, s) in pomdp_branch
                    ]
                    trans = Transition(
                        {
                            action: Branch(
                                branch
                                + (
                                    [(1 - total_prob, self.pomdp.get_initial_state())]
                                    if total_prob < 1
                                    else []
                                )
                            )
                        }
                    )
                else:
                    trans = Transition(
                        {
                            action: Branch(
                                [(p / total_prob, s) for (p, s) in pomdp_branch]
                            )
                        }
                    )

                if set_trans:
                    state.set_transitions(trans)
                    set_trans = False
                else:
                    state.add_transitions(trans)
            elif self.create_backlinks:
                # Redirect missing action to initial state
                if set_trans:
                    state.set_transitions([(action, self.pomdp.get_initial_state())])
                    set_trans = False
                else:
                    state.add_transitions([(action, self.pomdp.get_initial_state())])

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

    def create_product(self: Self, create_backlinks=True, use_step_label=False):
        self.create_backlinks = create_backlinks
        self.use_step_label = use_step_label

        # Create actions from labels of the markov chain and add the end trace action
        self.pomdp.actions = {}
        for mon_act in self.mon.actions.values():
            name = list(mon_act.labels)[0]
            if name not in self.pomdp.actions.keys():
                self.pomdp.new_action(name, mon_act.labels)
        self.normal_actions = self.pomdp.actions.copy()
        self.end_action = self.pomdp.new_action(name="end", labels=frozenset({"end"}))

        if self.use_step_label:
            self.observations = {}
            obs = 0
            for s in self.mon.states.values():
                step = int([l[5:] for l in s.labels if l.startswith("step=")][0])
                accepting = "accepting" in s.labels
                if (step, accepting) not in self.observations:
                    self.observations[(step, accepting)] = obs
                    obs += 1

        # Create the stop and goal state
        self.goal_state = self.pomdp.new_state("goal")
        self.stop_state = self.pomdp.new_state("stop")
        if self.use_step_label:
            self.goal_state.set_observation(max(self.observations.values()) + 1)
            self.stop_state.set_observation(max(self.observations.values()) + 2)
        else:
            self.goal_state.set_observation(len(self.mon.states) + 1)
            self.stop_state.set_observation(len(self.mon.states) + 2)

        logger.debug(
            f"Finished product setup, created {len(self.observations)} observations"
        )

        # Create the product states with appropriate labels and observations
        # They are stored in the states dict by (mon_id, gb_id, mc_id)
        self.states: dict[tuple[int, int, int], State] = {}
        for mon_id in self.mon.states.keys():
            for gb_id in self.gb.states.keys():
                for mc_id in self.mc.states.keys():
                    self.__create_product_state(mon_id, gb_id, mc_id)

        logger.debug(f"Created product {len(self.pomdp.states)} states")

        # Create transitions between states
        for mon_id in self.mon.states.keys():
            for gb_id in self.gb.states.keys():
                for mc_id in self.mc.states.keys():
                    self.__create_product_transition(
                        mon_id,
                        gb_id,
                        mc_id,
                    )

        logger.debug(
            f"Created product transitions for {len(self.pomdp.transitions)} states"
        )

        self.pomdp.add_self_loops()
        # remove_unreachable_states(self.pomdp)

    def show(self: Self):
        self.remove_unreachable_states()
        show(self.pomdp)

    def export_to_drn(self: Self, path: str):
        storm_pomdp = stormvogel_to_stormpy(self.pomdp)
        export_to_drn(storm_pomdp, path)

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
        self.storm_pomdp: SparsePomdp = stormvogel_to_stormpy(self.pomdp)

        formula = PrismParser.parse_property(str_prop)
        prop = paynt.verification.property.construct_property(formula, relative_error)
        specification = paynt.verification.property.Specification([prop])
        explicit_quotient = self.storm_pomdp

        paynt.verification.property.Property.initialize()
        explicit_quotient = payntbind.synthesis.addMissingChoiceLabels(
            explicit_quotient
        )
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
        simulator = create_simulator(self.storm_pomdp)

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
