import logging
from typing import Self

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp
import paynt.synthesizer.synthesizer
import paynt.utils.timer
import paynt.verification.property
import payntbind.synthesis
from paynt.parser.prism_parser import PrismParser
from stormpy import (
    get_maximal_end_components,
    export_to_drn,
    parse_properties,
    model_checking,
)
from stormpy.simulator import create_simulator
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


class MC_MON_Product:
    def __init__(
        self: Self, mc: Model, mon: Model, gb: Model, create_baclinks=True
    ) -> None:
        if mc.type != ModelType.DTMC:
            raise Exception("Wrong model type for the MC, should be DTMC", mc.type)
        if mon.type != ModelType.MDP:
            raise Exception("Wrong model type for the monitor, should be MDP", mon.type)
        if gb.type != ModelType.MDP:
            raise Exception(
                "Wrong model type for the Good/Bad DFA, should be MDP", gb.type
            )

        self.mc = mc
        self.mon = mon
        self.gb = gb
        self.create_backlinks = create_baclinks
        self.pomdp = new_pomdp("Product")

        # Create actions from labels of the markov chain and add the end trace action
        self.pomdp.actions = {}
        for mon_act in mon.actions.values():  # type: ignore
            name = list(mon_act.labels)[0]
            if name not in self.pomdp.actions.keys():
                self.pomdp.new_action(name, mon_act.labels)
        self.normal_actions = self.pomdp.actions.copy()
        self.end_action = self.pomdp.new_action(name="end", labels=frozenset({"end"}))

        # Create the stop and goal state
        self.goal_state = self.pomdp.new_state("goal")
        self.goal_state.new_observation(len(mon.states) + 1)
        self.stop_state = self.pomdp.new_state("stop")
        self.stop_state.new_observation(len(mon.states) + 2)

        # Create the product states with appropriate labels and observations
        # They are stored in the states dict by (mon_id, gb_id, mc_id)
        self.states: dict[tuple[int, int, int], State] = {}
        for mon_id in mon.states.keys():
            for gb_id in gb.states.keys():
                for mc_id in mc.states.keys():
                    self.__create_product_state(mon_id, gb_id, mc_id)

        # Create transitions between states
        for mon_id in mon.states.keys():
            for gb_id in gb.states.keys():
                for mc_id in mc.states.keys():
                    self.__create_product_transition(
                        mon_id,
                        gb_id,
                        mc_id,
                    )

        self.pomdp.add_self_loops()

    def __gen_observation(self: Self, state: State, dfa_id: int):
        c = len(self.normal_actions)
        for l in state.labels:
            if l in self.normal_actions.keys():
                c = sorted(self.normal_actions.keys()).index(l)
                break

        return c + dfa_id * (len(self.normal_actions) + 1)

    def __create_product_state(
        self: Self,
        mon_id: int,
        gb_id: int,
        mc_id: int,
    ):
        mc_state = self.mc.states[mc_id]
        mon_state = self.mon.states[mon_id]
        gb_state = self.gb.states[gb_id]
        labels = [f"g{gb_id}", f"l{mon_id}", f"s{mc_id}"] + [
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

        if "accepting" in mon_state.labels:
            state.add_label("accepting")

        if "horizon" in mon_state.labels:
            state.add_label("horizon")

        if "happy" in mon_state.labels:
            state.add_label("happy")

        state.new_observation(mon_id)
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
                        "good" if "good" in mc_new_state.labels else "bad"
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

    def remove_top_states(self: Self):
        unreachable_states = list(self.pomdp.states.values())
        unreachable_states.remove(self.pomdp.get_initial_state())
        for s_id, trans in self.pomdp.transitions.items():
            for branch in trans.transition.values():
                for p, s in branch.branch:
                    if float(p) > 0 and s.id != s_id and s in unreachable_states:
                        unreachable_states.remove(s)

        for s in unreachable_states:
            self.pomdp.states.pop(s.id)
            self.pomdp.transitions.pop(s.id)

    def remove_unreachable_states(self: Self):
        prev_num_states = len(self.pomdp.states) + 1
        while prev_num_states > len(self.pomdp.states):
            self.remove_top_states()
            prev_num_states = len(self.pomdp.states)

    def add_ret_to_bmecs(self: Self):
        storm_m = stormvogel_to_stormpy(self.pomdp)
        if storm_m is None:
            raise Exception("Mapping failed")

        init = self.pomdp.get_initial_state()

        mecs = get_maximal_end_components(storm_m)
        for mec in mecs:
            # Check if MEC is the goal or stop MEC, if this is the case skip it
            skip = False
            for s_id, choice in mec:
                state = self.pomdp.get_state_by_id(s_id)
                if (
                    "goal" in state.labels
                    or "stop" in state.labels
                    # or "init" in state.labels
                ):
                    skip = True
            if skip:
                continue

            # Check if MEC is bottom
            for s_id, choices in mec:
                state = self.pomdp.get_state_by_id(s_id)
                available_actions = set(state.available_actions())
                for choice in choices:
                    labels: set[str] = storm_m.choice_labeling.get_labels_of_choice(
                        choice
                    )
                    matched_actions = {
                        a for a in available_actions if not a.labels.isdisjoint(labels)
                    }
                    available_actions.difference_update(matched_actions)
                if len(available_actions) > 0:
                    break
            else:  # MEC is bottom, we did not break
                mec_ids = [id for id, _ in mec]
                print("found BMEC", mec_ids, "removing it")
                for trans in self.pomdp.transitions.values():
                    for action, branch in trans.transition.items():
                        new_branch: list[tuple[float, State]] = []
                        ret_prob = 0
                        for p, s in branch.branch:
                            if s.id in mec_ids:
                                ret_prob += float(p)
                            else:
                                new_branch.append((float(p), s))
                        if ret_prob > 0:
                            new_branch.append((ret_prob, init))

                        trans.transition[action] = Branch(new_branch)  # type: ignore

    def show(self: Self):
        self.remove_unreachable_states()
        show(self.pomdp)

    def export_to_drn(self: Self, path: str):
        storm_pomdp = stormvogel_to_stormpy(self.pomdp)
        export_to_drn(storm_pomdp, path)

    def check_storm_prop(self: Self, str_prop: str):
        self.pomdp.type = ModelType.MDP
        storm_pomdp = stormvogel_to_stormpy(self.pomdp)
        self.pomdp.type = ModelType.POMDP
        prop = parse_properties(str_prop)
        result = model_checking(storm_pomdp, prop[0], extract_scheduler=True)
        print("Result of model checking", result.at(storm_pomdp.initial_states[0]))
        # induced_mc = storm_pomdp.apply_scheduler(result.scheduler)
        simulator = create_simulator(storm_pomdp)
        scheduler = result.scheduler

        for m in range(3):
            state, reward, labels = simulator.restart()
            old_state = state
            print(f"s{state}, labels={' '.join(labels)} -->")
            for n in range(20):
                chosen_action = storm_pomdp.states[old_state].actions[
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
            print("---------------------------------------------------------------\n")

    def print_mec(self: Self):
        storm_pomdp = stormvogel_to_stormpy(self.pomdp)
        r = get_maximal_end_components(storm_pomdp)
        for mec in r:
            print(mec.size)
            for state, choices in mec:
                print("-", state, storm_pomdp.labeling.get_labels_of_state(state))
                for choice in choices:
                    try:
                        print(
                            "--",
                            storm_pomdp.choice_labeling.get_labels_of_choice(choice),
                            type(
                                storm_pomdp.choice_labeling.get_labels_of_choice(
                                    choice
                                ).pop()
                            ),
                        )
                    except:
                        pass

    def check_paynt_prop(self: Self, str_prop: str):
        storm_pomdp = stormvogel_to_stormpy(self.pomdp)
        paynt.cli.setup_logger()
        logging.getLogger().handlers.clear()

        formula = PrismParser.parse_property(str_prop)
        prop = paynt.verification.property.construct_property(formula, 0)
        specification = paynt.verification.property.Specification([prop])
        explicit_quotient = storm_pomdp

        paynt.verification.property.Property.initialize()
        explicit_quotient = payntbind.synthesis.addMissingChoiceLabels(
            explicit_quotient
        )
        quotient = paynt.quotient.pomdp.PomdpQuotient(explicit_quotient, specification)

        # synthesize 1-FSC
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
            quotient, "ar"
        )
        assignment = (
            synthesizer.synthesize()
        )  # use print_stats=False to remove synthesis summary
        if assignment is not None:
            print(
                "counterexample found: ",
                assignment,
                "\n--------------------------------------\n",
            )

            simulator = create_simulator(storm_pomdp)

            old_observation, reward, labels = simulator.restart()
            print(
                f"s{simulator._report_state()}, labels={' '.join(labels)}",
            )
            paths = [[]]
            while len(paths) < 10000:
                if old_observation >= assignment.num_holes:
                    a_options = [0]
                else:
                    a_options = assignment.hole_options(old_observation)
                observation, reward, labels = simulator.step(a_options[0])

                paths[-1].append(
                    (
                        f"\t--[{old_observation}, {a_options[0]}:{str(assignment.hole_to_option_labels[old_observation][a_options[0]])}]-->\n"
                        f"s{simulator._report_state()}, labels={' '.join(sorted(labels))}",
                        [
                            int(l[5:-1])
                            for l in labels
                            if len(l) > 5 and l.startswith("[pos")
                        ],
                    )
                )
                if simulator._report_state() == 2:
                    print("failed")
                if simulator._report_state() == 0:
                    paths.append([])
                if simulator.is_done():
                    break
                old_observation = observation

            print("\n".join([p for p, _ in paths[-1]]))
            print(f"it took {len(paths)} tries until the goal was reached")
            return [pos[0] for _, pos in paths[-1] if pos]
        else:
            print("counterexample not found")
