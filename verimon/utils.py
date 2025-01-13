import enum
import itertools
from json import loads
from typing import Any, Generator
from paynt.family.family import Family
from stormpy import SparsePomdp, SparseMdp, SparseModelComponents
import re


def get_pos(json_val):
    return loads(str(json_val))["pos"]


def stormpy_pomdp_to_mdp(pomdp: SparsePomdp) -> SparseMdp:
    components = SparseModelComponents(pomdp.transition_matrix, pomdp.labeling)
    components.choice_labeling = pomdp.choice_labeling
    components.state_valuations = pomdp.state_valuations

    return SparseMdp(components)


def hole_to_observations(assignment: Family) -> dict[int, str]:
    action_map = {}
    for i in range(assignment.num_holes):
        obs = int(assignment.hole_name(i)[2:-3])
        act = assignment.hole_to_option_labels[i][assignment.hole_options(i)[0]]
        action_map[obs] = act
    return action_map


def compact_json_str(json_str: str):
    return (
        json_str.replace("    ", "").replace('"', "").replace("\n", "").replace(" ", "")
    )


# Class to store experiments
class ObjectGroup:
    def __init__(self, prod_class, *args, **kwargs) -> None:
        self.prod_class = prod_class

        self.argss = []
        for arg in args:
            if not isinstance(arg, list):
                self.argss.append([arg])
            else:
                self.argss.append(arg)
        self.arg_prod_indexes = [i for i, arg in enumerate(args) if len(arg) > 1]
        self.kwargss: dict[str, Any] = {}
        for k, v in kwargs.items():
            if not isinstance(v, list):
                self.kwargss[k] = [v]
            else:
                self.kwargss[k] = v
        self.kwargss_prod_keys = [k for k, v in self.kwargss.items() if len(v) > 1]

    def get_objects(self) -> Generator[Any, None, None]:
        for args in itertools.product(*self.argss):
            for kwargs in itertools.product(*self.kwargss.values()):
                kwargs = dict(zip(self.kwargss.keys(), kwargs))
                variant = (
                    "("
                    + ",".join(
                        [
                            self.__value_to_str(arg)
                            for i, arg in enumerate(args)
                            if i in self.arg_prod_indexes
                        ]
                    )
                    + ("," if self.arg_prod_indexes and self.kwargss_prod_keys else "")
                    + ",".join(
                        [
                            str(k) + "=" + self.__value_to_str(v)
                            for k, v in kwargs.items()
                            if k in self.kwargss_prod_keys
                        ]
                    )
                    + ")"
                )
                yield self.prod_class(variant=variant, *args, **kwargs)

    def __value_to_str(self, value):
        if isinstance(value, str) and "/" in value:
            return value.split("/")[-1]
        return str(value)
