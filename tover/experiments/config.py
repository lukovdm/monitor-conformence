import itertools
from typing import Any, Generator


class ObjectGroup:
    """Generates Cartesian product of experiment parameter variants.

    Positional and keyword arguments can be lists (swept over) or single values.
    Yields one experiment object per combination, with a readable variant string.
    """

    def __init__(self, prod_class, *args, **kwargs) -> None:
        self.prod_class = prod_class

        self.argss = []
        for arg in args:
            self.argss.append(arg if isinstance(arg, list) else [arg])
        self.arg_prod_indexes = [i for i, arg in enumerate(args) if len(arg) > 1]

        self.kwargss: dict[str, Any] = {}
        for k, v in kwargs.items():
            self.kwargss[k] = v if isinstance(v, list) else [v]
        self.kwargss_prod_keys = [k for k, v in self.kwargss.items() if len(v) > 1]

    def get_objects(self) -> Generator[Any, None, None]:
        for args in itertools.product(*self.argss):
            for kwargs in itertools.product(*self.kwargss.values()):
                kwargs = dict(zip(self.kwargss.keys(), kwargs))
                variant = (
                    "("
                    + ",".join(
                        self._value_to_str(arg)
                        for i, arg in enumerate(args)
                        if i in self.arg_prod_indexes
                    )
                    + ("," if self.arg_prod_indexes and self.kwargss_prod_keys else "")
                    + ",".join(
                        f"{k}={self._value_to_str(v)}"
                        for k, v in kwargs.items()
                        if k in self.kwargss_prod_keys
                    )
                    + ")"
                )
                try:
                    yield self.prod_class(variant=variant, *args, **kwargs)
                except ValueError as e:
                    continue  # Skip invalid parameter combinations

    def _value_to_str(self, value) -> str:
        if isinstance(value, str) and "/" in value:
            return value.split("/")[-1]
        return str(value)
