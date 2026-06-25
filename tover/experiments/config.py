import itertools
from typing import Any, Generator


class ObjectGroup:
    """Generates Cartesian product of experiment parameter variants.

    Positional and keyword arguments can be lists (swept over) or single values.
    Yields one experiment object per combination, with a readable variant string.

    The reserved ``zip`` keyword couples several parameters so they vary *together*
    instead of forming a Cartesian product across the whole group. Its value is a
    list of bundles (dicts); within a bundle, list-valued keys are crossed with one
    another while scalar keys stay fixed, and the bundles are concatenated to form a
    single ``zip`` axis. For example, to pin an arithmetic precision per method::

        zip:
          - {conditional_method: [rejection, restart], use_exact: true}
          - {conditional_method: [bisection, bisection_pt], use_exact: false}

    yields four variants (two exact, two float) rather than the eight a normal cross
    would produce. Bundle values override any same-named keyword; the ``zip`` axis
    still crosses normally with the other (non-coupled) swept parameters.
    """

    def __init__(self, prod_class, *args, **kwargs) -> None:
        self.prod_class = prod_class

        self.argss = []
        for arg in args:
            self.argss.append(arg if isinstance(arg, list) else [arg])
        self.arg_prod_indexes = [i for i, arg in enumerate(args) if len(arg) > 1]

        # Expand each `zip` bundle into concrete assignments (crossing its own
        # list-valued keys), then concatenate them into a single coupled axis.
        self.zip_variants: list[dict[str, Any]] = []
        for bundle in kwargs.pop("zip", []):
            value_lists = [v if isinstance(v, list) else [v] for v in bundle.values()]
            for combo in itertools.product(*value_lists):
                self.zip_variants.append(dict(zip(bundle.keys(), combo)))

        self.kwargss: dict[str, Any] = {}
        for k, v in kwargs.items():
            self.kwargss[k] = v if isinstance(v, list) else [v]
        self.kwargss_prod_keys = [k for k, v in self.kwargss.items() if len(v) > 1]

    def get_objects(self) -> Generator[Any, None, None]:
        bundles = self.zip_variants or [{}]
        for args in itertools.product(*self.argss):
            for values in itertools.product(*self.kwargss.values()):
                kwargs = dict(zip(self.kwargss.keys(), values))
                for bundle in bundles:
                    variant = self._make_variant(args, kwargs, bundle)
                    try:
                        yield self.prod_class(
                            variant=variant, *args, **{**kwargs, **bundle}
                        )
                    except ValueError:
                        continue  # Skip invalid parameter combinations

    def _make_variant(self, args, kwargs, bundle) -> str:
        parts = [
            self._value_to_str(arg)
            for i, arg in enumerate(args)
            if i in self.arg_prod_indexes
        ]
        parts += [
            f"{k}={self._value_to_str(v)}"
            for k, v in kwargs.items()
            if k in self.kwargss_prod_keys
        ]
        if len(self.zip_variants) > 1:
            parts += [f"{k}={self._value_to_str(v)}" for k, v in bundle.items()]
        return "(" + ",".join(parts) + ")"

    def _value_to_str(self, value) -> str:
        if isinstance(value, str) and "/" in value:
            return value.split("/")[-1]
        return str(value)
