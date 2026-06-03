"""CLI entry point for replicating a single experiment from a result JSON file.

Reads the `experiment` block from a previous run's JSON, optionally overrides
any parameter, and reruns the experiment into a fresh `--base-dir`. The new
base directory must not exist (or `--force` must be passed) so the original
artifacts are never overwritten.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Literal, override

from tap import Tap

from tover.core.learning import LearningMethod
from tover.core.synthesis import ConditionalMethod
from tover.experiments.runner import LearningExperiment
from tover.utils.helpers import str_to_float


# Fields from the JSON `experiment` block that are not constructor arguments.
_NON_CONSTRUCTOR_FIELDS = {
    "result_json_file",
    "mc",
    "expr_manager",
    "alphabet",
    "initial_observation",
    "variant_hash",
}


class ReplicateArgs(Tap):
    json_file: str  # Path to the result JSON of the experiment to replicate
    base_dir: str = ""  # Output directory for the rerun (default: out/replicate-<ts>)
    force: bool = False  # Delete --base-dir if it already exists

    # Optional overrides — None means "use the value from JSON"
    file: str | None = None
    spec: str | None = None
    good_label: str | None = None
    loader: Literal["pomdp", "snakes_ladders"] | None = None
    parameters: str | None = None  # JSON string, e.g. '{"constants": "N=4,ENERGY=3"}'

    horizon: int | None = None
    threshold: float | None = None
    fp_slack: float | None = None
    fn_slack: float | None = None
    relative_error: float | None = None

    use_risk: bool | None = None
    use_horizon_in_filtering: bool | None = None
    use_dont_care: bool | None = None
    use_refrence_language: bool | None = None
    use_exact: bool | None = None

    conditional_method: ConditionalMethod | None = None
    learning_method: LearningMethod | None = None
    random_eq_method: str | None = None  # JSON string, or "default"/"none"

    @override
    def process_args(self) -> None:
        if not os.path.isfile(self.json_file):
            self.error(f"JSON file not found: {self.json_file}")


def _coerce(value):
    """Convert serialized fraction strings (e.g. '3/10') back to floats."""
    if isinstance(value, str) and "/" in value:
        try:
            return str_to_float(value)
        except ValueError:
            return value
    return value


def main():
    args = ReplicateArgs().parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    if "experiment" not in data:
        raise ValueError(f"{args.json_file} does not contain an 'experiment' block")

    exp_block = {
        k: v for k, v in data["experiment"].items() if k not in _NON_CONSTRUCTOR_FIELDS
    }

    # The JSON serializes fp_slack/fn_slack separately; the constructor wants a tuple.
    fp_slack = _coerce(exp_block.pop("fp_slack"))
    fn_slack = _coerce(exp_block.pop("fn_slack"))
    for key in ("threshold", "relative_error"):
        if key in exp_block:
            exp_block[key] = _coerce(exp_block[key])

    # Apply CLI overrides.
    overrides = {
        "file": args.file,
        "spec": args.spec,
        "good_label": args.good_label,
        "loader": args.loader,
        "horizon": args.horizon,
        "threshold": args.threshold,
        "relative_error": args.relative_error,
        "use_risk": args.use_risk,
        "use_horizon_in_filtering": args.use_horizon_in_filtering,
        "use_dont_care": args.use_dont_care,
        "use_refrence_language": args.use_refrence_language,
        "use_exact": args.use_exact,
        "conditional_method": args.conditional_method,
        "learning_method": args.learning_method,
    }
    for key, value in overrides.items():
        if value is not None:
            exp_block[key] = value

    if args.parameters is not None:
        exp_block["parameters"] = json.loads(args.parameters)
    if args.fp_slack is not None:
        fp_slack = args.fp_slack
    if args.fn_slack is not None:
        fn_slack = args.fn_slack
    if args.random_eq_method is not None:
        if args.random_eq_method == "none":
            exp_block["random_eq_method"] = None
        elif args.random_eq_method == "default":
            exp_block["random_eq_method"] = "default"
        else:
            exp_block["random_eq_method"] = json.loads(args.random_eq_method)
    elif exp_block.get("random_eq_method") == {}:
        exp_block["random_eq_method"] = "default"

    exp_block["slack"] = (fp_slack, fn_slack)

    # Tag the variant so the rerun is identifiable.
    exp_block["variant"] = f"{exp_block.get('variant', '')} [replicate]"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = args.base_dir or f"out/replicate-{timestamp}"

    if os.path.exists(base_dir):
        if not args.force:
            raise SystemExit(
                f"--base-dir {base_dir} already exists; pass --force to overwrite."
            )
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    experiment = LearningExperiment(**exp_block)
    print(f"Replicating {exp_block['name']} into {base_dir}")
    experiment.run(timestamp, base_dir, output_to_stdout=True)
    print(f"Done. Result: {experiment.result_json_file}")


if __name__ == "__main__":
    main()
