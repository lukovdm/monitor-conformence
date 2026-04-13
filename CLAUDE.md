# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToVer ("Verimon" in older code) is a research artifact for the paper "Learning Verified Monitors for Hidden Markov Models" (ATVA 2025). It learns DFA monitors for probabilistic systems (HMMs/POMDPs) and verifies them using PAYNT synthesis.

Dependencies are managed with **uv**. The local `../stormpy` is used as an editable source dependency. The project requires stormpy, stormvogel, paynt, and aalpy — all of which depend on a working Storm model checker installation, typically available only in the Docker container.

## Commands

```bash
# Install dependencies
uv sync

# Run a single monitor learning job (POMDP)
python -m tover.cli.run --file tests/premise/airportA-3.nm --loader pomdp \
  --constants "DMAX=3,PMAX=3" --spec 'Pmax=? [F<=4 "crash"]' \
  --good-label crash --threshold 0.3 --horizon 8 \
  --fp-slack 0.2 --fn-slack 0.05 --exact --base-dir out/my-run

# Run batch experiments from YAML configs
python -m tover.cli.experiment tests/reduced-exp/*.yml --concurrent --timeout 9300

# List experiments in a YAML file without running
python -m tover.cli.experiment tests/reduced-exp/*.yml --list

# Lint
ruff check tover/
ruff format tover/
```

Note: `--exact` or `--double` must be specified for `tover.cli.run`. Experiments run with `--concurrent` use all available cores by default; use `--cores N` to limit.

Output lands in `out/` (single runs) or `stats/` (experiments).

## Architecture

The learning loop is in `tover/core/`:

```
FilteringSUL (sul.py)
  - Wraps a SparseDtmc (MC with observations as state labels)
  - Uses a nondeterministic belief tracker (stormpy.pomdp) to compute risk per observation
  - step() returns True/False/"unknown" (don't-care) based on risk vs. threshold

ToVerEqOracle (oracles.py)
  - Equivalence oracle for AALpy's L* algorithm
  - Optionally first tries SamplingEqOracle (fast random walks through MC)
  - Main path: converts hypothesis DFA → Stormpy MDP, calls false_negative() then false_positive()

Verifier (synthesis.py)
  - Builds the product of MC × unrolled-monitor using the C++ GenerateMonitorVerifier (stormpy.pomdp)
  - Calls PAYNT (ar strategy) to find a scheduler maximising goal/stop probability
  - trace_of_assignment() extracts a counterexample trace by simulating the monitor MDP

verification.py
  - false_positive / false_negative / true_positive / true_negative
  - Each calls stormpy_unroll() then constructs a Verifier and runs PAYNT
  - Double-checks the found value via direct model checking on the induced DTMC

transformations.py
  - stormpy_unroll(): BFS-based finite-horizon unrolling of a cyclic MDP into an acyclic MDP
    (horizon states loop back on themselves; adds "step=i" labels)
  - stormvogel_unroll / stormvogel_product_unroll: stormvogel-based variants
```

**Model loading** (`tover/models/`):
- `pomdp.py`: Parses PRISM POMDP → SparseDtmc. POMDP actions are averaged to produce an MC; observation valuations become state labels (JSON-compacted strings) forming the alphabet.
- `automata.py`: Converts AALpy DFA ↔ Stormpy SparseMdp (the monitor representation). Monitors are MDPs where states are DFA states, actions are alphabet symbols, and transitions are deterministic (probability 1.0).
- `snakes.py`: Loads Snakes and Ladders board models.

**Three learning modes** (`tover/core/learning.py`):
1. `run_tover`: Full algorithm — L* + `ToVerEqOracle` (PAYNT synthesis)
2. `run_trad_learning`: L* + `RandomWMethodEqOracle` (no PAYNT)
3. `run_sampling_learning`: L* + `SamplingEqOracle` (MC random walks)

**Experiments** (`tover/experiments/`):
- YAML files define `LearningExperiment` or `VerifyExperiment` objects with parameter grids
- `scheduler.py` runs them concurrently using `multiprocessing`, with per-experiment timeouts and a 15 GiB memory limit
- Results are written as JSON to `{base_dir}/json/`

**Analysis** (`tover/analysis/`): Loads experiment JSON results and generates matplotlib plots and LaTeX tables for the paper.

## Key Concepts

- **Horizon**: The monitor is only required to be correct for traces up to this length. The MDP is unrolled to this depth before building the product.
- **fp_slack / fn_slack**: Dead-zone around the threshold — traces within `[threshold - fp_slack, threshold + fn_slack]` are treated as "don't care" by the SUL.
- **Risk**: The probability of satisfying the spec, computed by model checking the MC. `use_risk=True` stores this as a continuous value; `use_risk=False` uses a Boolean labelling.
- **Exact vs float**: Stormpy supports exact rational arithmetic (`SparseExactDtmc`, `SparseExactMdp`) or double-precision floats. PAYNT's AR strategy is required for exact models.
