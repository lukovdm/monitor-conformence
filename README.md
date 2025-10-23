Artifact for the paper "Learning Verified Monitors for Hidden Markov Models"
============================================================================

This repository contains the implementation of the algorithm called ToVer proposed in the paper:

- [1] "Learning Verified Monitors for Hidden Markov Models" by Luko van der Maas and Sebastian Junges (ATVA 2025)

This artifact requires significant amounts of memory and compute cores to run fully. We provide a reduced experiment set as per the ATVA artifact instructions. We suggest running this reduced set with at least 6 cores and 10 GB of memory, but a single core is supported. When run with 6 cores it should take around 1 hour to run.

The full experiments require at least 100 GB of memory and 40 cores to run in a timely manner (between 1-2 days).

### Contents
This readme gives an overview of the functionality. It also describes how to reproduce the results for [1]. In particular it contains _n_ parts:
- [Smoke testing](#smoke-testing) describes what steps to take to perform the smoke test.
- [Getting ToVer](#getting-tover) describes how to get the docker container.
- [Running ToVer](#running-tover) describes how to use the tool.
- [Running experiments](#running-experiments) describes how to reproduce the experiments.
- [Analyzing the results](#analyzing-the-results) describes how to recreate the figures from the paper.
- [Source code structure](#source-code-structure) describes the structure of this repo.
- [Creating the Docker](#creating-the-docker) describes how to create the docker used in this repository.

## Smoke testing

In order to smoke test this artifact, follow the steps in [Getting ToVer](#getting-tover). Then run the smoke test experiments:
```bash
python -m verimon.experiment tests/smoke-test.yml -c
```

This should generate statistics in the folder `stats/` which can either be inspected using [JupyterLab](http://127.0.0.1:8080/lab/) (Make sure to refresh the jupyter lab file browser if the stats folder is not shown after running the test), or using the command line opened by starting the docker container.

## Getting ToVer

The easiest way of getting a working version of ToVer is by getting the docker container. Details on how to build the docker container manually are attached at the bottom of this readme.

### 1. Loading or pulling the container
First either pull the docker container from dockerhub
```bash
docker pull lukovdm/tover:ATVA
```
or, in case you downloaded the docker container
```bash
docker load -i tover.tar
```
This container is based on the containers for several projects related to the probabilistic model checker storm:
- [Storm](https://www.stormchecker.org/) and [Stormpy](https://moves-rwth.github.io/stormpy/) as provided by the Storm developers.
- PAYNT as provided as provided here [https://github.com/randriu/synthesis/](https://github.com/randriu/synthesis/)
- [Stormvogel](https://moves-rwth.github.io/stormvogel/) as provided by the Storm developers.

### 2. Start the container
The container can be started with the following command 

```bash
docker run --rm -it -p 8080:8080 --name tover lukovdm/tover:ATVA
```

If you want to have a shared folder to copy out results, plots or other files you can instead run

```bash
docker run --volume ./stats:/app/stats --rm -it -p 8080:8080 --name tover lukovdm/tover:ATVA
```

All files included in the repo can found in the app folder in you current directory.

The storm and stormpy source code can be found in `/opt/storm` and `/opt/stormpy` or in the zip file.

## Running ToVer
You can use the tover algorithm on any prism POMDP model or snakes and ladder board configuration by invoking the `verimon/run.py` script. We show two examples below.

For a POMDP:
```bash
python -m verimon.run --file tests/premise/airportA-3.nm --loader pomdp --constants "DMAX=3,PMAX=3" --spec 'Pmax=? [F<=4 "crash"]' --good_label crash --threshold 0.3 --horizon 8 --relative_error 0.01 --use_risk --fp_slack 0.2 --fn_slack 0.05 --use_random_eq --walks_per_state 100 --walk_len 11 --use_horizon_in_filtering --base_dir stats/airport_experiment
```

For SnLs:
```bash
 python -m verimon.run --file tests/premise/airportA-7.nm --loader pomdp --constants "DMAX=3,PMAX=3" --spec 'Pmax=? [F<=4 "crash"]' --good_label crash --threshold 0.3 --horizon 10 --fp_slack 0.2 --fn_slack 0.05 --base_dir stats/airport_experiment
```

The inputs to `python -m verimon.run` are as follows:
- `--file tests/premise/airportA-3.nm` or `--file tests/snake_ladder/mc_u_nxn.pm` for the prism file containing the POMDP or SnLs model.
- `--loader pomdp` or `--loader snakes_ladders` specifying the loader to load the model with. Either POMDPs or SnLs.
- `--constants "DMAX=3,PMAX=3"` specify any constants used to build the prism POMDP model.
- `--n 100 --ladders "1:38,4:14,9:31,28:64,40:42,36:44,51:67,71:91,80:100" --snakes "98:76,95:75,93:73,87:24,64:60,62:19,55:53,49:11,47:26,16:6"` the parameters for the SnLs board. The amount of squares (should be a square number), the locations of the ladders with their destinations and similarly for the snakes.
- `--spec 'Pmax=? [F<=4 "crash"]'` gives the specification with which to generate the risks.
- `--good_label crash` gives the target label in the model.
- `--threshold 0.3` is the learning threshold.
- `--horizon 10` contains the horizon in which the monitor should be correct.
- `--fp_slack 0.2` is the area below the learning threshold considered as undetermined.
- `--fn_slack 0.05` is the area above he learning threshold considered as undetermined.
- `--base_dir stats/airport_experiment` defines where to save the statisics, models and dot file of the model.

## Running experiments
Our experiments are defined in yaml files and consumed by the python file `verimon/experiments.py`. 

The file `tests/reduced-exp/verify.yml` contains the reduced version of the experiments for the section "Efficiency of Monitor Verification", and the folder `tests/verify-exp/` contains the extended version of the experiments found in the verification secion of the paper.

The file `tests/reduced-exp/learn.yml` contains the reduced version of the experiments for the section "Efficiency of Monitor Learning", and the folder `tests/learn-exp/` contains the extended version of the experiments found in the learning section.


For obtaining the reduced results run the following command.
```bash
python -m verimon.experiment tests/reduced-exp/* -c -t 9300
```
This exectues all experiments found the the folder `tests/reduced-exp/` concurrently, each with a timeout of 10 minutes (600 seconds).

To obtain the full results run the below command. Depending on the amount of cores available this can take beween 1-2 days.

```bash
python -m verimon.experiment tests/verify-exp/* tests/learn-exp/* -c
```
The log files of our run of the full experiments can be found in the folders `stats/exp-2025-04-15_15-00-29-comp-base-prem_sam-premise-snl-snl_sam` and `stats/exp-2025-04-24_12-13-12-verify`. The folders can also be used to generate the plots as described in the next section.

All experiment commands take an optional `--core <number of cores>` to specify with how many cores to run the experiments, e.g., `--cores 44` to run with 44 cores. Without this argument it will run on all available cores. Each core can use at most 15GB of memory (many runs don't use the max memory) so it is advisable when running with many cores to have around 10GB of memory per core available. For the reduced set, 2.5GB of memory per core should be enough.

## Analyzing the results

Our results are stored in the folders `stats/exp-2025-04-15_15-00-29-comp-base-prem_sam-premise-snl-snl_sam` and `stats/exp-2025-04-24_12-13-12-verify`. These are the exact results used in the paper.

These results can be analyzed by first opening the `paper.ipynb` [notebook](http://127.0.0.1:8080/lab/tree/notebooks/paper.ipynb). Then, changing the `experiment_dir` to point to the created stats folder, e.g., `"../stats/exp-2025-04-15_15-00-29-learn-verify"`. Make sure that you replace both instances of the assignment of `experiment_dir` to the same path. To easily obtain the path, navigate to the stats folder in browser of jupytor lab and right click to copy to path. Finally, executing all cells. All figures are show in the notebook labeled by which figure they represent. The figures are also exported as `.pgf` files at `VerifiableMonitorsTex/inprogress/images/plots`, which are directly included in the paper. The table are only exported as Tex tables.

Note that when creating the plots for the reduced experiment set, the plots will be very limited and not representative of the full results.

If you forget to open the notebook before running the experiments the password token for the jupyter lab might be hard to find. You can execute the following command in the docker to get it.
```bash
cat /root/jupyter_password.txt
```

## Source code structure
Our source code contains two parts:
- The main model transformation implemented in C++ and located in the Storm model checker source code, together with python bindings located in the Stormpy project.
- The implementation of ToVer using the transformation implemented in python.

### The Transformation
The transformation of a CTR problem HMM into a Colored MDP is implemented in C++ in `storm/src/storm-pomdp/generator/GenerateMonitorVerifier.cpp` with its bindings implemented in `sormpy/src/pomdp/generator.cpp`.

### ToVer algorithm
We will shortly describe the purpose of each of the files contained in python part of the source code found in `ToVer/verimon`. Note that Verimon is an old name for our tool and it is still used in the source code to refer to ToVer.
- `verimon/algs.py` defines several well known algorithms on markov models and automata.
- `verimon/analysis.py` defines functions for creating the plots seen in the paper.
- `verimon/draw.py` draws Snakes and Ladder board states for visualizations.
- `verimon/experiment.py` runs experiments defined by yaml files.
- `verimon/generator.py` calls the C++ code to transform a CTR HMM into a Colored MDP and calls paynt on the Colored MDP.
- `verimon/loaders.py` loads in different kinds of models as stormpy and stormvogel models.
- `verimon/logger.py` defines a logger to be used by all other scripts.
- `verimon/MonitorLearning.py` is the entry point for learning monitors and defines the membership and equivalence queries.
- `verimon/run.py` contains a command line interface for monitor learning.
- `verimon/transformations.py` contains the transformation of a monitor and an HMM into the CTR problem with unrolled horizon. Thus it does both the product and the unrolling over the horizon.
- `verimon/utils.py` contains several utility functions.
- `verimon/verify.py` wrappers around `generator.py` for calculating the false positives and false negatives of a monitor for a model.

## Creating the Docker
The docker file used in this repository can be built with the following command.
```bash
docker build . -t lukovdm/tover:ATVA
```
