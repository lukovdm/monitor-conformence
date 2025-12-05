def generate_verify_table(data, save_figures=False, save_path="./", file_name="verify"):
    preamble = r"""% Auto generated table
\begin{longtable}[c]{@{}llrrrrrrrrrrrrrr@{}}
\caption{Table of all verification experiments.}
\label{tab:fullverifyres}\\                                                                                                                                                                                                                                                                                                        \\
\toprule
 & & \multicolumn{8}{c}{Benchmark} & \multicolumn{5}{c}{\alg}                                                                                                                                                       \\
\cmidrule(lr){3-11}\cmidrule(lr){12-16}
 & & $\lambda_l$ & $h$ & MA/FA & $|\Sts^\mc|$ & $|\ptrans^\mc|$ & $|Z|$ & $|\Sts^\dfa|$ & $|\ptrans^\dfa|$ & $|\lang{\mc}^{\leq h}|$ & Time (s) & Trans (s) & PAYNT (s) & $|\mdp_{\gtrdot h}|$ & $\lambda^{found}$  \\
\midrule
\endhead"""

    name_map = {
        "airport": r"\textsc{Airport}",
        "evade": r"\textsc{Evade}",
        "refuel": r"\textsc{Refuel}",
        "icy-driving": r"\textsc{Icy-Driving}",
        "hidden_incentive": r"\textsc{Hidden-Incen.}",
        "snakes_ladders": r"\textsc{SnL}",
    }
    file_map = {
        "tests/premise/airportA-3.nm": r"\textsc{AirportA-3}",
        "tests/premise/airportA-7.nm": r"\textsc{AirportA-7}",
        "tests/premise/airportB-3.nm": r"\textsc{AirportB-3}",
        "tests/premise/airportB-7.nm": r"\textsc{AirportB-7}",
        "tests/premise/refuel.nm": r"\textsc{Refuel}",
        "tests/premise/refuelB.nm": r"\textsc{RefuelB}",
    }

    tab_data = [
        [
            (
                file_map[d["experiment"]["mc"]]
                if d["experiment"]["mc"] in file_map
                else name_map[d["experiment"]["learn_experiment"]["name"]]
            ),
            d["experiment"]["short_name"],
            (
                d["experiment"]["threshold"]
                if d["experiment"]["threshold"] is not None
                else r"\checkmark"
            ),
            d["experiment"]["horizon"],
            "MA" if d["experiment"]["search"] == "fn" else "FA",
            d["mc"]["mc_states"],
            d["mc"]["mc_transitions"],
            d["mc"]["mc_observations"],
            d["monitor"]["monitor_states"],
            d["monitor"]["monitor_transitions"],
            (
                f"$10^{{{int(math.log10(d['family_size']))}}}$"
                if d["family_size"] is not None
                else r"-"
            ),
            (
                (
                    round(d["result"]["time"])
                    if d["result"]["time"] >= 1
                    else r"$\leq 1s$"
                )
                if "fake" not in d["result"]
                else r"-"
            ),
            (
                (
                    round(d["result"]["product_time"])
                    if d["result"]["product_time"] >= 1
                    else r"$\leq 1s$"
                )
                if "fake" not in d["result"]
                else r"-"
            ),
            (
                (
                    round(d["result"]["paynt_time"])
                    if d["result"]["paynt_time"] >= 1
                    else r"$\leq 1s$"
                )
                if "fake" not in d["result"]
                else r"-"
            ),
            (
                d["result"]["pomdp_states"]
                if "fake" not in d["result"] and d["result"]["pomdp_states"] is not None
                else r"-"
            ),
            (
                float(d["result"]["goal_threshold"])
                if d["result"]["goal_threshold"] is not None
                and "fake" not in d["result"]
                else (r"\checkmark" if "fake" not in d["result"] else r"-")
            ),
        ]
        for d in data
    ]
    tab_with_lines: list[Any] = [tab_data[0]]
    for line in tab_data[1:]:
        if line[0] != tab_with_lines[-1][0]:
            tab_with_lines.append("SEPERATING LINE")
        tab_with_lines.append(line)

    generate_table(preamble, tab_with_lines, save_path, file_name)


def generate_learn_table(data, save_figures=False, save_path="./", file_name="runtime"):
    preamble = r"""% Auto generated table
\begin{longtable}[c]{@{}llrrrrrrrrrrrrrrrr@{}}
\caption{Table of all learn experiments.}
\label{tab:fulllearnexp}\\                                                                                                                                                                                                                                                                                                    \\
\toprule
 & & \multicolumn{6}{c}{Benchmark} & \multicolumn{5}{c}{\alg} & \multicolumn{5}{c}{Baseline}                                                                                                                                                       \\
\cmidrule(lr){3-8}\cmidrule(lr){9-13}\cmidrule(lr){14-18}
 & & $\lambda_u$ & $\lambda_s$ & $h$ & $|\Sts|$ & $|\ptrans|$ & $|Z|$ & Time (s) & Rounds & $|\dfa|$ & $\lambda_u^{\min}$ & $\lambda_s^{\max}$ & Time (s) & $|\dfa|$ & Rounds & $\lambda_u^{\min}$ & $\lambda_s^{\max}$ \\
\midrule
\endhead"""

    name_map = {
        "airport": r"\textsc{Airport}",
        "evade": r"\textsc{Evade}",
        "refuel": r"\textsc{Refuel}",
        "icy-driving": r"\textsc{Icy-Driving}",
        "hidden_incentive": r"\textsc{Hidden-Incen.}",
        "snakes_ladders": r"\textsc{SnL}",
    }
    fake_map = {
        "timeout": r"$\infty$",
        "out of memory": r"OM",
    }
    tab_data = [
        [
            name_map[d["experiment"]["name"]],
            d["experiment"]["short_name"],
            d["experiment"]["threshold"] - d["experiment"]["fp_slack"],
            d["experiment"]["threshold"] + d["experiment"]["fn_slack"],
            d["experiment"]["horizon"],
            d["mc"]["mc_states"],
            d["mc"]["mc_transitions"],
            d["mc"]["mc_observations"],
            (
                (
                    round(d["verimon"]["time"])
                    if d["verimon"]["time"] >= 1
                    else r"$\leq 1s$"
                )
                if "fake" not in d["verimon"]
                else fake_map[d["verimon"]["time"]]
            ),
            len(d["verimon"]["monitors"]) if "fake" not in d["verimon"] else r"-",
            d["verimon"]["monitor_states"] if "fake" not in d["verimon"] else r"-",
            (
                (
                    float(d["verimon"]["false_positive"]),
                    d["verimon"]["false_positive"]
                    < d["experiment"]["threshold"] - d["experiment"]["fp_slack"],
                )
                if "fake" not in d["verimon"]
                else r"-"
            ),
            (
                (
                    float(d["verimon"]["false_negative"]),
                    d["verimon"]["false_negative"]
                    > d["experiment"]["threshold"] + d["experiment"]["fn_slack"],
                )
                if "fake" not in d["verimon"]
                else r"-"
            ),
            (
                (
                    round(d["sampling"]["time"])
                    if d["sampling"]["time"] >= 1
                    else r"$\leq 1s$"
                )
                if "fake" not in d["sampling"]
                else fake_map[d["sampling"]["time"]]
            ),
            d["sampling"]["monitor_states"] if "fake" not in d["sampling"] else r"-",
            d["sampling"]["learning_rounds"] if "fake" not in d["sampling"] else r"-",
            (
                (
                    float(d["sampling"]["false_positive"]),
                    d["sampling"]["false_positive"] < d["experiment"]["threshold"],
                )
                if "fake" not in d["sampling"]
                and d["sampling"]["false_positive"] is not None
                else r"-"
            ),
            (
                (
                    float(d["sampling"]["false_negative"]),
                    d["sampling"]["false_negative"] > d["experiment"]["threshold"],
                )
                if "fake" not in d["sampling"]
                and d["sampling"]["false_negative"] is not None
                else r"-"
            ),
        ]
        for d in data
        if "sampling" in d
    ]
    tab_with_lines: list[Any] = [tab_data[0]]
    for line in tab_data[1:]:
        if line[0] != tab_with_lines[-1][0]:
            tab_with_lines.append("SEPERATING LINE")
        tab_with_lines.append(line)

    generate_table(preamble, tab_with_lines, save_path, file_name)


def generate_table(preamble, data, save_path="./", file_name="runtime"):
    with open(f"{save_path}/{file_name}.tex", "w") as f:
        f.write(preamble)
        for line in data:
            if line == "SEPERATING LINE":
                f.write(r"\midrule" + "\n")
            else:
                str_line = []
                for e in line:
                    prefix = ""
                    postfix = ""
                    if isinstance(e, tuple):
                        if e[1]:
                            prefix = r"{\color{red} "
                            postfix = r"}"
                        e = e[0]

                    if isinstance(e, float):
                        str_line.append(f"{prefix}{e:.2f}{postfix}")
                    elif isinstance(e, Fraction):
                        str_line.append(
                            f"{prefix}\\sfrac{{{e.numerator}}}{{{e.denominator}}}{postfix}"
                        )
                    else:
                        str_line.append(str(e))
                f.write(" & ".join(e for e in str_line) + r"\\" + "\n")
        f.write(
            r"""
\bottomrule
\end{longtable}
            """
        )
