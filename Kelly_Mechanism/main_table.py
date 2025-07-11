import torch, csv
import numpy as np
from utils import *
from alpha_fair import *
from config_data import SIMULATION_CONFIG_table as config
from tabulate import tabulate
from collections import defaultdict

def run_simulation():
    # Extract config variables
    T = config["T"]
    beta = config["beta"]
    eta = config["eta"]
    price = config["price"]
    a = config["a"]
    mu = config["mu"]
    c = config["c"]
    delta = config["delta"]
    epsilon = config["epsilon"]
    alpha = config["alpha"]
    tol = config["tol"]

    lrMethods = config["lrMethods"]
    Hybrid_funcs = config["Hybrid_funcs"]
    list_n = config["list_n"]
    list_gamma = config["list_gamma"]
    total_runs = len(list_gamma) * len(list_n) * len(lrMethods)
    run_counter = 1


    # Collect results
    results = []
    print("")
    print(f"############## Comparison of Convergence Time: α = {alpha} ########################")
    print("")
    for gamma in list_gamma:
        for n in list_n:
            set1 = torch.arange(n, dtype=torch.long)
            nb_hybrid = len(Hybrid_funcs)

            Hybrid_sets = torch.chunk(set1, nb_hybrid)
            a_vector = torch.tensor([a / (i + 1) ** gamma for i in range(n)], dtype=torch.float64)
            c_vector = torch.tensor([c / (i + 1) ** mu for i in range(n)], dtype=torch.float64)

            for lrMethod in lrMethods:
                print(f"[{run_counter}/{total_runs}] Running simulation: gamma={gamma}, n={n}, method={lrMethod}")
                run_counter += 1
                eps = epsilon * torch.ones(n)
                bid0 = torch.ones(n)
                d_vector = torch.zeros(n)


                game = GameKelly(n, price, eps, delta, alpha, tol)
                bids, lsw, error = game.learning(lrMethod, a_vector, c_vector, d_vector,
                                                                               T, eta, bid0, vary=False,
                                                                               Hybrid_funcs=Hybrid_funcs,
                                                                               Hybrid_sets=Hybrid_sets)

                min_error = torch.min(error)
                nb_iter = int(torch.argmin(error).item()) if min_error <= tol else float('inf')

                results.append({
                    "gamma": gamma,
                    "n": n,
                    "method": lrMethod,
                    "iterations": nb_iter
                })

    return results


def display_results_plain(results, save_path=config["save_path"]):
    lrMethods = config["lrMethods"]
    # Organize results
    table_data = defaultdict(lambda: defaultdict(dict))
    for row in results:
        gamma = row["gamma"]
        n = row["n"]
        method = row["method"]
        iters = row["iterations"]
        table_data[gamma][n][method] = iters

    # Build rows
    rows = []
    headers = ["gamma", "n"] + lrMethods

    # Replace inf with ∞ symbol
    def fmt(val):
        return "∞" if val == float("inf") else str(val)

    for gamma in sorted(table_data.keys()):
        for n in sorted(table_data[gamma].keys()):
            row = [gamma, n]
            for lrMethod in lrMethods:
                time = table_data[gamma][n].get(lrMethod, "---")
                row.append(fmt(time))

            rows.append(row)

    # Print the table
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"############## ############################# ########################")

    # Save table to file if a path is provided
    if save_path:
        with open(save_path, mode="w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
        print(f"\n✅ Table saved to {save_path}")



if __name__ == "__main__":
    results = run_simulation()
    display_results_plain(results)
