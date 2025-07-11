from utils import *
from alpha_fair import *
from config_data import SIMULATION_CONFIG as config


def main():
    # Extraire les paramètres
    T = config["T"]
    n = config["n"]
    eta = config["eta"]
    price = config["price"]
    a = config["a"]
    mu = config["mu"]
    c = config["c"]
    delta = config["delta"]
    epsilon = config["epsilon"]
    Hybrid_funcs = config["Hybrid_funcs"]
    metric = config["metric"]

    #gammas = config["gammas"]
    x_label = config["x_label"]
    y_label = config["y_label"]
    ylog_scale = config["ylog_scale"]
    pltText = config["pltText"]

    lrMethods = config["lrMethods"]

    alpha = config["alpha"]
    gamma = config["gamma"]
    tol = config["tol"]

    saveFileName = config["saveFileName"] + f"alpha{alpha}_gamma{gamma}_n_{n}"

    x_data = np.arange(T)

    a_vector = torch.tensor([a / (i + 1) ** gamma for i in range(n)], dtype=torch.float64)
    #a_vector[-1] = torch.tensor(1e-3)
    c_vector = torch.tensor([c / (i + 1) ** mu for i in range(n)], dtype=torch.float64)

    y_data_speed = []
    y_data_lsw = []

    set1 = torch.arange(n, dtype=torch.long)

    nb_hybrid = len(Hybrid_funcs)

    Hybrid_sets = torch.chunk(set1, nb_hybrid)
    #LSWs_opt = lsw_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0)
    for lrMethod in lrMethods:
        eps = epsilon * torch.ones(1)
        bid0 = torch.ones(n)

        print(f"a_vector :{a_vector}")

        d_vector = torch.zeros(n)

        game_set = GameKelly(n, price, eps, delta, alpha, tol)

        matrix_bids_set, vec_LSW_set, error_NE_set = game_set.learning(lrMethod, a_vector, c_vector, d_vector, T, eta, bid0, vary=False, Hybrid_funcs=Hybrid_funcs,Hybrid_sets=Hybrid_sets)

        y_data_speed.append(error_NE_set.detach().numpy())
        y_data_lsw.append(vec_LSW_set.detach().numpy())

        nb_iter = torch.argmin(error_NE_set) if torch.min(error_NE_set) <= tol else torch.inf

        print(f"{lrMethod} equilibrium:\n {matrix_bids_set[-2]},\n Nbre Iteration: {nb_iter} err: {error_NE_set[-1]}")

    if metric == "speed":
        plotGame(x_data, y_data_speed, x_label, y_label, lrMethods, saveFileName=saveFileName,
                 ylog_scale=ylog_scale, pltText=pltText, step=1)
    if metric == "lsw":
        plotGame(x_data, y_data_lsw, x_label, y_label, lrMethods, saveFileName=saveFileName,
                 ylog_scale=ylog_scale, pltText=pltText, step=1)

    #if metric == "lpoa":


if __name__ == "__main__":
    main()
