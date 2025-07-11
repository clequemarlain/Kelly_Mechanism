

SIMULATION_CONFIG = {

    "lrMethods": ["SBRD", "XL", "DAQ", "DAH", "Hybrid"],  # Learning methods to test Synchronous Best Response Dynamic (SBRD or NumSBRD for every alpha>=0), Dual Averaging Quadratic Reg (DAQ), Exponential Learning (XL)
    "Hybrid_funcs": ["DAQ", "DAH"], #
    "T": 1000,                 # Number of iterations in the learning process
    "alpha": 1,                # Fairness parameter in utility (e.g., α-fair utility)
    "n": 70,                   # Number of players in the game
    "eta": 0.8,                # Step size for the learning update
    "price": 1.0,              # Price parameter in the game (can represent a resource price)
    "a": 100,                  # Parameter for the utility function heterogeneity (a_i)
    "mu": 0,                   # Exponent controlling the heterogeneity of the c_vector
    "c": 4000,                 # Constant part of the c_vector
    "delta": 0.1,              # Delta parameter (could model uncertainty, slack, or safety margin)
    "epsilon": 1e-3,           # Regularization term (to avoid division by zero, for stability)
    "tol": 1e-5,               # Tolerance threshold for considering the game as converged
    "IdxConfig": 1,            # Configuration index to select the regularizer or the response method
    "x_label": "Time step (t)",  # Label for the x-axis in the output plot
    "metric": "speed",         # "speed" or "lpoa",or "lsw"
    "y_label": "||BR(z) -z||",  # Label for the y-axis in the output plot (error between best response and current state)
    "ylog_scale": False,       # Whether to use a logarithmic scale on the y-axis in the plot, recommended for speed's convergence plot
    "saveFileName": "Error_",  # Prefix for the filename where results/plots are saved
    "pltText": True,           # Whether to display text annotations on the plot
    "gamma": 0                 # Exponent controlling the heterogeneity of the a_vector
}

SIMULATION_CONFIG_table = {
    # Simulation parameters
    "T": 4000,                     # Total number of iterations in the learning process
    "beta": 0.1,                   # Learning rate parameter for the algorithm
    "eta": 0.8,                    # Step size for updating bids
    "price": 1.0,                  # Price parameter in the utility or game setup
    "a": 100,                      # Parameter controlling the heterogeneity of utility functions
    "mu": 0,                       # Exponent applied to c_vector to adjust costs
    "c": 4000,                     # Base value for c_vector (resource costs)
    "delta": 0.1,                  # Regularization parameter in the game formulation
    "epsilon": 1e-3,               # Epsilon parameter to avoid numerical issues (e.g., division by zero)
    "alpha": 0,                    # Fairness parameter for the alpha-fair utility (alpha = 0 means log utility)
    "tol": 1e-5,                   # Tolerance threshold for stopping criteria (convergence)
    "save_path": "results_table.csv",  # Path to save the result (.csv file)

    # Learning methods to compare in the simulation
    "lrMethods": ["SBRD", "DAQ", "XL"],  # List of learning methods: SBRD = Best Response, DAQ = Dual Averaging Quadratic, XL = Extra Learning
    "Hybrid_funcs": ["DAQ", "DAH"],

    # Range of values for experiment parameters
    "list_n": [2, 3, 4, 20],       # List of numbers of players to simulate
    "list_gamma": [0.0, 0.5, 1.0]  # List of heterogeneity exponents for a_vector
}
