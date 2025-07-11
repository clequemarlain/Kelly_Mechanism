import numpy as np
import torch
from scipy import optimize
import matplotlib.pyplot as plt

from scipy.optimize import root_scalar
import sympy as sp
from tensorboard.compat.tensorflow_stub.dtypes import float64

colors = [
    "darkorange",  # Orange foncé
    "royalblue",   # Bleu vif

    "green",       # Vert
    # "red",       # (commenté)
    "purple",      # Violet
    "gold",        # Doré
    "teal",        # Bleu-vert
    "magenta",     # Magenta
    "brown",       # Marron
    "black",       # Noir
    "crimson",     # Rouge profond
    "darkcyan",    # Cyan foncé
    "indigo",      # Bleu indigo
    "salmon",      # Saumon
    "lime",        # Vert clair
    "navy",        # Bleu marine
    "coral",       # Corail
    "darkgreen",   # Vert foncé
    "orchid",      # Orchidée (rose-violet)
    "slategray",   # Gris ardoise
    "darkkhaki"    # Kaki clair
]

markers = ["s", "^", "v", "D", "*", "p", "x", "+", "|", "s", "^", "v", "D", "*", "p", "x", "+", "|","s", "^", "v", "D", "*", "p", "x", "+", "|"]


def solve_nonlinear_eq(a, s, alpha, eps, c_vector, price=1.0, max_iter=100, tol=1e-5):
    """
    Solves for z in: price * (z + s_i)^(2 - alpha) * z^alpha = a_i * s_i
    for each i, using the bisection method.
    """
    a = a.numpy()
    s = s.numpy()
   # c_vector = c_vector.numpy()
    n = len(a)
    z_list = []

    for i in range(n):
        def f(z):
            return price * (z + s[i]) ** (2 - alpha) * z ** alpha - a[i] * s[i]

        # Ensure the bracket is valid
        lower_bound = tol
        upper_bound = c_vector[i] / price

        if f(lower_bound) * f(upper_bound) > 0:
            br = Q1(lower_bound*torch.ones(n), eps, c_vector, price)
            return br
            #raise RuntimeError(f"No root found in the provided bracket for i={i}")

        sol = root_scalar(f, bracket=[lower_bound, upper_bound], method='bisect', xtol=tol)

        if  sol.converged:
            z_list.append(sol.root)
        else:
            z_list.append(lower_bound)
            #raise RuntimeError(f"No root found for i={i}")

        #z_list.append(sol.root)

    br = Q1(torch.tensor(z_list, dtype=torch.float32), eps, c_vector, price)
    return br


def V_func(x, alpha):
    if alpha == 1:
        V = torch.log(x)
    else:
        V = 1 / (1 - alpha) * (x) ** (1 - alpha)
    return V

def Q1(acc_gradient, eps, c, price):
    return torch.minimum(torch.maximum(eps/price, acc_gradient), c/price)


def Q2(acc_gradient, eps, c, price):
    return torch.maximum(eps/price, torch.minimum(torch.exp(acc_gradient - 1), c/price))





def BR_alpha_fair(eps, c_vector, z: torch.Tensor, p,
                  a_vector: torch.Tensor, delta, alpha, price: float, b=0):
    """Compute the best response function for an agent."""
    #p = torch.tensor(p, dtype=torch.float32)  # Ensure p is a tensor
    a_vector = a_vector.to(dtype=torch.float32)

    if alpha == 0:
        br = -p + torch.sqrt(a_vector * p / price)


    elif alpha == 1:
        if b == 0:
            br = (-p + torch.sqrt(p ** 2 + 4 * a_vector * p / price)) / 2
        else:
            #valid = (p > 0) & (p <= a_vector / (b * price))
            discriminant = p ** 2 + 4 * a_vector * p * (1 + b) / price
            br = (-p * (2 * b + 1) + torch.sqrt(discriminant)) / (2 * (1 + b))

    elif alpha == 2:
        br = torch.sqrt(a_vector * p / price)

    return  Q1(br, eps, c_vector, price)


def LSW(x, budgets, a_vector, d_vector, alpha):
    V = V_func(x, alpha)
    lsw = torch.minimum(a_vector * V + d_vector, budgets)
    return torch.sum(lsw)


class GameKelly:
    def __init__(self, n: int, price: float,
                 epsilon, delta, alpha, tol):


        self.n = n


        self.price = price
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.tol = tol

    def fraction_resource(self, z):
        return z / (torch.sum(z) + self.delta)



    def grad_phi(self,phi, bids):
        z = bids.clone().detach()
        #x = self.fraction_resource(z)
        s = torch.sum(z) - z +self.delta

        jacobi = torch.autograd.functional.jacobian(phi, z)#self.a_vector * s / (z + s)**(2 - self.alpha) * z**(self.alpha) - self.price #

        return jacobi.diag()

    def check_NE(self, z: torch.tensor, a_vector, c_vector, d_vector,):
        p = torch.sum(z) - z + self.delta
        if self.alpha  not in [0,1,2]:
            err = torch.maximum(torch.norm(solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon, c_vector, self.price, max_iter=1000, tol=self.tol)
                                           - z), self.tol * torch.ones(1))
        else:
            err =  torch.maximum(torch.norm(BR_alpha_fair(self.epsilon, c_vector, z, p,
                                            a_vector, self.delta, self.alpha, self.price,
                                            b=0) - z), self.tol * torch.ones(1))

        return err   # torch.norm(self.grad_phi(z))
    def Regret(self,  bids,t,a_vector, c_vector, d_vector,):
        def phi( z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        n = bids[0].shape[0]
        p = torch.sum(bids[t-2]) - bids[t-2] + self.delta
        z_t = BR_alpha_fair(self.epsilon, c_vector, bids[t-2], p,
                                         a_vector, self.delta, self.alpha, self.price, b=0)

        Reg = 1/n * torch.sum(torch.abs(phi(bids[t-1]) - phi(z_t)))
        return torch.maximum(Reg,1e-5*torch.ones(1))

    def XL(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi( z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)

        if vary:
            acc_grad_copy += grad_t / (t ** eta)
        else:
            acc_grad_copy += grad_t * eta
        z_t = torch.maximum(self.epsilon / self.price, c_vector / (1 + torch.exp(-acc_grad_copy)))
        return z_t, acc_grad_copy

    def Hybrid(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        acc_grad_copy = acc_grad.clone()

        z_t = torch.zeros_like(bids)
        for idx_set, func in enumerate(Hybrid_funcs):
            func = getattr(self, func)
            z_t[Hybrid_sets[idx_set]], acc_grad_copy[Hybrid_sets[idx_set]] = func(t, a_vector[Hybrid_sets[idx_set]], c_vector[Hybrid_sets[idx_set]],
                                                                    d_vector[Hybrid_sets[idx_set]], eta, bids[Hybrid_sets[idx_set]], acc_grad[Hybrid_sets[idx_set]], vary=vary)
        return z_t, acc_grad_copy

    def AsynXL(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)


        z_t = bids.clone()
        n = bids.shape[0]

        for i in range(n):
            grad_t = self.grad_phi(phi, z_t)
            if vary:
                acc_grad_copy[i] += grad_t[i] / (t ** eta)
            else:
                acc_grad_copy[i] += grad_t[i] * eta

            z_t[i] = torch.maximum(self.epsilon / self.price, c_vector[i] / (1 + torch.exp(-acc_grad_copy[i])))
        return z_t, acc_grad_copy

    def DAQ(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        if vary:
            acc_grad_copy += grad_t / (t ** eta)
        else:
            acc_grad_copy += grad_t * eta
        z_t = Q1(acc_grad_copy, self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy

    def DAH(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        if vary:
            acc_grad_copy += grad_t / (t ** eta)

        else:
            acc_grad_copy += grad_t * eta
        z_t = Q2(acc_grad_copy, self.epsilon, c_vector, self.price)

        return z_t, acc_grad_copy
    def AsynDAQ(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()

        z_t = bids.clone()

        n = bids.shape[0]
        for i in range(n):
            grad_t = self.grad_phi(phi, z_t)
            if vary:
                acc_grad_copy[i] += grad_t[i] / (t ** eta)
            else:
                acc_grad_copy[i] += grad_t[i] * eta

            z_t[i] = Q1(acc_grad_copy[i], self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy

    def SBRD(self, t, a_vector, c_vector, d_vector,  eta, bids, acc_grad, b=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        p = torch.sum(bids) - bids + self.delta
        z_t = BR_alpha_fair(self.epsilon, c_vector, bids, p,
                                         a_vector, self.delta, self.alpha, self.price, b=b)

        return z_t, acc_grad

    def NumSBRD(self,t, a_vector, c_vector, d_vector,  eta, bids, acc_grad, b=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        p = torch.sum(bids) - bids + self.delta
        z_t = solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon,  c_vector, self.price, max_iter=100,tol=self.tol)

        z_t = Q1(z_t, self.epsilon, c_vector, self.price)
        return z_t, acc_grad

    def AsynBRD(self, a_vector, c_vector, d_vector, t,  eta, bids, acc_grad, b=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        n = bids.shape[0]
        z_t = bids.clone()
        for i in range(n):
            p = torch.sum(z_t) - z_t[i] + self.delta


            z_t[i] = BR_alpha_fair(self.epsilon, c_vector[i], z_t[i], p,
                                         a_vector[i], self.delta, self.alpha, self.price, b=b)

            z_t[i] = Q1(z_t[i], self.epsilon, c_vector[i], self.price)
        return z_t, acc_grad

    def learning(self, func, a_vector, c_vector, d_vector, n_iter: int, eta, bids, vary: bool = False, stop=False, Hybrid_funcs=None, Hybrid_sets=None):
        func = getattr(self, func)

        acc_grad = torch.zeros(self.n, dtype=torch.float64)
        matrix_bids = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
        vec_LSW = torch.zeros(n_iter + 1, dtype=torch.float64)
        error_NE = torch.zeros(n_iter + 1, dtype=torch.float64)
        matrix_bids[0] = bids.clone()
        error_NE[0] = self.check_NE(bids,a_vector, c_vector, d_vector)

        k = 0

        for t in range(1, n_iter + 1):

            k = t
            matrix_bids[t], acc_grad = func(t, a_vector, c_vector, d_vector, eta, matrix_bids[t-1], acc_grad, vary=vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets)
            error_NE[t] = self.check_NE(matrix_bids[t], a_vector, c_vector, d_vector,)
            vec_LSW[t] = LSW(self.fraction_resource(matrix_bids[t]), c_vector, a_vector, d_vector, self.alpha)
            err = torch.min(error_NE[:k])#round(float(torch.min(error_NE[:k])),3)
            if stop and err <= self.tol:
                break
        return matrix_bids[:k, :], vec_LSW[:k], error_NE[:k]
        #return matrix_bids, vec_LSW, error_NE



################### End Generalized Bounded Kelly Nash (NBKG) algorithm ###################

def plotGame(x_data, y_data, x_label, y_label, legends, saveFileName, ylog_scale, fontsize=40, markersize=20, linewidth=12,linestyle="-", pltText=False, step=1):

    plt.figure(figsize=(18, 12))
    linewidth = linewidth; markersize = markersize;  y_data = np.array(y_data)

    plt.rcParams.update({'font.size': fontsize})

    x_data_copy = x_data.copy()

    if ylog_scale:
        plt.yscale("log")
    for i in range(len(legends)):  # Évite un dépassement d'index
        color = colors[i]
        if legends[i] == "Equilibrium":
            color = "red"


        if linestyle=="":
            mask = y_data[i] > 0
            #y_data[mask] = y_data[mask]
            print(y_data[i][mask].shape[0])
            x_data = [x_data_copy[i]]*y_data[i][mask].shape[0]
            plt.plot(x_data[::step],
                     (y_data[i][mask])[::step],
                     linestyle=linestyle, linewidth=linewidth, marker=markers[i], markersize=markersize, color=color,
                     label=f"{legends[i]}")

        else:
            plt.plot(x_data[::step],
                     (y_data[i])[::step],
                     linestyle=linestyle, linewidth=linewidth, marker=markers[i], markersize=markersize, color=color,
                     label=f"{legends[i]}")

        if pltText:
            last_x = len(y_data[i]) - 1
            last_y = y_data[i][-1]
            y_offset = 0

            plt.text(last_x, last_y + y_offset, f"{last_y:.2e}",fontweight="bold",
                          fontsize=fontsize, bbox=dict(facecolor='white', alpha=0.7),
                          verticalalignment='bottom', horizontalalignment='right')

        plt.legend(frameon=True, facecolor="white", edgecolor="black", prop={"weight": "bold"})

    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")  # ✅ Graduation des axes en gras

    plt.ylabel(f"{y_label}", fontweight="bold")
    plt.xlabel(f"{x_label}", fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{saveFileName}.pdf', format='pdf')
    plt.show()

