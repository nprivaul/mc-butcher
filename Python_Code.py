# Author: Jiang Yu Nguwi
import time
import math
import torch
from scipy import special
from torch.distributions.exponential import Exponential
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)  # set seed for reproducibility


class ODEBranch(torch.nn.Module):
    def __init__(
        self,
        fun,
        t_lo=0.0,
        t_hi=1.0,
        y0=1.0,
        nb_path_per_state=10000,
        nb_states=6,
        outlier_percentile=1,
        outlier_multiplier=1000,
        patch=1,
        epochs=3000,
        device="cpu",
        verbose=False,
        **kwargs,
    ):
        super(ODEBranch, self).__init__()
        self.fun = fun

        self.loss = torch.nn.MSELoss()
        self.nb_path_per_state = nb_path_per_state
        self.nb_states = nb_states
        self.outlier_percentile = outlier_percentile
        self.outlier_multiplier = outlier_multiplier
        self.patch = patch
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.y0 = y0
        self.dim = len(y0)
        self.epochs = epochs
        self.device = device
        self.verbose = verbose

    def forward(self, code=None):
        start = time.time()
        code = [-1] * self.dim if code is None else code  # start from identity code if not specified
        t = torch.linspace(self.t_lo, self.t_hi, steps=self.nb_states, device=self.device)
        t = t.repeat(self.nb_path_per_state).reshape(self.nb_path_per_state, -1).T
        nb_states_per_patch = math.ceil(self.nb_states / self.patch)
        cur_start_idx, cur_end_idx = 0, nb_states_per_patch
        mc_mean, mc_var = [], []
        y0, t0 = torch.tensor(self.y0, device=self.device), torch.tensor(self.t_lo, device=self.device)
        while cur_start_idx < cur_end_idx:
            self.code_to_fun_dict = {}
            t_this_patch = t[cur_start_idx:cur_end_idx]
            H_tensor = torch.ones_like(t_this_patch)
            mask_tensor = torch.ones_like(t_this_patch)
            mc_mean_this_patch = []
            mc_var_this_patch = []
            for i in range(self.dim):
                y = self.gen_sample_batch(
                    t_this_patch,
                    t0,
                    y0,
                    np.array(code),
                    H_tensor,
                    mask_tensor,
                    coordinate=i
                )
                # widen (outlier_percentile, 1 - outlier_percentile) by outlier_multiplier times
                # everything outside this range is considered outlier
                lo = y.nanquantile(self.outlier_percentile/100, dim=1, keepdim=True)
                hi = y.nanquantile(1 - self.outlier_percentile/100, dim=1, keepdim=True)
                lo, hi = lo - self.outlier_multiplier * (hi - lo), hi + self.outlier_multiplier * (hi - lo)
                mask = torch.logical_and(lo <= y, y <= hi)
                mc_mean_this_patch.append((y * mask).sum(dim=1) / mask.sum(dim=1))
                y = y - mc_mean_this_patch[-1].unsqueeze(dim=-1)
                mc_var_this_patch.append(torch.square(y * mask).sum(dim=1) / mask.sum(dim=1))

            # update y0, t0, idx
            mc_mean.append(torch.stack(mc_mean_this_patch))
            mc_var.append(torch.stack(mc_var_this_patch))
            y0, t0 = mc_mean[-1][:, -1], t_this_patch[-1][-1]
            cur_start_idx, cur_end_idx = cur_end_idx, min(cur_end_idx + nb_states_per_patch, self.nb_states)

        if self.verbose:
            print(f"Time taken for the simulations: {time.time() - start:.2f} seconds.")
        return t[:, 0], torch.cat(mc_mean, dim=-1), torch.cat(mc_var, dim=-1)

    @staticmethod
    def nth_derivatives(order, y, x):
        """
        calculate the derivatives of y wrt x with order `order`
        """
        for cur_dim, cur_order in enumerate(order):
            for _ in range(int(cur_order)):
                try:
                    grads = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
                except RuntimeError as e:
                    # when very high order derivatives are taken for polynomial function
                    # it has 0 gradient but torch has difficulty knowing that
                    # hence we handle such error separately
                    # logging.debug(e)
                    return torch.zeros_like(y)

                # update y
                y = grads[cur_dim]
        return y

    def code_to_function(self, code, t, y0, coordinate):
        code = tuple(code)
        if (code, coordinate) not in self.code_to_fun_dict.keys():
            # code (-1, -1, ..., -1) -> identity mapping
            if code == (-1,) * self.dim:
                self.code_to_fun_dict[(code, coordinate)] = y0[coordinate]
            else:
                y = y0.clone().requires_grad_(True)
                self.code_to_fun_dict[(code, coordinate)] = (
                    self.nth_derivatives(code, self.fun(y, coordinate), y).detach()
                )
        return self.code_to_fun_dict[(code, coordinate)]

    def gen_sample_batch(self, t, t0, y0, code, H, mask, coordinate):
        nb_states, _ = t.shape
        tau = Exponential(
            torch.ones(nb_states, self.nb_path_per_state, device=self.device)
        ).sample()
        ans = torch.zeros_like(t)

        ############################### for t + tau >= T
        mask_now = mask.bool() * (t0 + tau >= t)
        if mask_now.any():
            ans[mask_now] = (
                    H[mask_now]
                    * self.code_to_function(code, t0, y0, coordinate)
                    / torch.exp(-(t - t0)[mask_now])
            )

        ############################### for t + tau < T
        mask_now = mask.bool() * (t0 + tau < t)
        if (code == [-1] * self.dim).all():
            if mask_now.any():
                # code (-1, -1,..., -1) -> (0, 0,..., 0)
                tmp = self.gen_sample_batch(
                    t - tau, t0, y0, code + 1, H / torch.exp(-tau), mask_now, coordinate,
                )
                ans = ans.where(~mask_now, tmp)

        else:
            unif = torch.rand(nb_states, self.nb_path_per_state, device=self.device)
            idx = (unif * self.dim).long()
            for i in range(self.dim):
                mask_tmp = mask_now * (idx == i)
                if mask_tmp.any():
                    A = self.gen_sample_batch(
                        t - tau,
                        t0,
                        y0,
                        np.array([0] * self.dim),
                        torch.ones_like(t),
                        mask_tmp,
                        i,
                    )
                    code[i] += 1
                    tmp = self.gen_sample_batch(
                        t - tau,
                        t0,
                        y0,
                        code,
                        self.dim * A * H / torch.exp(-tau),
                        mask_tmp,
                        coordinate,
                    )
                    code[i] -= 1
                    ans = ans.where(~mask_tmp, tmp)
        return ans


if __name__ == "__main__":
    # problem configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = [
        "quadratic",
        "cosine",
        "example_3",
        "example_5",
        "example_6"
    ][4]
    dim = 1
    if problem == "quadratic":
        exact_fun = (lambda t, y, coordinate: y[coordinate] / (1 - y[coordinate] * t))
        f_fun = (lambda y, coordinate: y[coordinate] ** 2)
        t_lo, t_hi = 0, 0.5
        y0 = [1.0] * dim
        nb_states = 6
    elif problem == "cosine":
        def exact_fun(t, y, coordinate):
            return 2 * torch.atan(torch.tanh((t + 2 * math.atanh(math.tan(y[coordinate] / 2))) / 2))
        f_fun = (lambda y, coordinate: torch.cos(y[coordinate]))
        t_lo, t_hi = 0, 1.0
        y0 = [1.0] * dim
        nb_states = 6
    elif problem == "example_3":
        def exact_fun(t, y, coordinate):
            if coordinate == 0:
                return y[coordinate] + t
            else:
                return t + torch.sqrt(y[coordinate] + 2 * t ** 2)

        def f_fun(y, coordinate):
            if coordinate == 0:
                return torch.ones_like(y[0])
            else:
                return (y[coordinate] + y[0]) / (y[coordinate] - y[0])
        t_lo, t_hi = 0, 0.5
        y0 = [t_lo] + [1.0] * dim
        nb_states = 11
    elif problem == "example_5":
        def exact_fun(t, y, coordinate):
            if coordinate == 0:
                return y[coordinate] + t
            else:
                tensor_erfi = (lambda x: special.erfi(x.cpu()).to(device))
                return torch.exp(t**2/2) / (1/y[coordinate] - (math.pi / 2) ** 0.5 * tensor_erfi(t / 2 ** 0.5))

        def f_fun(y, coordinate):
            if coordinate == 0:
                return torch.ones_like(y[0])
            else:
                return y[0] * y[coordinate] + y[coordinate]**2
        t_lo, t_hi = 0, 1.0
        y0 = [t_lo] + [.5] * dim
        nb_states = 11
    elif problem == "example_6":
        def exact_fun(t, y, coordinate):
            if coordinate == 0:
                return t*torch.sin(torch.log(t))
            else:
                return t*torch.cos(torch.log(t))

        def f_fun(y, coordinate):
            if coordinate == 0:
                return ( y[1] + y[0] ) / torch.sqrt(y[0]**2+y[1]**2)
            else:
                return ( y[1] - y[0] ) / torch.sqrt(y[0]**2+y[1]**2)
        t_lo, t_hi = 1, 2
        y0 = [0.0] + [1.0]
        nb_states = 5

    # initialize model and calculate mc samples
    model = ODEBranch(
        f_fun,
        t_lo=t_lo,
        t_hi=t_hi,
        y0=y0,
        device=device,
        nb_states=nb_states,
        verbose=True,
        patch=2,
        outlier_percentile=0.1,
        outlier_multiplier=100,
    )
    t, mc_mean, mc_var = model()
    t_fine = torch.linspace(t_lo, t_hi, 100, device=device)  # finer grid for plotting exact solution
    torch.set_printoptions(precision=2, sci_mode=True)

    # plot exact vs numerical
    for i in range(model.dim):
        print(f"For dimension {i + 1}:")
        print(f"The variance of MC is {mc_var[i]}.")
        print(f"The error squared is {(mc_mean[i] - exact_fun(t, y0, i)) ** 2}.")
        plt.plot(t.cpu(), mc_mean[i].cpu(), '+', label="Numerical solution")
        plt.plot(t_fine.cpu(), exact_fun(t_fine, y0, i).cpu(), label="Exact solution")
        plt.title(f"Dimension {i + 1}")
        plt.legend()
        plt.show()
