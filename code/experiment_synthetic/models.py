# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import math
import logging
import random
import copy

from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

from tqdm import tqdm

from torch.autograd import grad

import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

def abscorr(v1, v2):
    v1_norm = v1 #- v1.mean()
    v2_norm = v2 #- v2.mean()
    x= ((v1_norm * v2_norm).sum() / (torch.norm(v1_norm, 2) * torch.norm(v2_norm, 2))).squeeze().abs()
    return x

def env_errors(environments):
    errors = []
    loss = torch.nn.MSELoss()
    for i, (x_e, y_e) in enumerate(environments):
        coef = torch.Tensor(LinearRegression(fit_intercept=False).fit(x_e, y_e).coef_).t()
        error = loss(x_e @ coef, y_e)
        errors.append(float(error.detach()))
    return errors

class InvariantRiskMinimization(object):
    def __init__(self, environments, args, setup_str=''):
        best_reg = 0
        best_err = 1e6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        environments = [(x.to(self.device), y.to(self.device)) for x,y in environments]

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            logging.debug(f'regularizer {reg}')
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        print(f"IRM best {setup_str}: reg={best_reg:.3g}) has {best_err:.3g} validation error.")
        self.phi = best_phi

    def train(self, environments, args, reg=0):
        reg = torch.Tensor([reg]).to(self.device)
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x, device=self.device))
        self.w = torch.ones(dim_x, 1, device=self.device)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss().to(self.device)


        for iteration in range(args["n_iterations"]):
            penalty = torch.zeros((1,1), device=self.device)
            error = torch.zeros((1,1), device=self.device)

            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).squeeze().backward()
            opt.step()

            if False and args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      float(reg.detach()),
                                                                      float(error.detach()),
                                                                      float(penalty.detach()),
                                                                      w_str))

    def solution(self):
        return self.phi @ self.w


class InvariantRiskMinimizationSimple(object):
    """ Same as above, but using a single scaling weight. In fact, works better! """
    def __init__(self, environments, args, setup_str=''):
        best_reg = 0
        best_err = 1e6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        environments = [(x.to(self.device), y.to(self.device)) for x,y in environments]

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [1e-2]: # [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            logging.debug(f'regularizer {reg}')
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRMS (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        print(f"IRMS best {setup_str}: reg={best_reg:.3g}) has {best_err:.3g} validation error.")
        self.phi = best_phi

    def train(self, environments, args, reg=0):
        reg = torch.Tensor([reg]).to(self.device)
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.rand(dim_x, 1))
        self.w = torch.ones(1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss().to(self.device)

        for iteration in range(args["n_iterations"]):
            penalty = torch.zeros((1,1), device=self.device)
            error = torch.zeros((1,1), device=self.device)

            penalties_raw = []
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi * self.w, y_e)
                penalty_raw =  grad(error_e, self.w, create_graph=True)[0]
                penalties_raw.append(penalty_raw)
                penalty += penalty_raw.pow(2).mean()
                error += error_e

            penalties_raw = torch.cat(penalties_raw, -1)
            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).squeeze().backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      float(reg.detach()),
                                                                      float(error.detach()),
                                                                      float(penalty.detach()),
                                                                      w_str))
                print('PEN', penalties_raw)

    def solution(self):
        return self.phi * self.w

class InvariantCausalPrediction(object):
    def __init__(self, environments, args, setup_str=''):
        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []

        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)

            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients


class EmpiricalRiskMinimizer(object):
    def __init__(self, environments, args, setup_str='', weighted=False):
        x_all = torch.cat([x for (x, y) in environments]).numpy()
        y_all = torch.cat([y for (x, y) in environments]).numpy()

        if not weighted:
            phi = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
            self.phi = torch.Tensor(phi)
        else:
            base = env_errors(environments)
            reg = min(base)
            self.w = [(min(base) + reg) / (x + reg) for x in base]
            w_all = np.concatenate([[self.w[i]] * len(environments[i][0]) for i in range(len(environments))])
            phi = LinearRegression(fit_intercept=False).fit(x_all, y_all, w_all).coef_
            self.phi = torch.Tensor(phi)

    def solution(self):
        return self.phi


class RiskMinimizationGames(object):
    def __init__(self, environments, args, setup_str=''):
        self.train(environments, args)

    def train(self, environments, args):

        dim_x = environments[0][0].size(1)
        self.phi = [torch.nn.Parameter(torch.rand(dim_x, 1)) for i in range(len(environments))]
        self.phi_best = None
        opt = torch.optim.Adam(self.phi, lr=1e-3)
        error_e = [0] * len(environments)
        score = [0] * len(environments)
        loss = torch.nn.MSELoss()
        error_best = float('inf')

        for iteration in range(args["n_iterations"]):
            error = 0
            opt.zero_grad()
            for i, (x_e, y_e) in enumerate(environments):
                for j in range(len(score)):
                    self.phi[j].requires_grad = (i == j)
                score = [x_e @ p / len(environments) for p in self.phi]
                error_e[i] = loss(sum(score), y_e)
                error_e[i].backward()
                error += float(error_e[i].detach())
            opt.step()
            if error < error_best:
                error_best = error
                self.phi_best = self.phi

            #if iteration % 1 == 0:
            #    #print('GAMES', iteration, error, error_e, 'phiavg', sum(self.phi)/len(self.phi), 'phi:', self.phi)
            #    print('GAMES', iteration, error)

        if args["verbose"]:
            print("RMG has {:.3f} validation error.".format(error_best))


    def solution(self):
        return sum(self.phi_best)/len(self.phi_best)


class MAML(object):
    """ Memory-augmented meta learning """
    def __init__(self, environments, args, setup_str=''):

        self.train(environments, args)


    def split_valid(self, e, k):
        i = random.choices(range(e[0].shape[0]),k=k)
        i_inv = list(set(range(e[0].shape[0])) - set(i))
        return [(x[i], x[i_inv]) for x in e]


    def train(self, environments, args, reg=0):

        n = environments[0][0].size(1)
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.rand(dim_x, 1))
        self.phi_best = copy.deepcopy(self.phi)
        err_best = float('inf')
        iter_best = 0

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        steps_inner = args["n_iterations_inner"]
        lr_inner = args["lr_inner"]
        sz_valid = int(.8 * n)
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            opt.zero_grad()
            error = torch.zeros(1)
            for x_e, y_e in environments:
                ((x_tr, x_ts), (y_tr, y_ts)) = self.split_valid((x_e, y_e), sz_valid)

                phi_inner = self.phi.clone()
                for j in range(steps_inner):
                    error_e = loss(x_tr @ phi_inner, y_tr)
                    grad = torch.autograd.grad(error_e, phi_inner)
                    phi_inner -= lr_inner * grad[0]

                error += loss(x_ts @ phi_inner, y_ts) / len(environments)

            err_flt = float(error.detach())
            if err_flt < err_best:
                iter_best = iteration
                self.phi_best = copy.deepcopy(self.phi)
                err_best = err_flt

            error.backward()
            opt.step()

            if args["verbose"] and iteration % 100 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {}".format(iteration,
                                                    err_flt,
                                                    w_str))
        print(f'best iter: {iter_best} {err_best}')
        self.phi = self.phi_best

    def solution(self):
        return self.phi


class MetaSGD(object):
    """ meta learning """
    def __init__(self, environments, args, setup_str=''):

        self.train(environments, args)


    def split_valid(self, e, k):
        i = random.choices(range(e[0].shape[0]),k=k)
        i_inv = list(set(range(e[0].shape[0])) - set(i))
        return [(x[i], x[i_inv]) for x in e]


    def train(self, environments, args, reg=0):

        n = environments[0][0].size(1)
        dim_x = environments[0][0].size(1)

        base = env_errors(environments)
        reg = min(base)
        self.w = [(min(base) + reg) / (x + reg) for x in base]
        #self.w = [1.0] * len(environments)
        print('WEIGHTS', self.w)

        self.phi = torch.nn.Parameter(torch.rand(dim_x, 1))
        self.alpha = torch.nn.Parameter(torch.ones(dim_x, 1) * args["lr_inner"])

        opt = torch.optim.Adam([self.phi, self.alpha], lr=args["lr"])
        steps_inner = args["n_iterations_inner"]
        sz_valid = int(.5 * n)
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            opt.zero_grad()
            error = torch.zeros(1)
            for i, (x_e, y_e) in enumerate(environments):
                ((x_tr, x_ts), (y_tr, y_ts)) = self.split_valid((x_e, y_e), sz_valid)

                phi_inner = self.phi.clone()
                for j in range(steps_inner):
                    error_e = loss(x_tr @ phi_inner, y_tr)
                    grad = torch.autograd.grad(error_e, phi_inner)
                    phi_inner -= self.alpha * grad[0]

                error_e = self.w[i] * loss(x_ts @ phi_inner, y_ts) / len(environments)
                if args["verbose"] and iteration % 100 == 0:
                    g = torch.autograd.grad(error_e, self.alpha, retain_graph = True)
                    alpha_str = pretty(g[0])
                    print(f'GRAD {i} {float(error_e.detach()):.5f} | {alpha_str}')
                error += error_e

            error.backward()
            opt.step()

            if args["verbose"] and iteration % 100 == 0:
                w_str = pretty(self.solution())
                alpha_str = pretty(self.alpha)
                err_flt = float(error.detach())

                print("{:05d} | {:.5f} | {} | {}".format(iteration,
                                                         err_flt,
                                                         w_str,
                                                         alpha_str))


    def solution(self):
        return self.phi


class SpecialistPenalty(object):

    """ WIP"""

    def __init__(self, environments, args, setup_str=''):

        self.train(environments, args)

    def train(self, environments, args, reg=0):

        loss = torch.nn.MSELoss()
        models = [EmpiricalRiskMinimizer([e], args) for e in environments]

        for i, (x_e, y_e) in enumerate(environments):
            with torch.no_grad():
                pred = x_e @ models[i].solution().t()
                error = loss(pred, y_e)
                err_flt = float(error.detach())
                w_str = pretty(models[i].solution())
                print("{} | {:.5f} | {}".format(i,
                                                    err_flt,
                                                    w_str))

        sys.exit(0)


        n = environments[0][0].size(1)
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.rand(dim_x, 1))

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        alpha = args["alpha"]


        for iteration in range(args["n_iterations"]):
            opt.zero_grad()
            error = torch.zeros(1)
            for i, (x_e, y_e) in enumerate(environments):
                y_imp = sum(x_e @ models[j].solution().t() for j in range(len(environments)) if j != i) / len(environments)

                pred = x_e @ self.phi
                error += (1 - alpha) * loss(pred, y_e) + alpha * loss(pred, y_imp)

            error.backward()
            opt.step()

            if args["verbose"] and iteration % 100 == 0:
                err_flt = float(error.detach())
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {}".format(iteration,
                                                    err_flt,
                                                    w_str))

    def solution(self):
        return self.phi


class SpecialistRiskGames(object):

    """ WIP """
    def __init__(self, environments, args, setup_str=''):
        self.train(environments, args)

    def train(self, environments, args, reg=0.05):

        pen_a = 0 # reg

        pen_reg = 0 # 1e-2
        pen_corr = reg

        dim_x = environments[0][0].size(1)
        n = len(environments) + 1
        # first model is common, the others are environment specialists
        self.phi = [torch.nn.Parameter(torch.rand(dim_x, 1)) for i in range(n)]
        opt = torch.optim.Adam(self.phi, lr=1e-3)

        loss = torch.nn.MSELoss()
        score_e = [0] * n
        error_e = torch.zeros(n-1)

        with torch.no_grad():
            for i, (x_e, y_e) in enumerate(environments):
                error_e[i] = y_e.var()

        w = torch.ones((n-1))
        for i in range(n-1):
            w[i] = 1.0 / error_e[i]
            w[i] = w[i].log()
        w = torch.nn.Parameter(w)

        w_rel = (n-1) * w.exp() / w.exp().sum()

        for iteration in range(args["n_iterations"]):
            #w_rel = 1 / error_e
            #w_rel /= w_rel.sum()

            opt.zero_grad()
            err_all = torch.zeros(1)
            err_fit_all = torch.zeros(1)
            err_reg_all = torch.zeros(1)
            err_a_all = torch.zeros(1)

            for i, (x_e, y_e) in enumerate(environments):

                v = torch.zeros(1)
                for j in range(n):
                    score_e[j] = x_e @ self.phi[j]
                    if j != 0 and j != i+1:
                        v += score_e[j].var()

                # err_a = pen_a * v
                score = score_e[0] + score_e[i+1]

                err_fit = loss(score, y_e)

                #err_irm = grad(error_fit, self.w0,
                #                create_graph=True)[0].pow(2).mean()

                # score_e[i] = err_fit.detach()

                err_corr = abscorr(score_e[i+1], score_e[0])

                err_all += (pen_corr * err_corr + (1-pen_corr) * err_fit) # * w_rel[i].detach()

            # err_reg = pen_reg * torch.norm(torch.cat(self.phi[1:]), 1) + pen_reg * torch.norm(self.phi[0],1) / (n-1)

            err_all.backward()
            opt.step()


    def solution(self):
        return self.phi[0]


class GradVarPenalty(object):
    """
    WIP
    use penalty w * var_task(grad)
    """
    def __init__(self, environments, args, setup_str=''):

        self.alpha = args["alpha"]
        self.train(environments, args)

    def train(self, environments, args, reg=0):

        n = environments[0][0].size(1)
        dim_x = environments[0][0].size(1)

        # self.phi = torch.nn.Parameter(torch.rand(dim_x, 1))
        self.phi = torch.nn.Parameter(EmpiricalRiskMinimizer(environments, args).solution().view(dim_x, 1))

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            opt.zero_grad()

            error = torch.zeros(1)
            pen_mrg = torch.zeros(1)
            grad = []
            for i, (x_e, y_e) in enumerate(environments):
                error_e = loss(x_e @ self.phi, y_e)
                #print('er', self.phi, error_e)
                grad_e = torch.autograd.grad(error_e, self.phi, retain_graph=True)[0]
                grad.append(grad_e)
                error += error_e

            grad = torch.cat(grad, dim=1)
            m = grad.mean(dim=1)
            p = grad.var(dim=1) # / (m * m + 1e-3)
            p_sum = (self.phi * self.phi * p).sum()
            #print(grad)
            assert not torch.isnan(p_sum).any()
            error_t = (1-self.alpha) * error + self.alpha * p_sum
            grad_p = torch.autograd.grad(p_sum, self.phi, retain_graph=True)[0]
            grad_ee = torch.autograd.grad(error, self.phi, retain_graph=True)[0]

            # print(error)
            error_t.backward()
            opt.step()
            if args["verbose"] and iteration % 1000 == 0:
                err_flt = (1-self.alpha) * float(error.detach())
                p_flt = self.alpha * float(p_sum.detach())
                w_str = pretty(self.solution())
                p_str = pretty(p)
                print("{:05d} | {:.5f} | {:.5f} | {} | {}".format(iteration,
                                                                  err_flt,
                                                                  p_flt,
                                                                  w_str,
                                                                  p_str))

                print('GRAD error', - grad)
                print('P', - p)
                print('uncentered', - grad.pow(2).sum(dim=1))
                print('GRAD penalty', - grad_p)
                print('GRAD EE', - grad_ee)
                print(f'phi grad {pretty(self.phi.grad)}')
                print(f'm {pretty(m)} v {pretty(p)}')



    def solution(self):
        return self.phi
