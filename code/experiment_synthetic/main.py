#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sem import ChainEquationModel
from models import *

import argparse
import torch
import numpy

import multiprocessing as mp
from joblib import Parallel, delayed

def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    i_causal = (w != 0).nonzero().view(-1)
    i_noncausal = (w == 0).nonzero().view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
        error_causal = error_causal.item()
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
        error_noncausal = error_noncausal.item()
    else:
        error_noncausal = 0

    return error_causal, error_noncausal


def run_methods(methods, setup_str, sem, environments, args, lock):

    logging.basicConfig(level=getattr(logging, args["log_level"]), format="[%(asctime)s\t%(levelname)s]\t%(message)s")

    solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str,
                                               pretty(sem.solution()), 0, 0)
        ]

    for method_name, method_constructor in methods.items():

        logging.info(f'run {method_name} {setup_str}')
        method = method_constructor(environments, args)
        msolution = method.solution()

        err_causal, err_noncausal = errors(sem.solution_unscrambled(), sem.scramble @ msolution.view(-1,1))

        solution = "{} {} {} {:.5f} {:.5f}".format(setup_str,
                                                   method_name,
                                                   pretty(msolution),
                                                   err_causal,
                                                   err_noncausal)
        logging.info(solution)
        solutions.append(solution)

    lock.acquire()
    with open(args["out_file"], "a") as f:
        for solution in solutions:
            f.write(solution + '\n')
    lock.release()

    return solutions


def run_experiment(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        # torch.set_num_threads(1)

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "ICP": InvariantCausalPrediction,
        "IRM": InvariantRiskMinimization,
        "RMG": RiskMinimizationGames
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_solutions = []
    all_environments = []
    all_setup_strs = []

    if args["setup_sem"] == "chain":
        for rep_i in range(args["n_reps"]):
            for hidden in args["setup_hidden"]:
                for hetero in args["setup_hetero"]:
                    for scramble in args["setup_scramble"]:
                        sem = ChainEquationModel(args["dim"],
                                                 hidden=hidden,
                                                 scramble=scramble,
                                                 hetero=hetero)
                        environments = [sem(args["n_samples"], .2),
                                        sem(args["n_samples"], 2.),
                                        sem(args["n_samples"], 5.)]
                        setup_str = "chain_hidden={}_hetero={}_scramble={}".format(
                            hidden,
                            hetero,
                            scramble)
                        all_sems.append(sem)
                        all_environments.append(environments)
                        all_setup_strs.append(setup_str)
    else:
        raise NotImplementedError

    # write lock to avoid overwriting output file from multiple threads
    m = mp.Manager()
    l = m.Lock()

    solutions = Parallel(n_jobs=4)(delayed(run_methods)
                                   (methods, setup_str, sem, environments, args, l) for sem, environments, setup_str in zip(all_sems, all_environments, all_setup_strs))

    all_solutions += solutions

    return all_solutions


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Invariant regression')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--skip_reps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)  # Negative is random
    parser.add_argument('--print_vectors', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default = 'DEBUG', help="Set the logging level")
    parser.add_argument('--methods', type=str, default="ERM,ICP,IRM,RMG")
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--setup_sem', type=str, default="chain")
    parser.add_argument('--setup_hidden', type=int, nargs='+', default=[0])
    parser.add_argument('--setup_hetero', type=int, nargs='+', default=[0])
    parser.add_argument('--setup_scramble', type=int, nargs='+', default=[0])
    parser.add_argument('--out_file', type=str, default='tmp.txt')
    args = dict(vars(parser.parse_args()))

    logging.basicConfig(level=getattr(logging, args["log_level"]), format="[%(asctime)s\t%(levelname)s]\t%(message)s")
    run_experiment(args)

