from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import (
    SBXCrossover,
    RandomSolutionSelection,
    PolynomialMutation,
    NullMutation,
)
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Rastrigin

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from jmetalpy_social.algorithm import *
from jmetalpy_social.operator import *
from jmetalpy_social.problem import *

import multiprocessing as mp

import json

shared_kwargs = {
    "population_size": 200,
    "mating_pool_size": 20,
    "offspring_population_size": 140,
    "crossover": SBXCrossover(probability=1.0, distribution_index=20),
    "selection": RandomSolutionSelection(),
}


def buildBaseAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": PolynomialMutation(probability=1 / D, distribution_index=20),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildFollowBestAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                FollowBestMutation(
                    probability=0.4, tracked_best_count=5, follow_rate=0.1
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildFollowSharedBestAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                FollowBestSharedGenesMutation(
                    probability=0.4,
                    tracked_best_count=5,
                    copy_genes_count=D // 2,
                    follow_rate=0.1,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildFollowDistinctBestAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                FollowBestDistinctGenesMutation(
                    probability=0.4,
                    tracked_best_count=5,
                    copy_genes_count=D // 2,
                    follow_rate=0.1,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildRepelWorstAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                RepelWorstMutation(
                    probability=0.4, tracked_worst_count=5, repel_rate=0.1
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildRepelWorstSparseAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                RepelWorstMutationSparse(
                    probability=0.4, tracked_worst_count=5, repel_rate=0.1
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildRepelWorstNormAlg(problem, iterations):
    e = float(np.e)
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                RepelWorstMutationNorm(
                    probability=0.4,
                    tracked_worst_count=5,
                    norm_fn=lambda x: x / abs(x + 1e-6) * (abs(x) + 1) ** -2,
                    # norm_fn=lambda x: -(x / 5 - x / abs(x + 1e-6)),
                    # norm_fn=lambda x: x / (abs(x) + 1e-6) * e ** (-abs(x)),
                    #  np.sign(x) * np.exp(-np.abs(x)),
                    repel_rate=0.1,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildRepelWorstMeanAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                RepelWorstMutationMean(
                    probability=0.4,
                    tracked_worst_count=5,
                    repel_rate=0.1,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildRepelWorstGravityAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                RepelWorstMutationGravity(
                    probability=0.4, tracked_worst_count=5, repel_rate=0.1
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildRepelWorstGravityMultistepAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                RepelWorstMutationGravityMutlistep(
                    probability=0.4, tracked_worst_count=5, repel_rate=0.1
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildDistinctGravityMultistepComboAlg(problem, iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update = {
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
                PolynomialMutation(probability=1 / D, distribution_index=20),
                FollowBestDistinctGenesMutation(
                    probability=0.4,
                    tracked_best_count=5,
                    copy_genes_count=D // 2,
                    follow_rate=0.1,
                ),
                RepelWorstMutationGravityMutlistep(
                    probability=1, tracked_worst_count=5, repel_rate=0.1
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def joinHistories(histories):
    joined_history = {}

    for k, v in histories[0].items():
        cat_vals = np.array([hist[k] for hist in histories])
        joined_history[k] = np.mean(cat_vals, axis=0).tolist()

    return joined_history


def evalAlg(problem, iterations, buildAlgFn, i):
    algorithm = buildAlgFn(problem, iterations)
    algorithm.verbose = False

    algorithm.run()

    return algorithm.history


def multirunEvalAlg(repeats, pool, problem, iterations, buildAlgFn):
    args = [(problem, iterations, buildAlgFn, i) for i in range(repeats)]
    results = pool.starmap(evalAlg, args)

    return joinHistories(results)


def saveHistory(history, fname):
    with open(f"results/{fname}.json", "w+") as f:
        f.write(json.dumps(history))


if __name__ == "__main__":

    D_dimension = 1000
    N_repeats = 4
    P_processes = 4
    I_iterations = 80

    problems = [
        AckleyProblem(D_dimension),
        # DeJongProblem(D_dimension),
        # RastriginProblem(D_dimension),
        # GriewankProblem(D_dimension),
    ]
    algorithms = [
        # (buildBaseAlg, "base_algorithm"),
        # (buildFollowBestAlg, "follow_best"),
        # (buildFollowSharedBestAlg, "follow_shared_best"),
        (buildFollowDistinctBestAlg, "follow_distinct_best"),
        # (buildRepelWorstAlg, "repel_worst"),
        # (buildRepelWorstSparseAlg, "repel_worst_sparse"),
        # (buildRepelWorstNormAlg, "repel_worst_norm"),
        # (buildRepelWorstMeanAlg, "repel_worst_mean"),
        # (buildRepelWorstGravityAlg, "repel_worst_gravity"),
        # (buildRepelWorstGravityMultistepAlg, "repel_worst_gravity_multistep_p04"),
        # (buildDistinctGravityMultistepComboAlg, "combo_distinct_gravity_multistep_p1"),
    ]

    with mp.Pool(P_processes) as pool:
        combos = [(p, alg, alg_name) for p in problems for alg, alg_name in algorithms]

        for problem, buildAlg, algName in tqdm(combos):
            h = multirunEvalAlg(N_repeats, pool, problem, I_iterations, buildAlg)

            saveHistory(h, f"{algName}_{problem.get_name()}")
