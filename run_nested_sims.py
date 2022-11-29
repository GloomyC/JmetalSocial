import json
import multiprocessing as mp
from itertools import product
from multiprocessing.pool import Pool

import numpy as np
from jmetal.operator import PolynomialMutation, RandomSolutionSelection, SBXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations
from tqdm import tqdm

from jmetalpy_social.algorithm import *
from jmetalpy_social.operator import *
from jmetalpy_social.problem import *

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


def buildFollowBestAlg(
    problem,
    iterations,
    probability: float = 0.4,
    tracked_best_count: int = 5,
    follow_rate: float = 0.1,
):
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
                    probability=probability,
                    tracked_best_count=tracked_best_count,
                    follow_rate=follow_rate,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildFollowDistinctBestAlg(
    problem,
    iterations,
    probability: float = 0.4,
    tracked_best_count: int = 5,
    copy_genes_alpha: float = 2,
    follow_rate: float = 0.1,
):
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
                    probability=probability,
                    tracked_best_count=tracked_best_count,
                    copy_genes_count=int(D / copy_genes_alpha),
                    follow_rate=follow_rate,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildRepelWorstGravityAlg(
    problem,
    iterations,
    probability: float = 0.4,
    tracked_worst_count: int = 5,
    repel_rate: float = 0.1,
):
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
                    probability=probability,
                    tracked_worst_count=tracked_worst_count,
                    repel_rate=repel_rate,
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
                    probability=0.4, tracked_worst_count=5, repel_rate=1
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildDistinctGravityComboAlg(
    problem,
    iterations,
    follow_probability: float = 0.4,
    tracked_best_count: int = 5,
    copy_genes_alpha: float = 2,
    follow_rate: float = 0.1,
    repel_probability: float = 0.4,
    tracked_worst_count: int = 5,
    repel_rate: float = 0.1,
):
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
                    probability=follow_probability,
                    tracked_best_count=tracked_best_count,
                    copy_genes_count=int(D / copy_genes_alpha),
                    follow_rate=follow_rate,
                ),
                RepelWorstMutationGravity(
                    probability=repel_probability,
                    tracked_worst_count=tracked_worst_count,
                    repel_rate=repel_rate,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def buildDistinctGravityMultistepComboAlg(
    problem,
    iterations,
    follow_probability: float = 0.4,
    tracked_best_count: int = 5,
    copy_genes_alpha: float = 2,
    follow_rate: float = 0.1,
    repel_probability: float = 0.4,
    tracked_worst_count: int = 5,
    repel_rate: float = 0.1,
):
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
                    probability=follow_probability,
                    tracked_best_count=tracked_best_count,
                    copy_genes_count=int(D / copy_genes_alpha),
                    follow_rate=follow_rate,
                ),
                RepelWorstMutationGravityMutlistep(
                    probability=repel_probability,
                    tracked_worst_count=tracked_worst_count,
                    repel_rate=repel_rate,
                ),
            ]
        ),
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)

    return algorithm


def joinHistories(histories: list[dict]) -> dict:
    joined_history = {}

    for k, v in histories[0].items():
        cat_vals = np.array([hist[k] for hist in histories])
        joined_history[k] = np.mean(cat_vals, axis=0).tolist()

    return joined_history


def evalAlg(
    problem: FloatProblem, iterations: int, buildAlgFn: callable, build_kwargs: dict
) -> dict:
    algorithm = buildAlgFn(problem=problem, iterations=iterations, **build_kwargs)
    algorithm.verbose = False

    algorithm.run()

    return algorithm.history


def multirunEvalAlg(
    repeats: int,
    pool: Pool,
    problem: FloatProblem,
    iterations: int,
    buildAlgFn: callable,
    build_kwargs: dict,
) -> dict:
    args = [(problem, iterations, buildAlgFn, build_kwargs) for i in range(repeats)]
    results = pool.starmap(evalAlg, args)

    return joinHistories(results)


def saveHistory(history: dict, fname: str) -> None:
    with open(f"results/{fname}.json", "w+") as f:
        f.write(json.dumps(history))


def save_build_kwargs(kwargs: dict, fname: str) -> None:
    with open(f"results/kwargs/{fname}.json", "w") as f:
        json.dump(kwargs, f, indent=2)


if __name__ == "__main__":

    N_repeats = 12
    P_processes = 4
    I_iterations = 100

    problem_sizes = [
        100,
        500,
        1000,
    ]

    problem_types = [
        AckleyProblem,
        DeJongProblem,
        RastriginProblem,
        GriewankProblem,
    ]

    runs = [
        (buildBaseAlg, "base_algorithm", {}),
        (
            buildFollowBestAlg,
            "follow_best",
            {
                "probability": [0.1, 0.4, 1],
                "tracked_best_count": [1, 5, 10, 100],
                "follow_rate": [0.1, 0.2, 0.5],
            },
        ),
        (
            buildFollowDistinctBestAlg,
            "follow_distinct_best",
            {
                "probability": [0.1, 0.4, 1],
                "tracked_best_count": [1, 5, 10, 100],
                "copy_genes_alpha": [1, 2, 10],
                "follow_rate": [0.1, 0.2, 0.5],
            },
        ),
        (
            buildRepelWorstGravityAlg,
            "repel_worst_gravity",
            {
                "probability": [0.4, 1],
                "tracked_worst_count": [1, 10, 100],
                "repel_rate": [0.1, 0.5, 1, 10],
            },
        ),
        (
            buildDistinctGravityComboAlg,
            "combo_distinct_gravity",
            {
                "follow_probability": [0.1, 0.4, 1],
                "tracked_best_count": [1, 5, 10, 100],
                "copy_genes_alpha": [1, 2, 10],
                "follow_rate": [0.1, 0.2, 0.5],
                "repel_probability": [0.4, 1],
                "tracked_worst_count": [1, 10, 100],
                "repel_rate": [0.1, 0.5, 1, 10],
            },
        ),
        (
            buildDistinctGravityMultistepComboAlg,
            "combo_distinct_gravity_multistep",
            {
                "follow_probability": [0.1, 0.4, 1],
                "tracked_best_count": [1, 5, 10, 100],
                "copy_genes_alpha": [1, 2, 10],
                "follow_rate": [0.1, 0.2, 0.5],
                "repel_probability": [0.4, 1],
                "tracked_worst_count": [1, 10, 100],
                "repel_rate": [0.1, 0.5, 1, 10],
            },
        ),
    ]

    problems = [ptype(psize) for ptype in problem_types for psize in problem_sizes]

    with mp.Pool(P_processes) as pool:

        combos = (
            (p, alg, alg_name, {k: v for k, v in zip(nkw.keys(), vals)}, i)
            for p in problems
            for (alg, alg_name, nkw) in runs
            for i, vals in enumerate(product(*nkw.values()))
        )

        for problem, build_alg, alg_name, build_kwargs, i in tqdm(combos):
            h = multirunEvalAlg(
                repeats=N_repeats,
                pool=pool,
                problem=problem,
                iterations=I_iterations,
                buildAlgFn=build_alg,
                build_kwargs=build_kwargs,
            )

            fname = f"{alg_name}_{i}_{problem.get_name()}_{problem.number_of_variables}"
            saveHistory(history=h, fname=fname)
            save_build_kwargs(kwargs=build_kwargs, fname=fname)
