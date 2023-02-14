import json
import multiprocessing as mp
from itertools import product
from multiprocessing.pool import Pool
import sys
import os

import numpy as np
from jmetal.operator import PolynomialMutation, RandomSolutionSelection, SBXCrossover, BinaryTournamentSelection
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
    "selection": BinaryTournamentSelection(),
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
        cat_vals = [hist[k] for hist in histories]
        joined_history[k] = cat_vals

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
        
        
def read_build_kwargs(fname: str) -> dict:
    fpath = f"results/kwargs/{fname}.json"
    if not os.path.isfile(fpath):
        return {}
    with open(f"results/kwargs/{fname}.json", "r") as f:
        return json.load(f)

        
def find_kwargs():
    P_processes = 4
    I_iterations = 15

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
        (
            buildFollowBestAlg,
            "follow_best",
            {
                "probability": [0.4, 1],
                "tracked_best_count": [5, 10, 20],
                "follow_rate": [0.1],
            },
        ),
        (
            buildFollowDistinctBestAlg,
            "follow_distinct_best",
            {
                "probability": [0.4, 1],
                "tracked_best_count": [5, 10, 20],
                "copy_genes_alpha": [1, 2, 5],
                "follow_rate": [0.1],
            },
        ),
        (
            buildRepelWorstGravityAlg,
            "repel_worst_gravity",
            {
                "probability": [0.4, 1],
                "tracked_worst_count": [5, 10, 20],
                "repel_rate": [0.1],
            },
        ),
        (
            buildDistinctGravityComboAlg,
            "combo_distinct_gravity",
            {
                "follow_probability": [0.4, 1],
                "repel_probability": [0.4, 1],
                "tracked_best_count": [5, 10, 20],
                "copy_genes_alpha": [1, 2, 5],
                "tracked_worst_count": [5, 10, 20],
                "follow_rate": [0.1],
                "repel_rate": [0.1],
            },
        ),
        (
            buildDistinctGravityMultistepComboAlg,
            "combo_distinct_gravity_multistep",
            {
                "follow_probability": [0.4, 1],
                "repel_probability": [0.4, 1],
                "tracked_best_count": [5, 10, 20],
                "copy_genes_alpha": [1, 2, 5],
                "tracked_worst_count": [5, 10, 20],
                "follow_rate": [0.1],
                "repel_rate": [0.1],
            },
        ),
    ]

    problems = [ptype(psize) for ptype in problem_types for psize in problem_sizes]

    with mp.Pool(P_processes) as pool:
        
        alg_combos = [
            (alg, alg_name, nkw, problem)
            
            for (alg, alg_name, nkw) in runs 
            for problem in problems]
        
        for (build_alg_fn, alg_name, nkw, problem) in tqdm(alg_combos):
       
            arg_combos = list((
                ({k: v for k, v in zip(nkw.keys(), vals)})

                for vals in product(*nkw.values())
                for p in problems
            ))
                
            eval_args = [(problem, I_iterations, build_alg_fn, build_kwargs) for build_kwargs in arg_combos]
            
            results = pool.starmap(evalAlg, eval_args)

            best_kwargs, best_h = min(zip(arg_combos,results), key=lambda res: res[1]["best"][-1])

            fname = f"{alg_name}_{problem.get_name()}_{problem.number_of_variables}"
            
            save_build_kwargs(kwargs=best_kwargs, fname=fname)
    
def eval_kwargs():
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
        (
            buildBaseAlg,
            "base_algorithm", 
        ),
        (
            buildFollowBestAlg,
            "follow_best",
        ),
        (
            buildFollowDistinctBestAlg,
            "follow_distinct_best",
        ),
        (
            buildRepelWorstGravityAlg,
            "repel_worst_gravity",
        ),
        (
            buildDistinctGravityComboAlg,
            "combo_distinct_gravity",
        ),
        (
            buildDistinctGravityMultistepComboAlg,
            "combo_distinct_gravity_multistep",
        ),
    ]

    problems = [ptype(psize) for ptype in problem_types for psize in problem_sizes]

    with mp.Pool(P_processes) as pool:
        
        alg_combos = [
            (alg, alg_name, problem)
            
            for (alg, alg_name) in runs 
            for problem in problems]
        
        for (build_alg, alg_name, problem) in tqdm(alg_combos):
            
            fname = f"{alg_name}_{problem.get_name()}_{problem.number_of_variables}"
            
            build_kwargs = read_build_kwargs(fname)

            h = multirunEvalAlg(
                repeats=N_repeats,
                pool=pool,
                problem=problem,
                iterations=I_iterations,
                buildAlgFn=build_alg,
                build_kwargs=build_kwargs
            )
            
            saveHistory(history=h, fname=fname)

    
    
if __name__ == "__main__":
    usage = "pass argument \"find\" or \"eval\" to run script"
    
    if len(sys.argv) != 2:
        print(usage)
        exit()
    
    if sys.argv[1] == "find":
        find_kwargs()
        exit()
    elif sys.argv[1] == "eval":
        eval_kwargs()
        exit()
        
    
    
    
    
    
    
    
    
    
    
    