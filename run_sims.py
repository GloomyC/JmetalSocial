from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import SBXCrossover, RandomSolutionSelection, PolynomialMutation, NullMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Rastrigin

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from jmetalpy_social.algorithm import *
from jmetalpy_social.operator import *
from jmetalpy_social.problem import *

import multiprocessing as mp


shared_kwargs={
    "population_size": 200,
    "mating_pool_size": 20,
    "offspring_population_size": 140,
    "crossover": SBXCrossover(probability=1.0, distribution_index=20),
    "selection": RandomSolutionSelection(),
}

def buildAlg0(problem,iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update={
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": PolynomialMutation(probability=1/D, distribution_index=20)
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)
    
    return algorithm

def buildAlg1(problem,iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update={
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
            PolynomialMutation(probability=1/D, distribution_index=20),
            FollowBestMutation(
                probability=0.4,
                tracked_best_count=5),
            ]
        )
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)
    
    return algorithm

def buildAlg2(problem,iterations):
    D = problem.number_of_variables
    kwargs = shared_kwargs.copy()
    kwargs_update={
        "problem": problem,
        "termination_criterion": StoppingByEvaluations(max_evaluations=iterations),
        "keep_parents": True,
        "mutation": IndependantMutations(
            [
            PolynomialMutation(probability=1/D, distribution_index=20),
            FollowBestSharedGenesMutation(
                probability=0.4,
                tracked_best_count=5,
                copy_genes_count=D//4),
            ]
        )
    }
    kwargs.update(kwargs_update)
    algorithm = ObservedPaperAlgorithm(**kwargs)
    
    return algorithm

def joinHistories(histories):
    
    joined_history = {}
    
    for k,v in histories[0].items():
        
        cat_vals = np.array([hist[k] for hist in histories])
        
        joined_history[k] = np.mean(cat_vals, axis=0)
        
    return joined_history

def evalAlg(problem, iterations, buildAlgFn, i):
    print(f"EVAL STARTED {i}")
    algorithm = buildAlgFn(problem,iterations)
    algorithm.verbose = False
    
    algorithm.run()
    print(f"EVAL DONE {i}")

        
    return algorithm.history

def multirunEvalAlg(repeats, pool, problem, iterations, buildAlgFn):
        
    args = [(problem, iterations, buildAlgFn, i) for i in range(repeats)]
    
    results = pool.starmap(evalAlg,args)
    
    return joinHistories(results)
    
if __name__ == '__main__':
    
    D_dimension = 400
    N_repeats = 5
    P_processes = 2
    I_iterations = 100
    
    problems = [AckleyProblem(D_dimension)]
    algorithms = [buildAlg0, buildAlg1, buildAlg2]

    
    with mp.Pool(P_processes) as pool:
        for problem in problems:
            for buildAlg in algorithms:
                h = multirunEvalAlg(N_repeats, pool, problem, I_iterations, buildAlg)

                print(h['best'])
                print(h['average'])

    
    
