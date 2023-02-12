from jmetal.algorithm.singleobjective import GeneticAlgorithm
import numpy as np
import random
import time
from tqdm import tqdm


class PaperAlgorithm(GeneticAlgorithm):
    def __init__(self, 
               *args,
               mating_pool_size: int,
               keep_parents: bool,
               **kwargs):
        GeneticAlgorithm.__init__(self,*args, **kwargs)

        self.keep_parents = keep_parents
        self.offspring_population_size = kwargs['offspring_population_size']
        self.mating_pool_size = mating_pool_size
        self.mutation_operator.set_tracked_algorithm(self)

    def reproduction(self, mating_population):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        offspring_population = []
        for i in range(0, self.offspring_population_size):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(random.choice(mating_population))

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                solution = self.mutation_operator.execute(solution)
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def step(self):

        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        if not self.keep_parents:
            self.solutions = set(self.solutions)
            for parent in mating_population:
                if parent in self.solutions:
                    self.solutions.remove(parent)
            self.solutions = list(self.solutions)

        self.solutions = self.replacement(self.solutions, offspring_population)
        
        
class ObservedPaperAlgorithm(PaperAlgorithm):
    def __init__(self, *args, **kwargs):
        PaperAlgorithm.__init__(self, *args, **kwargs)

        self.history = {"best":[],
                        "average":[],
                        "population":[],
                        "dimension":self.problem.number_of_variables,
                        }
        self.verbose = True
        
    def step(self):
        PaperAlgorithm.step(self)

        self.history["best"].append(self.solutions[0].objectives[0])
        self.history["average"].append(np.mean(list(map(lambda x: x.objectives[0], self.solutions))))
        self.history["population"].append(list(map(lambda x: x.objectives[0], self.solutions)))

    def run(self):
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()

        self.solutions = self.evaluate(self.solutions)

        self.init_progress()
        
        if self.verbose:
            #we use StoppingByEvaluations as StoppingByIterations
            for i in tqdm(range(self.termination_criterion.max_evaluations)):
                self.step()
                self.update_progress()
        else:
            for i in range(self.termination_criterion.max_evaluations):
                self.step()
                self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time
        self.history["time"] = self.total_computing_time
        
        