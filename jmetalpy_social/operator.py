from jmetal.core.operator import Mutation
from jmetal.algorithm.singleobjective import GeneticAlgorithm
import random
import numpy as np


class ExclusiveMutations(Mutation):
    def __init__(self, mutations, probabilities):
        self.mutations = mutations
        self.probabilities = probabilities

        self.thresholds = [sum(probabilities[:i]) for i in range(len(probabilities)+1)]
        
    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        for mut in self.mutations:
            mut.set_tracked_algorithm(algorithm)
          
    def execute(self,solution):
        rand = random.random()

        for i, mut in enumerate(self.mutations):
            if rand >= self.thresholds[i] and rand < self.thresholds[i+1]:
                solution = mut.execute(solution)
                break

        return solution

    def get_name(self):
        return "ExclusiveMutations"

class IndependantMutations(Mutation):
    def __init__(self, mutations):
        self.mutations = mutations
        
    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        for mut in self.mutations:
            mut.set_tracked_algorithm(algorithm)
          
    def execute(self,solution):
        for mut in self.mutations:
            solution = mut.execute(solution)

        return solution
        
    def get_name(self):
        return "IndependantMutations"


class FollowBestMutation(Mutation):
    def __init__(self,probability: float, tracked_best_count: int):
        Mutation.__init__(self,probability)
        self.tracked_best_count = tracked_best_count
        
    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm
    
    def execute(self,solution):
        
        teacher_idx = random.randint(0,self.tracked_best_count-1)
        for i in range(len(solution.variables)):
            rand = random.random()
            if rand < self.probability:

                solution.variables[i] = self.tracked_algorithm.solutions[teacher_idx].variables[i]

        return solution

    def get_name(self):
        return "FollowBestMutation"

class FollowBestSharedGenesMutation(Mutation):
    def __init__(self,probability: float, tracked_best_count: int, copy_genes_count:int ):
        Mutation.__init__(self,probability)
        self.tracked_best_count = tracked_best_count
        self.copy_genes_count = copy_genes_count
        
    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm
    
    def execute(self,solution):
        rand = random.random()

        if rand < self.probability:
        
            teacher_idx = random.randint(0,self.tracked_best_count-1)

            genes_mat = np.array([sol.variables for sol in self.tracked_algorithm.solutions[:self.tracked_best_count]])
            genes_mat = np.transpose(genes_mat)

            sharing_rating = -np.std(genes_mat,axis=-1)
            # sharing_rating = sharing_rating/np.max(sharing_rating)
            # sharing_rating = 1-sharing_rating
            sharing_rating = np.exp(sharing_rating)/np.sum(np.exp(sharing_rating))
            
            chosen_genes = np.random.choice(np.arange(0,sharing_rating.shape[0]), p=sharing_rating, size=(self.copy_genes_count,))

            # print(sharing_rating)
            # print(np.argsort(sharing_rating))

            # chosen_genes = np.argsort(sharing_rating)[0:self.copy_genes_count]

            for i in chosen_genes:
                solution.variables[i] = self.tracked_algorithm.solutions[teacher_idx].variables[i]

        return solution

    def get_name(self):
        return "FollowBestSharedGenesMutation"

class FollowBestMutationSingleGene(Mutation):
    def __init__(self,probability: float, tracked_best_count: int):
        Mutation.__init__(self,probability)
        self.tracked_best_count = tracked_best_count
        
    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm
    
    def execute(self,solution):
        rand = random.random()

        if rand < self.probability:
            teacher_idx = random.randint(0,self.tracked_best_count-1)
            gene_swap_idx = random.randint(0,len(solution.variables)-1)

            solution.variables[gene_swap_idx] = self.tracked_algorithm.solutions[teacher_idx].variables[gene_swap_idx]

        return solution

    def get_name(self):
        return "FollowBestMutationSingleGene"

