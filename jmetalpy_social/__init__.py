from jmetal.core.solution import FloatSolution
from jmetal.core.operator import Mutation
from jmetal.algorithm.singleobjective import GeneticAlgorithm

def FloatSolutonHash(self: FloatSolution):
    params = self.objectives + self.variables + self.constraints
    return hash(sum(params))

def MutationSetTrackedAlgorithm(self: Mutation, tracked_algorithm: GeneticAlgorithm):
    pass

setattr(FloatSolution,"__hash__",FloatSolutonHash)
setattr(Mutation,"set_tracked_algorithm",MutationSetTrackedAlgorithm)