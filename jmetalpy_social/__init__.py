from jmetal.core.solution import FloatSolution

def FloatSolutonHash(self: FloatSolution):
    params = self.objectives + self.variables + self.constraints
    return hash(sum(params))

setattr(FloatSolution,"__hash__",FloatSolutonHash)