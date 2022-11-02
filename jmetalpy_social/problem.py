from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import numpy as np


def AckleyFn(x: np.ndarray):
    dim = x.shape[-1]
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x * x, axis=-1) / dim))
        - np.exp(np.sum(np.cos(2 * np.pi * x) / dim, axis=-1))
        + np.e
        + 20
    )


def DeJongFn(x: np.ndarray):
    return np.sum(np.power(x, 2), axis=-1)


def RastriginFn(x: np.ndarray):
    dim = x.shape[-1]
    return 10 * dim + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x), axis=-1)


def GriewankFn(x: np.ndarray):
    dim = x.shape[-1]
    i_s = np.arange(1, dim + 1)
    return (
        np.sum(np.power(x, 2) / 4000, axis=-1) - np.prod(np.cos(x / i_s), axis=-1) + 1
    )


class AckleyProblem(FloatProblem):
    def __init__(
        self,
        number_of_variables: int,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
    ):
        super(AckleyProblem, self).__init__()

        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = number_of_variables

        self.lower_bound = [lower_bound for i in range(number_of_variables)]
        self.upper_bound = [upper_bound for i in range(number_of_variables)]

        self.obj_labels = ["Ack(x)"]

        self.obj_directions = [self.MINIMIZE]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        solution.objectives[0] = AckleyFn(np.array(solution.variables))

        return solution

    def get_name(self) -> str:
        return "AckleyProblem"


class DeJongProblem(FloatProblem):
    def __init__(
        self,
        number_of_variables: int,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
    ):

        super(DeJongProblem, self).__init__()

        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = number_of_variables

        self.lower_bound = [lower_bound for i in range(number_of_variables)]
        self.upper_bound = [upper_bound for i in range(number_of_variables)]

        self.obj_labels = ["DeJ(x)"]

        self.obj_directions = [self.MINIMIZE]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = DeJongFn(np.array(solution.variables))
        return solution

    def get_name(self) -> str:
        return "DeJongProblem"


class RastriginProblem(FloatProblem):
    def __init__(
        self,
        number_of_variables: int,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
    ):

        super(RastriginProblem, self).__init__()

        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = number_of_variables

        self.lower_bound = [lower_bound for i in range(number_of_variables)]
        self.upper_bound = [upper_bound for i in range(number_of_variables)]

        self.obj_labels = ["Rastr(x)"]

        self.obj_directions = [self.MINIMIZE]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = RastriginFn(np.array(solution.variables))
        return solution

    def get_name(self) -> str:
        return "RastriginProblem"


class GriewankProblem(FloatProblem):
    def __init__(
        self,
        number_of_variables: int,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
    ):

        super(GriewankProblem, self).__init__()

        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = number_of_variables

        self.lower_bound = [lower_bound for i in range(number_of_variables)]
        self.upper_bound = [upper_bound for i in range(number_of_variables)]

        self.obj_labels = ["Griew(x)"]

        self.obj_directions = [self.MINIMIZE]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        solution.objectives[0] = GriewankFn(np.array(solution.variables))

        return solution

    def get_name(self) -> str:
        return "GriewankProblem"
