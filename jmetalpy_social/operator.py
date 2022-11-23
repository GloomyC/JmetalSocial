import random

import numpy as np
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.operator import Mutation


def my_sign(x: float) -> float:
    return x / (abs(x) + 1e-6)


class ExclusiveMutations(Mutation):
    def __init__(self, mutations, probabilities):
        self.mutations = mutations
        self.probabilities = probabilities

        self.thresholds = [
            sum(probabilities[:i]) for i in range(len(probabilities) + 1)
        ]

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        for mut in self.mutations:
            mut.set_tracked_algorithm(algorithm)

    def execute(self, solution):
        rand = random.random()

        for i, mut in enumerate(self.mutations):
            if rand >= self.thresholds[i] and rand < self.thresholds[i + 1]:
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

    def execute(self, solution):
        for mut in self.mutations:
            solution = mut.execute(solution)

        return solution

    def get_name(self):
        return "IndependantMutations"


class FollowBestMutation(Mutation):
    def __init__(self, probability: float, tracked_best_count: int, follow_rate: float):
        Mutation.__init__(self, probability)
        self.tracked_best_count = tracked_best_count
        self.follow_rate = follow_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):

        teacher_idx = random.randint(0, self.tracked_best_count - 1)
        for i in range(len(solution.variables)):
            rand = random.random()
            if rand < self.probability:
                diff = (
                    self.tracked_algorithm.solutions[teacher_idx].variables[i]
                    - solution.variables[i]
                )

                solution.variables[i] += self.follow_rate * diff

        return solution

    def get_name(self):
        return "FollowBestMutation"


class FollowBestSharedGenesMutation(Mutation):
    def __init__(
        self,
        probability: float,
        tracked_best_count: int,
        copy_genes_count: int,
        follow_rate: float,
    ):
        Mutation.__init__(self, probability)
        self.tracked_best_count = tracked_best_count
        self.copy_genes_count = copy_genes_count
        self.follow_rate = follow_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        rand = random.random()

        if rand < self.probability:

            teacher_idx = random.randint(0, self.tracked_best_count - 1)

            genes_mat = np.array(
                [
                    sol.variables
                    for sol in self.tracked_algorithm.solutions[
                        : self.tracked_best_count
                    ]
                ]
            )
            genes_mat = np.transpose(genes_mat)

            sharing_rating = -np.std(genes_mat, axis=-1)
            sharing_rating = np.exp(sharing_rating) / np.sum(np.exp(sharing_rating))

            chosen_genes = np.random.choice(
                np.arange(0, sharing_rating.shape[0]),
                p=sharing_rating,
                size=(self.copy_genes_count,),
            )

            for g in chosen_genes:
                diff = (
                    self.tracked_algorithm.solutions[teacher_idx].variables[g]
                    - solution.variables[g]
                )

                solution.variables[g] += self.follow_rate * diff

        return solution

    def get_name(self):
        return "FollowBestSharedGenesMutation"


class FollowBestDistinctGenesMutation(Mutation):
    def __init__(
        self,
        probability: float,
        tracked_best_count: int,
        copy_genes_count: int,
        follow_rate: float,
    ):
        Mutation.__init__(self, probability)
        self.tracked_best_count = tracked_best_count
        self.copy_genes_count = copy_genes_count
        self.follow_rate = follow_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        rand = random.random()

        if rand < self.probability:

            teacher_idx = random.randint(0, self.tracked_best_count - 1)

            genes_mat = np.array(
                [
                    sol.variables
                    for sol in self.tracked_algorithm.solutions[
                        : self.tracked_best_count
                    ]
                ]
            )
            genes_mat = np.transpose(genes_mat)

            sharing_rating = np.std(genes_mat, axis=-1)
            sharing_rating = np.exp(sharing_rating) / np.sum(np.exp(sharing_rating))

            chosen_genes = np.random.choice(
                np.arange(0, sharing_rating.shape[0]),
                p=sharing_rating,
                size=(self.copy_genes_count,),
            )

            for g in chosen_genes:
                diff = (
                    self.tracked_algorithm.solutions[teacher_idx].variables[g]
                    - solution.variables[g]
                )

                solution.variables[g] += self.follow_rate * diff

        return solution

    def get_name(self):
        return "FollowBestDistinctGenesMutation"


class RepelWorstMutation(Mutation):
    def __init__(self, probability: float, tracked_worst_count: int, repel_rate: float):
        super().__init__(probability)
        self.tracked_worst_count = tracked_worst_count
        self.repel_rate = repel_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        n = len(self.tracked_algorithm.solutions)
        to_repel_idx = n - random.randint(1, self.tracked_worst_count)
        if random.random() < self.probability:
            for i in range(len(solution.variables)):
                diff = (
                    solution.variables[i]
                    - self.tracked_algorithm.solutions[to_repel_idx].variables[i]
                )
                solution.variables[i] += -self.repel_rate * (
                    diff / self.tracked_algorithm.problem.lower_bound[i]
                    - diff / abs(diff + 1e-6)
                )
                solution.variables[i] = min(
                    max(
                        solution.variables[i],
                        self.tracked_algorithm.problem.lower_bound[i],
                    ),
                    self.tracked_algorithm.problem.upper_bound[i],
                )
        return solution

    def get_name(self) -> str:
        return "RepelWorstMutation"


class RepelWorstMutationSparse(Mutation):
    def __init__(self, probability: float, tracked_worst_count: int, repel_rate: float):
        super().__init__(probability)
        self.tracked_worst_count = tracked_worst_count
        self.repel_rate = repel_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        n = len(self.tracked_algorithm.solutions)
        to_repel_idx = n - random.randint(1, self.tracked_worst_count)
        for i in range(len(solution.variables)):
            if random.random() < self.probability:
                diff = (
                    solution.variables[i]
                    - self.tracked_algorithm.solutions[to_repel_idx].variables[i]
                )
                solution.variables[i] += -self.repel_rate * (
                    diff / self.tracked_algorithm.problem.lower_bound[i]
                    - diff / abs(diff + 1e-6)
                )
                solution.variables[i] = min(
                    max(
                        solution.variables[i],
                        self.tracked_algorithm.problem.lower_bound[i],
                    ),
                    self.tracked_algorithm.problem.upper_bound[i],
                )
        return solution

    def get_name(self) -> str:
        return "RepelWorstMutationSparse"


class RepelWorstMutationNorm(Mutation):
    def __init__(
        self,
        probability: float,
        tracked_worst_count: int,
        norm_fn: callable,
        repel_rate: float,
    ):
        super().__init__(probability)
        self.tracked_worst_count = tracked_worst_count
        self.norm_fn = norm_fn
        self.repel_rate = repel_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        n = len(self.tracked_algorithm.solutions)
        to_repel_idx = n - random.randint(1, self.tracked_worst_count)
        if random.random() < self.probability:
            for i in range(len(solution.variables)):
                diff = (
                    solution.variables[i]
                    - self.tracked_algorithm.solutions[to_repel_idx].variables[i]
                )
                solution.variables[i] += self.repel_rate * float(self.norm_fn(diff))
                solution.variables[i] = min(
                    max(
                        solution.variables[i],
                        self.tracked_algorithm.problem.lower_bound[i],
                    ),
                    self.tracked_algorithm.problem.upper_bound[i],
                )
        return solution

    def get_name(self) -> str:
        return "RepelWorstMutationNorm"


class RepelWorstMutationMean(Mutation):
    def __init__(
        self,
        probability: float,
        tracked_worst_count: int,
        repel_rate: float,
    ):
        super().__init__(probability)
        self.tracked_worst_count = tracked_worst_count
        self.repel_rate = repel_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        n = len(self.tracked_algorithm.solutions)
        if random.random() < self.probability:
            for i in range(len(solution.variables)):
                val = solution.variables[i]
                fn = lambda x: -1 * (
                    x / self.tracked_algorithm.problem.lower_bound[i]
                    - x / abs(x + 1e-6)
                )
                diff = (
                    sum(
                        fn(val - self.tracked_algorithm.solutions[n - j].variables[i])
                        for j in range(1, self.tracked_worst_count)
                    )
                    / self.tracked_worst_count
                )
                solution.variables[i] += self.repel_rate * diff
                solution.variables[i] = min(
                    max(
                        solution.variables[i],
                        self.tracked_algorithm.problem.lower_bound[i],
                    ),
                    self.tracked_algorithm.problem.upper_bound[i],
                )
        return solution

    def get_name(self) -> str:
        return "RepelWorstMutationMean"


class RepelWorstMutationGravity(RepelWorstMutationNorm):
    def __init__(self, probability: float, tracked_worst_count: int, repel_rate: float):
        gravity = lambda x: x / abs(x + 1e-6) * (abs(x) + 1) ** -2
        super().__init__(
            probability=probability,
            tracked_worst_count=tracked_worst_count,
            norm_fn=gravity,
            repel_rate=repel_rate,
        )

    def get_name(self) -> str:
        return "RepelWorstMutationGravity"


class RepelWorstMutationGravityMutlistep(Mutation):
    def __init__(
        self,
        probability: float,
        tracked_worst_count: int,
        repel_rate: float,
    ):
        super().__init__(probability)
        self.tracked_worst_count = tracked_worst_count
        self.repel_rate = repel_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        n = len(self.tracked_algorithm.solutions)
        if random.random() < self.probability:
            for i in range(len(solution.variables)):
                gravity = lambda x: x / abs(x + 1e-6) * (abs(x) + 1) ** -2
                mean_val = (
                    sum(
                        self.tracked_algorithm.solutions[n - j].variables[i]
                        for j in range(1, self.tracked_worst_count)
                    )
                    # / self.tracked_worst_count
                )
                diff = solution.variables[i] - mean_val
                solution.variables[i] += self.repel_rate * gravity(diff)
                solution.variables[i] = min(
                    max(
                        solution.variables[i],
                        self.tracked_algorithm.problem.lower_bound[i],
                    ),
                    self.tracked_algorithm.problem.upper_bound[i],
                )
        return solution

    def get_name(self) -> str:
        return "RepelWorstMutationGravityMutlistep"


class RepelWorstMutationGravityShared(Mutation):
    def __init__(
        self,
        probability: float,
        tracked_worst_count: int,
        repel_rate: float,
    ):
        super().__init__(probability)
        self.tracked_worst_count = tracked_worst_count
        self.repel_rate = repel_rate

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):
        n = len(self.tracked_algorithm.solutions)
        if random.random() < self.probability:
            for i in range(len(solution.variables)):
                gravity = lambda x: x / abs(x + 1e-6) * (abs(x) + 1) ** -2
                mean_val = (
                    sum(
                        self.tracked_algorithm.solutions[n - j].variables[i]
                        for j in range(1, self.tracked_worst_count)
                    )
                    # / self.tracked_worst_count
                )
                diff = solution.variables[i] - mean_val
                solution.variables[i] += self.repel_rate * gravity(diff)
                solution.variables[i] = min(
                    max(
                        solution.variables[i],
                        self.tracked_algorithm.problem.lower_bound[i],
                    ),
                    self.tracked_algorithm.problem.upper_bound[i],
                )
        return solution

    def get_name(self) -> str:
        return "NOTE IMPLEMENTED"


class MagneticMutation(Mutation):
    def __init__(
        self,
        probability: float,
        margin_count: int,
        effect_rate: float,
    ):
        super().__init__(probability)
        self.effect_rate = effect_rate
        self.margin_count = margin_count

    def set_tracked_algorithm(self, algorithm: GeneticAlgorithm):
        self.tracked_algorithm = algorithm

    def execute(self, solution):

        if random.random() >= self.probability:
            return solution

        n = len(self.tracked_algorithm.solutions)
        # indices = list(range(self.margin_count)) + list(range(n - self.margin_count, n))
        indices = list(range(n - self.margin_count, n))
        genes_mat = np.array(
            [self.tracked_algorithm.solutions[i].variables for i in indices]
        )

        this_genes = np.array(solution.variables)
        distances = np.linalg.norm(genes_mat - this_genes, axis=1)

        idx = np.random.choice(
            distances.shape[0],
            p=(1 / (distances + 1e-2)) / (1 / (distances + 1e-2)).sum(),
        )
        other_idx = indices[idx]

        other_genes = self.tracked_algorithm.solutions[other_idx].variables
        dist = float(distances[idx])
        k = my_sign(x=dist) / (abs(dist) + 1) ** 3

        max_obj_diff = (
            self.tracked_algorithm.solutions[-1].objectives[0]
            - self.tracked_algorithm.solutions[0].objectives[0]
        )
        force = (
            -(
                solution.objectives[0]
                - self.tracked_algorithm.solutions[other_idx].objectives[0]
            )
            / max_obj_diff
        )

        for i in range(len(solution.variables)):
            diff_i = solution.variables[i] - other_genes[i]
            delta_i = k * diff_i
            solution.variables[i] += force * self.effect_rate * delta_i
            solution.variables[i] = min(
                max(
                    solution.variables[i],
                    self.tracked_algorithm.problem.lower_bound[i],
                ),
                self.tracked_algorithm.problem.upper_bound[i],
            )

        return solution

    def get_name(self) -> str:
        return "MagneticFieldMutation"
