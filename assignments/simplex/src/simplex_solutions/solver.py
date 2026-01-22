import logging
from dataclasses import dataclass
from enum import StrEnum

import jaxtyping
import numpy as np

from simplex_solutions import lp_problem, math, pivoting_strategy

logger = logging.getLogger(__name__)


class SolverStatus(StrEnum):
    SUCCESS = "Success"
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    CYCLING = "Cycling"


@dataclass(frozen=True)
class SolveResult:
    basis: jaxtyping.Float[np.ndarray, " m"]
    solution: jaxtyping.Float[np.ndarray, " n"]
    objective_value: float


class SolveFailedError(RuntimeError):
    pass


MIN_CYCLE_LEN = 2
OBJECTIVE_IMPROVEMENT_TOL = 1e-9


class Solver:
    def __init__(
        self,
        pivoting_strategy: pivoting_strategy.PivotingStrategy,
        max_iterations: int = 100,
    ) -> None:
        self.pivoting_strategy_ = pivoting_strategy
        self.basis_history_: list[tuple[int, ...]] = []
        self.objective_history_: list[float] = []
        self.max_iter_ = max_iterations

    def _is_cycling(
        self,
        current_basis: jaxtyping.Int[np.ndarray, " m"],
    ) -> bool:
        """Checks for cycling in the simplex algorithm."""
        if len(self.objective_history_) < MIN_CYCLE_LEN:
            return False

        if (
            abs(self.objective_history_[-1] - self.objective_history_[-2])
            > OBJECTIVE_IMPROVEMENT_TOL
        ):
            # If the objective improved, we are not cycling, so we can clear the history.
            self.basis_history_.clear()
            return False

        # A cycle can only occur with degenerate pivots (no change in objective)
        basis_signature = tuple(sorted(current_basis))
        if basis_signature in self.basis_history_:
            return True

        self.basis_history_.append(basis_signature)
        return False

    def find_basic_feasible_solution(
        self,
        problem: lp_problem.LpProblem,
    ) -> jaxtyping.Int[np.ndarray, " n"]:
        """
        Finds a basic feasible solution by solving an auxiliary LP. Throw a SolveFailedError if it fails.
        """
        # TODO(you): Set up an auxiliary LP whose solution is a basic feasible solution to the original problem
        # See page 378 in the book, for example.
        return np.zeros(problem.objective.shape, dtype=int)

    def solve(
        self,
        problem: lp_problem.LpProblem,
        initial_basis: jaxtyping.Int[np.ndarray, " m"] | None = None,
    ) -> tuple[
        SolverStatus,
        SolveResult | None,
    ]:
        if initial_basis is not None:
            basis = np.array(initial_basis)
        else:
            try:
                basis = self.find_basic_feasible_solution(problem)
            except SolveFailedError:
                return (SolverStatus.INFEASIBLE, None)

        inv_basis_matrix: jaxtyping.Float[np.ndarray, "m m"] = np.linalg.inv(
            problem.constraint_matrix[:, basis]
        )
        # x_basis is the values of the basic variables
        # TODO(you): set the correct values for x_basis here
        x_basis: jaxtyping.Float[np.ndarray, " m"] = np.zeros(0)

        logger.info("Starting simplex algorithm...")

        self.basis_history_ = []  # Used to check for cycling, only degenerate steps are recorded.
        self.objective_history_ = [problem.objective[basis] @ x_basis]

        for iteration in range(1, self.max_iter_):
            # Step 1: Compute reduced costs
            # TODO(you): set to the correct reduced costs
            reduced_costs: jaxtyping.Float[np.ndarray, " n-m"] = np.zeros(0)
            if np.all(reduced_costs >= 0):
                break

            # Step 2: Determine the entering variable
            non_basic_vars = np.array(
                i for i in range(problem.num_variables) if i not in set(basis)
            )
            entering_variable: int = self.pivoting_strategy_.pick_entering_index(
                reduced_costs, non_basic_vars
            )

            # d is the "basic direction" of the entering variable
            d = inv_basis_matrix @ problem.constraint_matrix[:, entering_variable]

            if np.all(d <= 0):
                return (SolverStatus.UNBOUNDED, None)

            # Step 3: Determine the exiting variable
            basic_exiting_index = self.pivoting_strategy_.pick_exiting_index(
                basis, x_basis, d
            )
            exiting_variable = basis[basic_exiting_index]

            # Step 4: Update the inverse of the basis matrix (feel free to change input args to update_inverse if desirable)
            basis[basic_exiting_index] = entering_variable
            inv_basis_matrix = math.update_inverse(
                problem.constraint_matrix,
                inv_basis_matrix,
                entering_variable,
                basic_exiting_index,
            )  # <-- TODO in here.

            # Step 5: Update the basic solution from the basic direction
            # TODO(you): ...

            self.objective_history_.append(problem.objective[basis] @ x_basis)

            logger.info(
                f"Iteration {iteration} ::: "
                f"Leaving index: {exiting_variable}, "
                f"Entering index: {entering_variable}, "
                f"Objective: {self.objective_history_[-1]}"
            )

            if self._is_cycling(basis):
                return SolverStatus.CYCLING, None

        logger.info(
            f"Simplex algorithm terminated after {len(self.objective_history_) - 1} iterations."
        )

        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        return SolverStatus.SUCCESS, SolveResult(
            basis=basis,
            solution=solution,
            objective_value=self.objective_history_[-1],
        )
