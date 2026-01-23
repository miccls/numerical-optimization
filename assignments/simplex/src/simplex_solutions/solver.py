import logging
from dataclasses import dataclass
from enum import StrEnum

import jaxtyping
import numpy as np

from simplex_solutions import lp_problem, math, pivoting_strategy
from simplex_solutions.numpy_type_aliases import ArrayF, ArrayI

logger = logging.getLogger(__name__)


class SolverStatus(StrEnum):
    SUCCESS = "Success"
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    CYCLING = "Cycling"
    ITERATION_LIMIT = "Iteration limit reached"


@dataclass(frozen=True)
class SolveResult:
    basis: jaxtyping.Int[ArrayI, " m"]
    solution: jaxtyping.Float[ArrayF, " n"]
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
        self.basis_history_: list[tuple[np.integer, ...]] = []
        self.objective_history_: list[float] = []
        self.max_iter_ = max_iterations

    def _is_cycling(
        self,
        current_basis: jaxtyping.Int[ArrayI, " m"],
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

    def find_initial_basis(
        self,
        problem: lp_problem.LpProblem,
    ) -> jaxtyping.Int[ArrayI, " {problem.constraint_matrix.shape[0]}"]:
        """
        Finds a basic feasible solution by solving an auxiliary LP. Throw a SolveFailedError if it fails.

        Need to use the symbolic expression " {problem.constraint_matrix.shape[0]}" instead of " m" in the
        jaxtyping annotation. The constraint_matrix field in the LpProblem dataclass is annotated with "m n",
        but the runtime type checker can't see the annotations inside the dataclass, see
        https://github.com/patrick-kidger/jaxtyping/issues/342.
        """

        # TODO(you): Set up an auxiliary LP whose solution is a basic feasible solution to the original problem
        # See page 378 in the book, for example.
        return np.zeros(problem.constraint_matrix.shape[0], dtype=int)

    def _compute_reduced_costs(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"],
    ) -> jaxtyping.Float[ArrayF, " num_nonbasic"]:
        lagrange_parameters = inv_basis_matrix @ problem.objective[basis]
        return (
            problem.objective[non_basic_vars]
            - problem.constraint_matrix[:, non_basic_vars].T @ lagrange_parameters
        )

    def _finalize_result(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
    ) -> SolveResult:
        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        return SolveResult(
            basis=basis,
            solution=solution,
            objective_value=self.objective_history_[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> tuple[
        SolverStatus,
        SolveResult | None,
    ]:
        if initial_basis is not None:
            basis = np.array(initial_basis)
        else:
            try:
                basis = self.find_initial_basis(problem)
            except SolveFailedError:
                return (SolverStatus.INFEASIBLE, None)

        inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])

        # x_basis is the values of the basic variables
        # TODO(you): set the correct values for x_basis here
        x_basis = inv_basis_matrix @ problem.rhs

        logger.info("Starting simplex algorithm...")

        self.basis_history_ = []  # Used to check for cycling, only degenerate steps are recorded.
        self.objective_history_ = [float(problem.objective[basis] @ x_basis)]

        for iteration in range(1, self.max_iter_):
            non_basic_vars = np.array(
                [i for i in range(problem.num_variables) if i not in set(basis)]
            )

            # Step 1: Compute reduced costs
            # TODO(you): set to the correct reduced costs
            reduced_costs = self._compute_reduced_costs(
                problem, basis, non_basic_vars, inv_basis_matrix
            )
            if np.all(reduced_costs >= 0):
                logger.info(
                    f"Simplex algorithm finished after {iteration - 1} iterations."
                )
                return SolverStatus.SUCCESS, self._finalize_result(
                    problem, basis, x_basis
                )

            # Step 2: Determine the entering variable
            entering_variable: int = self.pivoting_strategy_.pick_entering_index(
                reduced_costs, non_basic_vars
            )

            # d is the "basic direction" of the entering variable
            d = inv_basis_matrix @ problem.constraint_matrix[:, entering_variable]

            if np.all(d <= 0):
                return SolverStatus.UNBOUNDED, None

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
            x_entering = float(x_basis[basic_exiting_index] / d[basic_exiting_index])
            x_basis -= x_entering * d
            x_basis[basic_exiting_index] = x_entering

            self.objective_history_.append(float(problem.objective[basis] @ x_basis))

            logger.info(
                f"Iteration {iteration} ::: "
                f"Entering variable: {entering_variable}, "
                f"Exiting variable: {exiting_variable}, "
                f"Objective: {self.objective_history_[-1]}"
            )

            if self._is_cycling(basis):
                return SolverStatus.CYCLING, None

        logger.info(
            f"Simplex algorithm terminated due to {self.max_iter_} iteration limit"
        )
        return SolverStatus.ITERATION_LIMIT, None
