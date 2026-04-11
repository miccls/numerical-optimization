import logging
import time

import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI

from simplex import linear_algebra, pivoting_strategy
from simplex_util import (
    OPTIMALITY_TOL,
    InfeasibleLpError,
    IterationLimitError,
    SolveFailedError,
    SolveHistory,
    SolveResult,
    UnboundedLpError,
)

logger = logging.getLogger(__name__)


def is_linearly_independent(
    problem: lp_problem.LpProblem,
    inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"],
    entering_var: int,
    exiting_index: int,
) -> bool:
    """Helper function to check if a potential pivot maintains basis nonsingularity."""
    # TODO(you): Implement independence check
    return True


def purge_aux_vars(
    problem: lp_problem.LpProblem,
    basis: jaxtyping.Int[ArrayI, " m"],
    num_variables: int,
) -> jaxtyping.Int[ArrayI, " m"]:
    """Pivots out any auxiliary variables still present in the basis after Phase 1."""
    # TODO(you): Implement the logic to pivot out artificial variables.
    return basis


class PrimalSimplex:
    pivoting_strategy_: pivoting_strategy.PrimalPivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.PrimalPivotingStrategy | None = None,
    ) -> None:
        if pivot_strategy is not None:
            self.pivoting_strategy_ = pivot_strategy
        else:
            self.pivoting_strategy_ = pivoting_strategy.BlandsRule()

        self.solve_history_ = SolveHistory()

    @property
    def history(self) -> SolveHistory:
        return self.solve_history_

    def find_initial_basis(
        self, problem: lp_problem.LpProblem, max_iterations: int = 100
    ) -> jaxtyping.Int[ArrayI, " {problem.constraint_matrix.shape[0]}"]:
        """
        Finds a basic feasible solution by solving an auxiliary LP. Throws a SolveFailedError if it fails.

        Need to use the symbolic expression " {problem.constraint_matrix.shape[0]}" instead of " m" in the
        jaxtyping annotation. The constraint_matrix field in the LpProblem dataclass is annotated with "m n",
        but the runtime type checker can't see the annotations inside the dataclass, see
        https://github.com/patrick-kidger/jaxtyping/issues/342.
        """

        logger.info("Solving auxiliary phase one LP to find a starting basis...")

        # TODO(you): Set up an auxiliary LP whose solution is a basic feasible solution to the original problem
        # See page 378 in the book, for example.

        num_constraints = len(problem.rhs)

        # TODO(you): Construct an appropriate constraint matrix
        phase_one_constraint_matrix = np.zeros(
            (
                problem.constraint_matrix.shape[0],
                problem.constraint_matrix.shape[1] + num_constraints,
            )
        )

        # TODO(you): What is the correct objective function?
        phase_one_objective = np.zeros(problem.num_variables + num_constraints)

        phase_one_problem = lp_problem.LpProblem(
            constraint_matrix=phase_one_constraint_matrix,
            rhs=np.array(problem.rhs),
            objective=phase_one_objective,
        )

        phase_one_solver = PrimalSimplex(pivot_strategy=pivoting_strategy.BlandsRule())
        phase_one_result = phase_one_solver.solve(
            phase_one_problem,
            # TODO(you): What is valid starting basis for the Phase 1 problem?
            initial_basis=np.zeros(num_constraints, dtype=int),
            max_iterations=max_iterations,
        )
        if phase_one_result.objective_value > OPTIMALITY_TOL:
            raise SolveFailedError(
                f"Phase one objective value {phase_one_result.objective_value} is positive: Original problem is infeasible"
            )

        max_print_size = 10
        logger.info(
            f"Found starting basis {phase_one_result.basis if len(phase_one_result.basis) < max_print_size else ''}"
        )

        # TODO(you): The final basis could contain some of the auxiliary variables introduced in Phase 1.
        # Use purge_aux_vars to handle this.
        basis = purge_aux_vars(
            phase_one_problem, phase_one_result.basis, problem.num_variables
        )

        return basis

    def _compute_reduced_costs(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"],
    ) -> jaxtyping.Float[ArrayF, " num_nonbasic"]:
        """Computes the reduced costs for the non-basic variables."""
        # TODO(you): Implement reduced cost calculation
        return np.zeros(len(non_basic_vars))

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
            objective_value=self.solve_history_.objective_history[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> SolveResult:
        self.solve_history_ = SolveHistory()

        if initial_basis is not None:
            basis = np.array(initial_basis)
        else:
            try:
                basis = self.find_initial_basis(problem, max_iterations=max_iterations)
            except SolveFailedError as e:
                raise InfeasibleLpError(
                    "Failed to find an initial simplex basis"
                ) from e

        inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])

        # x_basis is the values of the basic variables
        # TODO(you): set the correct values for x_basis here
        x_basis = inv_basis_matrix @ problem.rhs

        logger.info("Starting simplex algorithm...")
        self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
        logger.info(
            f"Initial objective value {self.solve_history_.objective_history[-1]}"
        )

        logger.info("Iter     Objective      Primal Inf.    Dual Inf.    Time")
        start = time.time()
        for iteration in range(1, max_iterations):
            non_basic_vars = np.array(
                [i for i in range(problem.num_variables) if i not in set(basis)]
            )

            # Step 1: Compute reduced costs
            # TODO(you): set to the correct reduced costs
            reduced_costs = self._compute_reduced_costs(
                problem, basis, non_basic_vars, inv_basis_matrix
            )
            if np.all(reduced_costs >= -pivoting_strategy.PIVOTING_TOLERANCE):
                logger.info(
                    f"Simplex algorithm found optimal objective {self.solve_history_.objective_history[-1]} after {iteration - 1} iterations."
                )
                return self._finalize_result(problem, basis, x_basis)

            # Step 2: Determine the entering variable
            entering_variable: int = self.pivoting_strategy_.pick_entering_index(
                reduced_costs, non_basic_vars
            )

            # d is the "basic direction" of the entering variable
            d = inv_basis_matrix @ problem.constraint_matrix[:, entering_variable]

            if np.all(d <= 0):
                raise UnboundedLpError

            # Step 3: Determine the exiting variable
            basic_exiting_index = self.pivoting_strategy_.pick_exiting_index(
                basis, x_basis, d
            )

            # Step 4: Update the inverse of the basis matrix (feel free to change input args to update_inverse if desirable)
            basis[basic_exiting_index] = entering_variable

            # TODO(you): Update the inverse of the basis matrix using linear_algebra.update_inverse
            inv_basis_matrix = linear_algebra.update_inverse(
                problem.constraint_matrix,
                inv_basis_matrix,
                entering_variable,
                basic_exiting_index,
            )

            # Step 5: Update the basic solution from the basic direction
            # TODO(you): ...
            x_basis = np.zeros(len(basis))

            self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
            logger.info(
                f"{iteration:4d}    {problem.objective[basis].T @ x_basis:10.3e}     "
                f"{np.sum(np.abs(problem.constraint_matrix[:, basis] @ x_basis - problem.rhs)):10.3e}     {max(0.0, np.sum(problem.constraint_matrix.T @ (inv_basis_matrix @ problem.objective[basis]) - problem.objective)):10.3e}"
                f"    {time.time() - start:.4}s"
            )

        logger.info(
            f"Simplex algorithm terminated due to {max_iterations} iteration limit"
        )
        raise IterationLimitError
