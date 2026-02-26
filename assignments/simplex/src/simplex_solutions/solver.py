import logging
from dataclasses import dataclass

import jaxtyping
import numpy as np

from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI
from simplex_solutions import linear_algebra, pivoting_strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveResult:
    basis: jaxtyping.Int[ArrayI, " m"]
    solution: jaxtyping.Float[ArrayF, " n"]
    objective_value: float


class SolveFailedError(RuntimeError):
    pass


class InfeasibleLpError(SolveFailedError):
    pass


class UnboundedLpError(SolveFailedError):
    pass


class SimplexCyclingError(SolveFailedError):
    pass


class IterationLimitError(SolveFailedError):
    pass


MIN_CYCLE_LEN = 2
OBJECTIVE_IMPROVEMENT_TOL = 1e-9
OPTIMALITY_TOL = 1e-9


class SolveHistory:
    def __init__(self) -> None:
        self.basis_: list[tuple[int, ...]] = []
        self.objective_: list[float] = []
        self.basis_cycle_: set[tuple[int, ...]] = set()

    @property
    def basis_history(self) -> list[tuple[int, ...]]:
        return self.basis_

    @property
    def objective_history(self) -> list[float]:
        return self.objective_

    def update(
        self, basis: jaxtyping.Int[ArrayI, " m"], objective_value: float
    ) -> None:
        self.objective_.append(objective_value)

        if (
            len(self.objective_) > 1
            and abs(self.objective_[-1] - self.objective_[-2])
            > OBJECTIVE_IMPROVEMENT_TOL
        ):
            # If the objective improved, we are not cycling, so we can clear the history.
            self.basis_cycle_.clear()

        basis_signature = tuple(int(b) for b in sorted(basis))
        self.basis_.append(basis_signature)

        if basis_signature in self.basis_cycle_:
            raise SimplexCyclingError(
                f"Basis cycle of length {len(self.basis_cycle_)} detected"
            )
        self.basis_cycle_.add(basis_signature)


class Solver:
    pivoting_strategy_: pivoting_strategy.PivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.PivotingStrategy | None = None,
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
        self,
        problem: lp_problem.LpProblem,
    ) -> jaxtyping.Int[ArrayI, " {problem.constraint_matrix.shape[0]}"]:
        """
        Finds a basic feasible solution by solving an auxiliary LP. Throws a SolveFailedError if it fails.

        Need to use the symbolic expression " {problem.constraint_matrix.shape[0]}" instead of " m" in the
        jaxtyping annotation. The constraint_matrix field in the LpProblem dataclass is annotated with "m n",
        but the runtime type checker can't see the annotations inside the dataclass, see
        https://github.com/patrick-kidger/jaxtyping/issues/342.
        """

        logger.info("Solving auxiliary phase one LP to find a starting basis...")

        e_matrix = np.diag(np.array([1.0 if b >= 0 else -1.0 for b in problem.rhs]))
        phase_one_problem = lp_problem.LpProblem(
            constraint_matrix=np.concatenate(
                [problem.constraint_matrix, e_matrix], axis=1
            ),
            rhs=np.array(problem.rhs),
            objective=np.concatenate(
                [np.zeros(problem.num_variables), np.ones(len(problem.rhs))]
            ),
        )
        # Use the smallest subscript rule to hopefully basis containing the original variables
        phase_one_solver = Solver(pivot_strategy=pivoting_strategy.BlandsRule())
        phase_one_result = phase_one_solver.solve(
            phase_one_problem,
            initial_basis=np.array(
                range(problem.num_variables, phase_one_problem.num_variables)
            ),
        )
        if phase_one_result.objective_value > OPTIMALITY_TOL:
            raise SolveFailedError(
                f"Phase one objective value {phase_one_result.objective_value} is positive: Original problem is infeasible"
            )

        max_print_size = 10
        logger.info(
            f"Found starting basis {phase_one_result.basis if len(phase_one_result.basis) < max_print_size else ''}"
        )

        # TODO(danielw): How to do Phase 2 in the book? Even with the smallest subscript rule i think we can have
        # some of the auxiliary variables in the optimal phase one basis.
        # For example what happens if the original LP is Ax = 0?

        return phase_one_result.basis

        # TODO(you): Set up an auxiliary LP whose solution is a basic feasible solution to the original problem
        # See page 378 in the book, for example.
        # return np.zeros(problem.constraint_matrix.shape[0], dtype=int)

    def _compute_reduced_costs(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"],
    ) -> jaxtyping.Float[ArrayF, " num_nonbasic"]:
        n_matrix = problem.constraint_matrix[:, non_basic_vars]

        return problem.objective[non_basic_vars] - n_matrix.T @ (
            inv_basis_matrix.T @ problem.objective[basis]
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
                basis = self.find_initial_basis(problem)
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

        for iteration in range(1, max_iterations):
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
            exiting_variable = basis[basic_exiting_index]

            # Step 4: Update the inverse of the basis matrix (feel free to change input args to update_inverse if desirable)
            basis[basic_exiting_index] = entering_variable
            inv_basis_matrix = linear_algebra.update_inverse(
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

            self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
            logger.info(
                f"Iteration {iteration} ::: "
                f"Entering variable: {entering_variable}, "
                f"Exiting variable: {exiting_variable}, "
                f"Objective: {self.solve_history_.objective_history[-1]}"
            )

        logger.info(
            f"Simplex algorithm terminated due to {max_iterations} iteration limit"
        )
        raise IterationLimitError
