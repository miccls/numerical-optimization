import logging

import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI

from simplex import pivoting_strategy
from simplex_util import (
    SolveHistory,
    SolveResult,
)

logger = logging.getLogger(__name__)


class DualSimplex:
    pivoting_strategy_: pivoting_strategy.DualPivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.DualPivotingStrategy | None = None,
    ) -> None:
        if pivot_strategy is not None:
            self.pivoting_strategy_ = pivot_strategy
        else:
            self.pivoting_strategy_ = pivoting_strategy.DualBlandsRule()

        self.solve_history_ = SolveHistory()

    @property
    def history(self) -> SolveHistory:
        return self.solve_history_

    def _setup_artificial_problem(
        self, problem: lp_problem.LpProblem, big_m: float = 1e6
    ) -> tuple[lp_problem.LpProblem, jaxtyping.Int[ArrayI, " m+1"]]:
        """
        Sets up the problem for a Phase 1 Dual Simplex by adding a single artificial constraint.
        """
        # TODO(you): Implement the setup for the artificial problem to find a dual-feasible basis.
        m, _ = problem.constraint_matrix.shape
        return problem, np.zeros(m, dtype=int)

    def _finalize_result(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        is_augmented: bool = False,
        original_num_variables: int = 0,
    ) -> SolveResult:
        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        final_basis = basis

        # TODO(you): Handle the augmented problem case if necessary.

        return SolveResult(
            basis=final_basis,
            solution=solution,
            objective_value=self.solve_history_.objective_history[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> SolveResult:
        # TODO(you): Implement the dual simplex algorithm.
        return SolveResult(np.array([]), np.array([]), 0.0)
