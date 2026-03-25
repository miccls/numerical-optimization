import logging
from dataclasses import dataclass

import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI

from simplex_solutions import linear_algebra, pivoting_strategy
from simplex_util import (
    SolveHistory,
)

logger = logging.getLogger(__name__)

# Solve with primal simplex
# N^T \lambda + s = c
# Add q
# N^T \lambda + s + Eq = c
# with E_ii = -1 if c<0 else 1
# Then solve min q subject to that constraint
# with initial basis all of the q's and q = |c|.

@dataclass
class DualVariables:
    s: jaxtyping.Float[ArrayF, " n"]
    lam: jaxtyping.Float[ArrayF, " m"]


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

    def compute_dual_basic_feasible(
        self, problem: lp_problem.LpProblem, basis: jaxtyping.Int[ArrayI, " m"]
    ) -> DualVariables:
        # lam_basis is the values of the basic dual variables
        lam_basis = np.linalg.solve(
            problem.constraint_matrix[:, basis].T, problem.objective[basis]
        )

        # Reduced costs (if it had been a primal problem)
        s = np.zeros(len(problem.objective), dtype=float)
        non_basic_vars = [v for v in range(problem.num_variables) if v not in basis]
        s[non_basic_vars] = problem.objective[non_basic_vars] - (problem.constraint_matrix[:, non_basic_vars].T @ lam_basis)
        return DualVariables(s=s, lam=lam_basis)

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> None:

        if initial_basis is None:
            initial_basis = np.zeros(problem.constraint_matrix.shape[0], dtype=int)

        basis = initial_basis
        inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])
        primal_vars = inv_basis_matrix @ problem.rhs
        lam = inv_basis_matrix.T @ problem.objective[basis]
        s = problem.objective - problem.constraint_matrix.T @ lam
        dual_variables = DualVariables(s=s, lam=lam)

        iteration = 0
        while iteration < max_iterations:
            non_basic_vars = np.array([v for v in range(problem.num_variables) if v not in basis])
            exiting_index = self.pivoting_strategy_.pick_exiting_index(primal_vars, basis)
            w = problem.constraint_matrix[:, non_basic_vars].T @ ((inv_basis_matrix.T)[:, exiting_index])
            entering_index = self.pivoting_strategy_.pick_entering_index(non_basic_vars, dual_variables.s, w)
            alpha = dual_variables.s[exiting_index] / w[exiting_index]

            # Update vars
            v = inv_basis_matrix.T[exiting_index]
            dual_variables.s[non_basic_vars] -= alpha * w
            dual_variables.s[exiting_index] = alpha
            dual_variables.s[basis] = 0
            dual_variables.lam += alpha * v

            basic_direction = inv_basis_matrix @ problem.constraint_matrix[:, entering_index]
            gamma = primal_vars[exiting_index] / basic_direction[exiting_index]
            primal_vars -= gamma * basic_direction
            primal_vars[exiting_index] = gamma

            # Update basis
            basis[basis == exiting_index] = entering_index

            # Update inverse
            inv_basis_matrix = linear_algebra.update_inverse(problem.constraint_matrix, inv_basis_matrix, entering_index, exiting_index)

            iteration += 1
