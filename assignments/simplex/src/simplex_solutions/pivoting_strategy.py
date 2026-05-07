from abc import ABC, abstractmethod
from typing import override

import jaxtyping
import numpy as np
from common.lp_problem import LpProblem
from common.numpy_type_aliases import ArrayF, ArrayI

from simplex_util import get_non_basic_vars

PIVOTING_TOLERANCE = 1e-4


class PrimalPivotingStrategy(ABC):
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        """
        Gives stateful pivoting strategies a chance to reset for a new LP/basis.
        Stateless rules intentionally leave this as a no-op.
        """
        del problem, basis

    @abstractmethod
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        """
        Selects the variable that should enter the basis.

        Args:
            reduced_costs: reduced costs
            non_basic_vars: variable indices of the non basic variables, assumed to be sorted
        Returns:
            The variable index, i.e. an element of the array `non_basic_vars`, of a variable that should enter the basis.
        """
        ...

    @abstractmethod
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        """
        Selects the index exiting the basis.

        If we compare the arguments with "Numerical Optimization", Nocedal & Wright, page 370,
        we can make the following identifications:

        `x_basis[p]` is the value for the decision variable `x_k`, where `k=basis[p]`.

        `basic_direction` is `d`, with `d = B^-1 * A_q`, where B is the basis matrix `A[:, basis]`,
        and `A_q = A[:, q]` is the column of the constraint matrix for the entering variable `x_q`.

        Args:
            basis: Variable indices for the basic variables.
            x_basis: Values for the basic variables.
            basic_direction: The basic direction for the entering variable.

        Returns:
            Index `p` in ``basis`` array for the decision variable that should be removed from the basis.
        """
        ...


class DualPivotingStrategy(ABC):
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        """
        Gives stateful pivoting strategies a chance to reset for a new LP/basis.
        Stateless rules intentionally leave this as a no-op.
        """
        del problem, basis

    @abstractmethod
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        """
        TODO(martins): Describe purpose of picking entering index
        """
        ...

    @abstractmethod
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        s: jaxtyping.Float[ArrayF, " num_nonbasic"],
        pivot_direction: jaxtyping.Float[ArrayF, " num_nonbasic"],
    ) -> int:
        """
        TODO(martins): Describe purpose of picking exiting index
        """
        ...


def index_of_smallest_ratio(
    basis: jaxtyping.Int[ArrayI, " m"],
    x_basis: jaxtyping.Float[ArrayF, " m"],
    basic_direction: jaxtyping.Float[ArrayF, " m"],
) -> int:
    """
    Args:
        basis: Variable indices for the basic variables.
        x_basis: Values for the basic variables.
        basic_direction: The basic direction for the entering variable.

    Returns:
        Index `i` in `basis` array with the smallest positive ratio `x_basis[i] / basic_direction[i]`,
        choosing the index corresponding to the lowest variable index `basis[i]` in case of ties.
    """

    smallest_ratio_with_smallest_var_index = min(
        (max(0.0, float(x_basis[i])) / basic_direction[i], basis[i], i)
        for i in range(len(x_basis))
        if basic_direction[i] > (PIVOTING_TOLERANCE)
    )

    return smallest_ratio_with_smallest_var_index[2]


class BlandsRule(PrimalPivotingStrategy):
    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        return int(non_basic_vars[reduced_costs < -PIVOTING_TOLERANCE].min())

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        return index_of_smallest_ratio(basis, x_basis, basic_direction)


class DantzigsRule(PrimalPivotingStrategy):
    """Dantzig's rule is one of the simplest pivoting strategies. It was
    suggested by George Dantzig, inventor of the Primal Simplex algorithm.
    It simply selectes the variable with the most negative reduced cost.

    See section 13.5 in "Numerical Optimization" for more details.

    Since all rules use smallest subscript for the exiting index, that is not tested
    """

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        # non_basic_vars are sorted in my implementation
        return int(non_basic_vars[np.argmin(reduced_costs)])

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        return index_of_smallest_ratio(basis, x_basis, basic_direction)


class SteepestEdgeRule(PrimalPivotingStrategy):
    """Primal steepest-edge pricing for the entering variable."""

    def __init__(
        self,
        problem: LpProblem | None = None,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> None:
        self.problem: LpProblem | None = None
        self.entering_variable = -1
        self.non_basic_vars = np.array([], dtype=int)
        self.norm_eta_squared = np.array([], dtype=float)

        if problem is not None and initial_basis is not None:
            self.initialize(problem, initial_basis)

    @override
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        self.problem = problem
        self.entering_variable = -1
        self.non_basic_vars = get_non_basic_vars(problem.num_variables, basis)
        self.iteration = 0

        b_inv = np.linalg.inv(problem.constraint_matrix[:, basis])
        basic_directions = b_inv @ problem.constraint_matrix[:, self.non_basic_vars]
        self.norm_eta_squared = 1.0 + np.sum(
            basic_directions * basic_directions, axis=0
        )

    def _update_eta(
        self,
        exiting_index: int,
        basis: jaxtyping.Int[ArrayI, " m"],
        b_inv: jaxtyping.Float[ArrayF, "m m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> None:
        if self.problem is None:
            raise RuntimeError("SteepestEdgeRule must be initialized before use.")

        entering_position = int(
            np.flatnonzero(self.non_basic_vars == self.entering_variable)[0]
        )
        exiting_variable = int(basis[exiting_index])
        pivot = float(basic_direction[exiting_index])

        keep_mask = self.non_basic_vars != self.entering_variable
        remaining_non_basic_vars = self.non_basic_vars[keep_mask]
        remaining_gamma = self.norm_eta_squared[keep_mask]

        remaining_directions = (
            b_inv @ self.problem.constraint_matrix[:, remaining_non_basic_vars]
        )
        direction_dot_products = basic_direction @ remaining_directions

        # The steepest-edge recurrence is computed in the old basis. The
        # exiting row gives d_j[p] for each old non-basic column, and gamma_q
        # is the old squared norm of the entering edge direction.
        alpha = remaining_directions[exiting_index, :] / pivot
        entering_gamma = self.norm_eta_squared[entering_position]
        updated_remaining_gamma = (
            remaining_gamma
            - 2.0 * alpha * direction_dot_products
            + (alpha * alpha) * entering_gamma
        )

        # After the pivot, the entering variable is basic and the exiting
        # variable is non-basic. Its edge norm follows from B_new^-1 A_exit.
        exiting_gamma = entering_gamma / (pivot * pivot)
        updated_non_basic_vars = np.append(remaining_non_basic_vars, exiting_variable)
        updated_gamma = np.append(updated_remaining_gamma, exiting_gamma)

        order = np.argsort(updated_non_basic_vars)
        self.non_basic_vars = updated_non_basic_vars[order]
        self.norm_eta_squared = np.maximum(updated_gamma[order], PIVOTING_TOLERANCE)

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        if not np.array_equal(non_basic_vars, self.non_basic_vars):
            raise RuntimeError(
                "Steepest-edge weights are not aligned with non-basic variables."
            )

        candidate_mask = reduced_costs < -PIVOTING_TOLERANCE
        candidate_scores = np.full_like(reduced_costs, np.inf, dtype=float)
        candidate_scores[candidate_mask] = reduced_costs[candidate_mask] / np.sqrt(
            self.norm_eta_squared[candidate_mask]
        )
        self.entering_variable = int(non_basic_vars[np.argmin(candidate_scores)])
        return self.entering_variable

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        if inv_basis_matrix is None:
            raise ValueError(
                "SteepestEdgeRule requires the current inverse basis matrix."
            )

        exiting_index = index_of_smallest_ratio(basis, x_basis, basic_direction)
        self._update_eta(exiting_index, basis, inv_basis_matrix, basic_direction)
        return exiting_index


class DualBlandsRule(DualPivotingStrategy):
    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        negative_basic_vars = [
            (variable_index, basis_index, var)
            for (basis_index, var), variable_index in zip(
                enumerate(primal_vars), basic_vars, strict=True
            )
            if var < -PIVOTING_TOLERANCE
        ]
        return min(negative_basic_vars)[1]

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        return index_of_smallest_ratio(non_basic_vars, s, pivot_direction)


class DualDantzigsRule(DualPivotingStrategy):
    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        return int(np.argmin(primal_vars))

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        return index_of_smallest_ratio(non_basic_vars, s, pivot_direction)


class DualSteepestEdgeRule(DualPivotingStrategy):
    """Dual steepest-edge leaving-row rule."""

    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        del basic_vars
        if inv_basis_matrix is None:
            raise ValueError(
                "DualSteepestEdgeRule requires the current inverse basis matrix."
            )

        row_norms_squared = np.sum(inv_basis_matrix * inv_basis_matrix, axis=1)
        candidate_mask = primal_vars < -PIVOTING_TOLERANCE
        candidate_scores = np.full_like(primal_vars, np.inf, dtype=float)
        # Exact dual steepest edge uses ||e_i^T B^-1|| as the edge length for
        # each candidate leaving row. Recomputing keeps the rule aligned with
        # the current basis after inverse refreshes and Phase I augmentation.
        candidate_scores[candidate_mask] = primal_vars[candidate_mask] / np.sqrt(
            np.maximum(row_norms_squared[candidate_mask], PIVOTING_TOLERANCE)
        )
        return int(np.argmin(candidate_scores))

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        return index_of_smallest_ratio(non_basic_vars, s, pivot_direction)
