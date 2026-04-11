from abc import ABC, abstractmethod
from typing import override

import jaxtyping
from common.numpy_type_aliases import ArrayF, ArrayI

PIVOTING_TOLERANCE = 1e-6


class PrimalPivotingStrategy(ABC):
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
    @abstractmethod
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
    ) -> int:
        """
        Selects the variable that should exit the basis in the dual simplex method.
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
        Selects the variable that should enter the basis in the dual simplex method.
        """
        ...


class BlandsRule(PrimalPivotingStrategy):
    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        # TODO(you): Pick entering index according to Bland's rule.
        return -1

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        # TODO(you): Pick exiting index according to Bland's rule.
        return -1


class DantzigsRule(PrimalPivotingStrategy):
    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        # TODO(you): Pick entering index according to Dantzig's rule.
        return -1

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        # TODO(you): Pick exiting index according to Dantzig's rule.
        return -1


class DualBlandsRule(DualPivotingStrategy):
    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
    ) -> int:
        # TODO(you): Pick exiting index according to Bland's rule for the dual simplex.
        return -1

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        s: jaxtyping.Float[ArrayF, " num_nonbasic"],
        pivot_direction: jaxtyping.Float[ArrayF, " num_nonbasic"],
    ) -> int:
        # TODO(you): Pick entering index according to Bland's rule for the dual simplex.
        return -1
