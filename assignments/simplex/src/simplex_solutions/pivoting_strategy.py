from abc import ABC, abstractmethod
from typing import override

import jaxtyping

from simplex_solutions.numpy_type_aliases import ArrayF, ArrayI


class PivotingStrategy(ABC):
    @abstractmethod
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int: ...

    @abstractmethod
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int: ...


class SmallestSubscriptRule(PivotingStrategy):
    """
    Implements Bland's rule.
    """

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        return int(non_basic_vars[reduced_costs < 0].min())

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        """
        Selects the exiting variable index based on the Smallest Subscript Rule.

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
            Index in ``basis`` array for the decision variable that should be removed from the basis.
        """

        smallest_ratio_with_smallest_var_index = min(
            (x_basis[i] / basic_direction[i], basis[i], i)
            for i in range(len(x_basis))
            if basic_direction[i] > 0
        )

        return smallest_ratio_with_smallest_var_index[2]
