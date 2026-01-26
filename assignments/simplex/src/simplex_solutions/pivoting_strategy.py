from abc import ABC, abstractmethod
from typing import override

import jaxtyping
import numpy as np

from common.numpy_type_aliases import ArrayF, ArrayI


class PivotingStrategy(ABC):
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
        (x_basis[i] / basic_direction[i], basis[i], i)
        for i in range(len(x_basis))
        if basic_direction[i] > 0
    )

    return smallest_ratio_with_smallest_var_index[2]


class BlandsRule(PivotingStrategy):
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
        return index_of_smallest_ratio(basis, x_basis, basic_direction)


class DantzigsRule(PivotingStrategy):
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
    ) -> int:
        return index_of_smallest_ratio(basis, x_basis, basic_direction)
