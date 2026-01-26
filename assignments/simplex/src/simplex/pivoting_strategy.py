from abc import ABC, abstractmethod
from typing import Any, override

import jaxtyping
import numpy as np


class PivotingStrategy(ABC):
    @abstractmethod
    def pick_entering_index(self, *args: Any, **kwargs: Any) -> int: ...  # noqa: ANN401

    @abstractmethod
    def pick_exiting_index(self, *args: Any, **kwargs: Any) -> int: ...  # noqa: ANN401


class SmallestSubscriptRule(PivotingStrategy):
    """
    Implement a pivoting strategy which guarantees finite termination
    of the Simplex algorithm.
    Feel free to change input or output
    """

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[np.ndarray, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[np.ndarray, " num_nonbasic"],
    ) -> int:
        return 0  # TODO(you): Implement and return an appropriate index

    @override
    def pick_exiting_index(self) -> int:
        return 0  # TODO(danielw): What is this supposed to do? The test of this makes no sense
