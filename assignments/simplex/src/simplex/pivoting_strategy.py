from typing import Protocol
import numpy as np

from jaxtyping import Float

class PivotingStrategy(Protocol):
    def pick_entering_index(self, *args, **kwargs) -> int:
        # Used as model
        pass

    def pick_exiting_index(self, *args, **kwargs) -> int:
        # Used as model
        pass

class MyPivotStrategy:
    """
    Implement a pivoting strategy which guarantees finite termination
    of the Simplex algorithm.
    Feel free to change input or output
    """

    @staticmethod
    def pick_entering_index(
        *args, **kwargs
    ) -> int:
        pass

    @staticmethod
    def pick_exiting_index(
        *args, **kwargs
    ) -> int:
        pass
