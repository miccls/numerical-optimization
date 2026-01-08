from typing import Protocol
import numpy as np

from jaxtyping import Float

class PivotingStrategy(Protocol):
    def pick_entering_index(self, reduced_costs: Float[np.ndarray, "non_basic_vars"], non_basic_vars: list[int]) -> int:
        # Used as model
        pass

    def pick_exiting_index(self, distance_to_boundary: Float[np.ndarray, "basic_vars"]) -> int:
        # Used as model
        pass

class SmallestSubscriptRule:
    """
    Implements Bland's Rule:
    When multiple choices are available, always choose the variable 
    with the smallest index (subscript) to prevent cycling.
    """
    
    # Tolerance for floating point comparisons
    EPSILON = 1e-9

    @staticmethod
    def pick_entering_index(
        reduced_costs: Float[np.ndarray, "non_basic_vars"], 
        non_basic_vars: list[int]
    ) -> int:
        """
        Returns the variable ID (subscript) of the entering variable.
        Among all variables with non positive reduced cost, the entering will be the one
        with the smallest subscript.
        """
        candidate_mask = reduced_costs < -SmallestSubscriptRule.EPSILON
        if not np.any(candidate_mask):
            raise RuntimeError("Expects some negative reduced costs.")

        candidates = np.array(non_basic_vars)[candidate_mask]
        return int(np.min(candidates))

    @staticmethod
    def pick_exiting_index(
        ratios: Float[np.ndarray, "constraints"], 
        u: Float[np.ndarray, "constraints"], 
        basic_vars: list[int]
    ) -> int:
        """
        Returns the variable ID of the exiting variable.
        Among the variables which allow the smallest change for the entering variable,
        the one with the smallest subscript will be chosen.
        """
        valid_pivot_mask = u > SmallestSubscriptRule.EPSILON
        
        if not np.any(valid_pivot_mask):
            # If no u > 0, the problem is unbounded
            raise RuntimeError("No valid pivot found (Problem is Unbounded).")

        # Filter to only valid pivots
        valid_ratios = ratios[valid_pivot_mask]
        valid_basic_vars = np.array(basic_vars)[valid_pivot_mask]

        min_ratio = np.min(valid_ratios)
        
        # Find all basic variables that correspond to the minimum ratio
        min_ratio_candidates = valid_basic_vars[np.isclose(valid_ratios, min_ratio)]
        
        # Of those, pick the one with the smallest index
        return int(np.min(min_ratio_candidates))
