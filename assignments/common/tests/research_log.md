
# Research log

## Background

In this file I will keep notes on the state of the current solvers implemented and how the are performing on some netlib problems.
I currently have the following solvers:

* `PredictiorCorrector` : An interior point method
* `PrimalSimplex` : The Revised Simplex Algorithm
* `DualSimplex` : The Revised Dual Simplex Algorithm

## 29-03-2026

These are the solve times and iterations on the problems tested in `test_netlib_problems.py`:

1. 

I think I may need to update the tolerances of the dual and primal methods as they reach the solution almost but then fail on some unexpected errors from numpy.
I have changed the production of `non_basic_variables` by looking if the index of a variable is in `basis` instead of `set(basis)`.
It could be beneficial to switch back at some point in the future, but for the problem sizes I have considering at the moment, it seems more efficient to use linear search.
Maybe basis should be a set from the get-go.

Example:
On the "stocfor2" I manage to overcome 41 iterations in 10s using:

```python
non_basic_vars = np.array(
    [i for i in range(problem.num_variables) if i not in basis]
)
```

but only 13 iterations in 10s using:

```python
non_basic_vars = np.array(
    [i for i in range(problem.num_variables) if i not in set(basis)]
)
```
