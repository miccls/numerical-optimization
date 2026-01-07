import unittest
import time 
import numpy as np

from simplex import pivoting_strategy, math, lp_problem, solver

class TestPivoting(unittest.TestCase):

    def setUp(self):
        self.smallest_subscript_rule = pivoting_strategy.SmallestSubscriptRule()

    def test_smallest_subscript_entering(self):
        reduced_costs = np.array([1, 2, 3, 4, -1, -2, -3])
        non_basic_vars = [3,4,5,6,0,1,2]
        self.assertEqual(self.smallest_subscript_rule.pick_entering_index(reduced_costs, non_basic_vars), 0) 

    def test_smallest_subscript_exiting(self):
        x = np.array([1,2,3,4,5,6])
        u = np.array([1,-4,1,8,1,12])
        basic_vars = [10,2,4,6,7,20]
        ratios = np.array([xi / ui for xi, ui in zip(x, u)])
        self.assertEqual(self.smallest_subscript_rule.pick_exiting_index(ratios, u, basic_vars), 6) 
        
class TestInverseComputation(unittest.TestCase):
    
    def test_update_inverse(self):
        A = np.array([
             [1,2,3,4],
             [4,3,2,1],
            ])
        basis = [1, 2]
        basis_matrix = A[:, basis]
        self.assertTrue((basis_matrix == [[2,3],[3,2]]).all())
        Binv = np.linalg.inv(basis_matrix)
        
        # Switch out index 2 for index three
        entering_variable = 3
        exiting_variable = 2
        new_basis = [1,3]
        
        Binv = math.update_inverse(A, Binv, entering_variable, basis.index(exiting_variable))
        
        self.assertTrue(np.allclose(Binv, np.linalg.inv(A[:, new_basis])))

    def test_update_speed(self):
        
        A = np.concatenate((np.eye(500), np.random.rand(500,1)), axis=1)
        basis = list(range(500))
        entering_variable = 500
        exiting_variable = 499
        new_basis = basis[:-1] + [500]

        Binv = np.eye(500)
        d = A[:, entering_variable]

        start_time = time.perf_counter()
        Binv = math.update_inverse(A, Binv, entering_variable, basis.index(exiting_variable))
        end_time = time.perf_counter()
        
        time_in_ms = (end_time - start_time) * 1000
        self.assertLess(time_in_ms, 10, msg = f"Should take less than 10 ms, took {time_in_ms}")
        self.assertTrue(np.allclose(Binv @ A[:, new_basis], np.eye(500)))
        
        
class TestSolver(unittest.TestCase):
    
    def test_problem_with_start_solution(self):
        # Example 3.5 in "Introduction to Linear Programming", page 101.    
        A = np.array([
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1],
        ])
        
        b = np.array([20, 20, 20]).T
        
        c = np.array([-10, -12, -12, 0, 0, 0])
        
        # Starting basis.
        B = [3,4,5]
        
        Binv = np.linalg.inv(A[:, B])
        x = Binv @ b
        
        # Assert feasibility of starting point.
        self.assertTrue(np.allclose(A[:, B] @ x, b))
        self.assertTrue((x >= 0).all())
        
        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        
        test_problem = lp_problem.LpProblem(A, b, c)
        (status, (basis, solution, objective_value)) = simplex_solver.solve(test_problem, B = B)
        
        # Check we have solved the problem correctly!
        self.assertTrue(status == solver.SolverStatus.SUCCESS)
        self.assertTrue(all(i in [0,1,2] for i in basis))
        self.assertTrue(np.allclose(solution, [4,4,4,0,0,0]))
        self.assertEqual(-136, objective_value)

    def test_problem_without_start_solution(self):
        # Example 3.5 in "Introduction to Linear Programming", page 101.    
        A = np.array([
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1],
        ])
        
        b = np.array([20, 20, 20]).T
        
        c = np.array([-10, -12, -12, 0, 0, 0])
        
        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        
        test_problem = lp_problem.LpProblem(A, b, c)
        (status, (basis, solution, objective_value)) = simplex_solver.solve(test_problem)
        
        # Check we have solved the problem correctly!
        self.assertTrue(status, f"Expected status to be SUCCESS, but got {status}")
        self.assertTrue(all(i in [0,1,2] for i in basis))
        self.assertTrue(np.allclose(solution, [4,4,4,0,0,0]))
        self.assertAlmostEqual(-136, objective_value)
        
    def test_chvatals_example_for_cycling(self):
        # See for ref: https://www.matem.unam.mx/~omar/math340/degenerate.html or
        # https://sites.math.washington.edu/~vinzant/teaching/407/Chvatal.pdf, page 31.
        # This is an example of an LP which cycles under some pivot rules.
        A = np.array([
            [-0.5, 5.5, 2.5, -9, -1, 0, 0],
            [-0.5, 1.5, 0.5, -1, 0, -1, 0],
            [1, 0, 0, 0, 0, 0, 1],
        ])

        b = np.array([0, 0, 1]).T

        c = np.array([-10, 57, 9, 24, 0, 0, 0])

        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        status, _ = simplex_solver.solve(lp_problem.LpProblem(A,b,c), log = True)

        self.assertTrue(status == solver.SolverStatus.SUCCESS, f"Expected status to be SUCCESS, but got {status}")


if __name__ == "__main__":
    unittest.main()