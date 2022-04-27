# -*- coding: utf-8 -*-
import regcma
import numpy as np

# Define the objective function to be minimized.


def quadratic(x): return x.dot(x)


# Define the initial solution.
DIMENSION = 100
x0 = np.random.randn(DIMENSION)

result = regcma.solve(quadratic, x0, {'iteration_max': 10000})
print(result)
