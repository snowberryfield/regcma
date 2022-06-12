# -*- coding: utf-8 -*-
import regcma
import numpy as np

# Define the objective function to be minimized.
def quadratic(x): return x.dot(x)

# Define the initial solution.
DIMENSION = 10
x0 = np.random.randn(DIMENSION)

option = {
    'iteration_max': 100
}

result = regcma.solve(quadratic, x0, option)
print(result)
