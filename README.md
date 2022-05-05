# RegCMA
**RegCMA** is a Python implementation of Regulated Evolution Strategies [1] with Covariance Matrix Adaption [2] for continuous "Black-Box" optimization problems.

## Installation
```
pip install git+https://github.com/snowberryfield/regcma.git
```

## Example
```python
# -*- coding: utf-8 -*-
import regcma
import numpy as np

# Define the objective function to be minimized.
def quadratic(x): return x.dot(x)

# Define the initial solution.
DIMENSION = 10
x0 = np.random.randn(DIMENSION)

result = regcma.solve(quadratic, x0)
print(result)
```

## License
**RegCMA** is distributed under [MIT license](https://opensource.org/licenses/MIT).

## References
- [1] Yuji Koguma : Regulated Evolution Strategies: A Framework of Evolutionary Algorithms with Stability Analysis Result, IEEJ Transactions on Electrical and Electronic Engineering, Vol.15, No.9, pp.1337-1349 (2020).
https://onlinelibrary.wiley.com/doi/abs/10.1002/tee.23201

- [2]  N.Hansen: The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772 [cs.LG] (2016). 
https://arxiv.org/abs/1604.00772