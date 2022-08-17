# RegCMA
**RegCMA** is a Python implementation of Regulated Evolution Strategies [1] with Covariance Matrix Adaption [2] for continuous "Black-Box" optimization problems. RegCMA is suitable for optimization in situations where the allowed number of iterations is limited such as hyper-parameter tuning in machine learning model and optimization algorithm, and it attempts to make the most of them to find good solutions. 

## Installation
```
pip install git+https://github.com/snowberryfield/regcma.git
```

## Algorithm
**Regulated Evolution Strategies** *(hereinafter referred to as RES)* is a evolutionary algorithm framework to attempts to find reasonable solutions of following unconstrained minimization problems:

$$
(\mathrm{P}): \underset{\boldsymbol{x}}{\mathrm{minimize}}\enspace f(\boldsymbol{x})
$$

where $\boldsymbol{x}$ denotes $N$-dimensional decision variables, and $f:R^N \mapsto R$ denotes "Black-Box" objective function to be minimized. RES-based algorithm should satisfy the following updating structure:

$$
\begin{aligned}
\boldsymbol{x}^{p}(k) &= \boldsymbol{\mu}(k) + \sqrt{\alpha(k) r(k)} \boldsymbol{y}^{p}(k) \enspace (p=1,\dots,P), \\
 \boldsymbol{S}_{\boldsymbol{y}}(k) &= \frac{1}{P} \sum_{p=1}^{P} \boldsymbol{y}^{p}(k)\boldsymbol{y}^{p\top}(k), \\
T(k)          &= \alpha(k) r(k) \mathrm{Tr}  \boldsymbol{C}(k) \exp \left(\frac{\mathrm{Tr} \boldsymbol{S}_{\boldsymbol{y}}(k)}{\mathrm{Tr}  \boldsymbol{C}(k)} - 1\right), \\
r(k+1)        &= \left(\frac{T(k)}{\mathrm{Tr} \boldsymbol{C}(k+1)} \right)^{1-\beta}r^{\beta}(k),
\end{aligned}
$$

where $k = 0,1,\dots$ denotes the iteration, $P \in \mathbb{N}$ denotes the number of samples and $\boldsymbol{x}^{p}(k) \in \mathbb{R}^{N}$ denote individual samples. The random vector $\boldsymbol{y}^{p}(k) \in \mathbb{R}^{N}$ to generate $\boldsymbol{x}^{p}(k)$ are drawn from ${\cal N}(\boldsymbol{0},\boldsymbol{C}(k))$. In RES framework, the distribution parameters $\boldsymbol{\mu}(k) \in \mathbb{R}^{N}$ and $\boldsymbol{C}(k) \in \mathbb{R}^{N \times N}$ can be updated in an arbitrary manner, meanwhile RegCMA employs that of CMA-ES. The symbol $r(k)$ is the *internal* regulator that attenuates the influence of the current covariance matrix $\boldsymbol{C}(k)$ on $k \to \infty$. It is initialized by $r(k)=1$ and updated by with the delay factor $\beta \in (0,1)$. The *external* regulator $\alpha(k) \in\{0,1\}$ is also a parameter that controls convergence speed of the sample dispersion (In this program, only constant $\alpha$ can be specified). Internal state $T(k)$, which indicates the dispersion of samples, is an intentional approximation of 

$$
\mathrm{Tr} \boldsymbol{S}_{\boldsymbol{x}}(k) = \sum_{p=1}^{P} \frac{1}{P} \boldsymbol{x}^{p}(k)\boldsymbol{x}^{p\top}(k)
$$

to enable rigorous analysis of the RES framework.

RES provides the following theorem that states convergence of sample dispersion.

**Theorem (Sufficient stability condition of RES-based algorithms) [1]**: Let ${\cal A}$ denote an RES-based algorithm. Suppose that the constants $L_{\log \alpha}(k) \in \mathbb{R}$ and $L_{\log r} > 0$ exist such that $\log \alpha(k) \le L_{\log \alpha}(k) < +\infty (\forall k\ge 0)$ and $\left\lvert \log  T(k) / \mathrm{Tr} \boldsymbol{C}(k+1) \right\rvert \le L_{\log r} < +\infty\, (\forall k\ge 0)$. We also assume that $\boldsymbol{y}^{p}(k) \, (k \ge 0, p=1,\dots,P)$ are independent of each other. If ${\cal A}$ is designed so that

$$
\lim_{k \to \infty} \frac{1}{\sqrt{k+1}} \sum_{\kappa=0}^{k} L_{\log \alpha}(\kappa) = -\infty,
$$

then

$$
\lim_{k \to \infty} \Pr \left(T(k) \ge \varepsilon \right) = 0\, (\forall \varepsilon > 0)
$$

holds. Also, as a byproduct of this theorem, the following estimator is obtained:

$$
\log T(k) \approx \sum_{\kappa=0}^{k}\log \alpha(\kappa) + \log \mathrm{Tr} \boldsymbol{C}(0).
$$

With this estimator, we can design the value of $\alpha(k)$ with allowed iteration and target tolerance of $T(\simeq \boldsymbol{S}_{\boldsymbol{x}})$.

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

option = {
    'iteration_max': 100,
    'initial_covariance': 1E0,
    'convergence_tolerance': 1E-10
}

result = regcma.solve(quadratic, x0, option, plot=True)
```

In the example above, RegCMA minimizes the objective function with regulating its sampling dispersion so that the convergence index (mean of diagonal components of the $\boldsymbol{S}_{\boldsymbol{x}}$) gradually reaches `1E-10` at iteration `100`. The following plot depicts the search trend. The chart of *Convergence Index* shows that actual convergence index tracks the theoretical reference.

![](./asset/sample_plot.png)

## Options
Please refer [List of Options](options.md).

## License
**RegCMA** is distributed under [MIT license](https://opensource.org/licenses/MIT).

## References
- [1] Yuji Koguma : Regulated Evolution Strategies: A Framework of Evolutionary Algorithms with Stability Analysis Result, IEEJ Transactions on Electrical and Electronic Engineering, Vol.15, No.9, pp.1337-1349 (2020).
https://onlinelibrary.wiley.com/doi/abs/10.1002/tee.23201

- [2]  N.Hansen: The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772 [cs.LG] (2016). 
https://arxiv.org/abs/1604.00772