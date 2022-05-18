# List of Options

## Global Options
| Name                    |     Type     |                     Default                      | Description                                                                                                                   |
|:------------------------|:------------:|:------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------|
| `seed`                  |    `int`     |                       `0`                        | Random seed.                                                                                                                  |
| `population_size`       |    `int`     | `None`<br/>(Use the recommended value of CMA-ES) | Population size.                                                                                                              |
| `time_max`              |    `int`     |              `None`<br/>(Infinite)               | Allowed maximum computational time for optimization calculation..                                                             |
| `iteration_max`         |    `int`     |                     `10000`                      | Allowed maximum number of iterations.                                                                                         |
| `function_call_max`     |    `int`     |              `None`<br/>(Infinite)               | Allowed maximum number of function calls.                                                                                     |
| `initial_covariance`    |   `float`    |                      `1E0`                       | Initial covariance value for sampling distribution, which scales the identity matrix.                                         |
| `convergence_tolerance` |   `float`    |                     `1E-10`                      | Convergence tolerance, which is checked by mean of diagonal elements of covariance matrix of sampled coordinates.             |
| `lower_bounds`          | `array-like` |             `None` <br/>(Unbounded)              | Lower bounds of variables.                                                                                                    |
| `upper_bounds`          | `array-like` |             `None` <br/>(Unbounded)              | Upper bounds of variables.                                                                                                    |
| `penalty_coefficient`   |   `float`    |                      `1E7`                       | Penalty coefficient for violating lower or upper bounds.                                                                      |
| `verbose`               |    `bool`    |                      `True`                      | If this option is set `True`, calculation log will be displayed in standard output.                                           |
| `log_interval`          |    `int`     |                      `100`                       | Iteration interval to print the calculation log to standard output. This option is effective only if `verbose` is set `True`. |
| `number_of_parallels`   |    `int`     |                       `1`                        | Number of threads to parallelize the computation of objective function values.                                                |

## Options on Regulated Evolution Strategies
| Name                 |  Type   | Default | Description                                                                                                                                      |
|:---------------------|:-------:|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| `external_regulator` | `float` |  `1.0`  | Regulator for sample dispersion convergence. Values smaller than `1.0` (with `delay_factor < 1.0`) guarantee the convergence.                    |
| `delay_factor`       | `float` |  `1.0`  | Delay factor for current covariance matrix. Values smaller than `1.0` delays the influence of the current covariance matrix to subsequent steps. |

## Options on Covariance Matrix Adaption
| Name                                |  Type   |                     Default                      | Description                                                                                |
|:------------------------------------|:-------:|:------------------------------------------------:|:-------------------------------------------------------------------------------------------|
| `learning_rate_center`              | `float` | `None`<br/>(Use the recommended value of CMA-ES) | Learning rate for the center vector of sampling distribution.                              |
| `learning_rate_covariance_rank_one` | `float` | `None`<br/>(Use the recommended value of CMA-ES) | Learning rate for rank-1 updating for the covariance matrix the sampling distribution.     |
| `learning_rate_covariance_rank_mu`  | `float` | `None`<br/>(Use the recommended value of CMA-ES) | Learning rate for rank-`\mu` updating for the covariance matrix the sampling distribution. |
| `is_enabled_step_size_adaption`     | `bool`  |                      `True`                      | If this option is set `True`, Step-Size-Adaption mechanism will be activated.              |