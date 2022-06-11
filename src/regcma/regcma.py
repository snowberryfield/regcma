# -*- coding: utf-8 -*-

# RegCMA: A Python implementation of Regulated Evolution Strategies[1] with Covariance
# Matrix Adaption[2] for continuous "Black-Box" optimization problems.
#
# [1] Yuji Koguma : Regulated Evolution Strategies: A Framework of Evolutionary
# Algorithms with Stability Analysis Result, IEEJ Transactions on Electrical and
# Electronic Engineering, Vol.15, No.9, pp.1337-1349 (2020).
# https://onlinelibrary.wiley.com/doi/abs/10.1002/tee.23201
#
# [2]  N.Hansen: The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772 [cs.LG]
# (2016). https://arxiv.org/abs/1604.00772
#
# Copyright 2022-2022 Yuji KOGUMA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
import numpy as np

import bokeh.plotting
import bokeh.models
import bokeh.layouts
import bokeh.palettes

from pathos.multiprocessing import ProcessingPool
from datetime import datetime
from dataclasses import dataclass
from typing import Final


class RegCMA:
    @dataclass
    class __Option:
        # Common options and their initial values.
        seed: int = 0
        population_size: int = None
        time_max: float = None
        iteration_max: int = 100000
        function_call_max: int = None
        initial_covariance: float = 1E0
        convergence_tolerance: float = 1E-10
        lower_bounds: np.ndarray = None
        upper_bounds: np.ndarray = None
        penalty_coefficient: float = 1E7
        verbose: bool = True
        log_interval: int = 100
        step_size_and_covariance_normalize_threshold: float = 10.0  # exp

        # RES options and their initial values.
        external_regulator: float = 1.0
        external_regulator_updater: callable = None
        delay_factor: float = 1.0
        delta_limit: float = 5.0  # exp

        # CMA options and their initial values;
        learning_rate_center: float = None
        learning_rate_covariance_rank_one: float = None
        learning_rate_covariance_rank_mu: float = None
        is_enabled_step_size_adaption: bool = True

        # Parallelization
        number_of_parallels: int = 1

    @dataclass
    class __State:
        # Common states and their initial values.
        dimension: int = 0

        iteration: int = 0
        start_time: float = None
        end_time: float = None
        elapsed_time: float = 0
        function_call: int = 0

        incumbent_objective: float = np.inf
        incumbent_solution: np.ndarray = None

        center: np.ndarray = None
        covariance: np.ndarray = None
        step_size: float = 1.0
        covariance_with_step_size: np.ndarray = None

        solutions: np.ndarray = None
        solutions_evaluate: np.ndarray = None

        transformed_moves: np.ndarray = None
        raw_moves: np.ndarray = None

        objectives: np.ndarray = None
        penalties: np.ndarray = None
        augmented_objectives: np.ndarray = None
        ranks: list = None

        convergence_index: float = np.inf

        # Storages for historical plots.
        augmented_objective_mean_trend: list = None
        augmented_objective_best_trend: list = None
        augmented_objective_stdev_trend: list = None
        step_size_trend: list = None
        condition_number_trend: list = None
        internal_regulator_trend: list = None
        external_regulator_trend: list = None
        dispersion_trend: list = None
        dispersion_trend_reference: list = None
        convergence_index_trend: list = None

        # RES states and their initial values.
        internal_regulator: float = 1.0
        external_regulator: float = 1.0
        dispersion: float = 0.0
        dispersion_reference: float = 0.0

        # CMA states and their initial values.
        evolution_path: np.ndarray = None
        conjugate_evolution_path: np.ndarray = None

    @dataclass
    class __CMA:
        # \lambda
        population_size: int = None

        # \mu
        half_population_size: int = None

        # \mu_{eff}
        effective_half_population_size: float = 0.0

        # \alpha_{cov}
        alpha_cov: float = 2.0

        # w
        weights: np.ndarray = None  # w

        # c_{c}
        learning_rate_evolution_path: float = 0.0

        # c_{\sigma}
        learning_rate_conjugate_evolution_path: float = 0.0

        # d_{\sigma}
        step_size_damper: float = 0.0

        # c_{m}
        learning_rate_center: float = 0.0

        # c_{1}
        learning_rate_covariance_rank_one: float = 0.0

        # c_{\mu}
        learning_rate_covariance_rank_mu: float = 0.0

        # E[\mathcal{N}(0, I)]
        chi_n: float = 0.0

    def __init__(self, fun, x0, option=None) -> None:
        if option is not None:
            self.__option = self.__Option(**option)
        else:
            self.__option = self.__Option()
        self.__fun = copy.deepcopy(fun)
        self.__x0 = copy.deepcopy(x0)

        # Setup CMA Parameter.
        self.__setup_cma_parameter()

        # Initialize the sampler state.
        self.__initialize_state()

        # Setup multiprocess pool
        self.__pool = None
        if self.__option.number_of_parallels > 1:
            self.__pool = ProcessingPool(self.__option.number_of_parallels)

    def __setup_cma_parameter(self) -> None:
        """

        Set up the CMA-ES parameters, which is recommended in the paper,
        N.Hansen: The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772
        [cs.LG] (2016).

        """
        # Create aliases to member objects.
        option = self.__option

        # The symbol "d" denotes the dimension of the problem in this method.
        d = len(self.__x0)

        # Initialize the CMA parameter.
        cma = self.__CMA()

        # Determine the population size. If the option explicitly specifies a
        # value of population size, that value is used. Otherwise, it will be
        # determined according to the dimension of the given initial solution
        # vector.
        if option.population_size is not None:
            cma.population_size = option.population_size
        else:
            cma.population_size = 4 + int(3 * np.floor(np.log(d)))

        # Compute the half population size.
        cma.half_population_size = int(cma.population_size / 2)

        # Define alpha_cov.
        cma.alpha_cov = 2.0

        # Compute the weights for higher-rank search points.
        cma.weights = np.zeros(cma.population_size)
        for i in range(cma.half_population_size):
            cma.weights[i] = (
                np.log((cma.population_size + 1.0) / 2.0) - np.log(i + 1.0)
            )
        cma.weights /= np.sum(cma.weights)

        # Compute the effective half population size.
        cma.effective_half_population_size = (
            1.0 / cma.weights.dot(cma.weights)
        )

        # Compute the learning rate for the evolution path.
        cma.learning_rate_evolution_path = (
            (4.0 + cma.effective_half_population_size / d)
            / (d + 4.0 + 2.0 * cma.effective_half_population_size / d)
        )

        # Compute the learning rate for the conjugate evolution path.
        cma.learning_rate_conjugate_evolution_path = (
            (cma.effective_half_population_size + 2.0)
            / (d + cma.effective_half_population_size + 5.0)
        )

        # Compute the step size dumper.
        cma.step_size_damper = (
            1.0 + 2.0
            * max(0.0, np.sqrt((cma.effective_half_population_size - 1.0)
                               / (d + 1.0)) - 1.0)
            + cma.learning_rate_conjugate_evolution_path
        )

        # Compute the learning rate for the center vector.
        if option.learning_rate_center is not None:
            cma.learning_rate_center = (
                option.learning_rate_center
            )
        else:
            cma.learning_rate_center = 1.0

        # Compute the learning rate for Rank-One covariance matrix.
        if option.learning_rate_covariance_rank_one is not None:
            cma.learning_rate_covariance_rank_one = (
                option.learning_rate_covariance_rank_one
            )
        else:
            cma.learning_rate_covariance_rank_one = (
                cma.alpha_cov
                / (pow(d + 1.3, 2) + cma.effective_half_population_size)
            )

        if option.learning_rate_covariance_rank_mu is not None:
            cma.learning_rate_covariance_rank_mu = (
                option.learning_rate_covariance_rank_mu
            )
        else:
            # Compute the learning rate for Rank-\mu covariance matrix.
            NUMERATOR: Final(float) = (
                cma.effective_half_population_size - 2.0
                + 1.0 / cma.effective_half_population_size
            )

            DENOMINATOR: Final(float) = (
                pow(d + 2.0, 2) + cma.alpha_cov *
                cma.effective_half_population_size / 2.0
            )

            cma.learning_rate_covariance_rank_mu = (
                min(1.0 - cma.learning_rate_covariance_rank_one,
                    cma.alpha_cov * NUMERATOR / DENOMINATOR)
            )

        # Compute the (approximate) expected value of Euclidean norm of random
        # vector drawn from d-dimensional the standard multivariate normal
        # distribution.
        cma.chi_n = (
            pow(d, 0.5) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * pow(d, 2.0)))
        )

        self.__cma = cma

    def __initialize_state(self) -> None:
        option = self.__option
        population_size = self.__cma.population_size
        dimension = len(self.__x0)

        state = self.__State()
        state.dimension = dimension
        state.iteration = 0
        state.start_time = None
        state.end_time = None
        state.elapsed_time = 0
        state.function_call = 0

        state.incumbent_objective = np.inf
        state.incumbent_solution = None

        state.center = self.__x0.copy()
        state.covariance = np.identity(dimension) * option.initial_covariance
        state.step_size = 1.0
        state.covariance_with_step_size = state.covariance

        state.solutions = np.zeros([population_size, dimension])
        state.solutions_evaluate = np.zeros([population_size, dimension])

        state.transformed_moves = np.zeros([population_size, dimension])
        state.raw_moves = np.zeros([population_size, dimension])

        state.objectives = np.inf * np.ones(population_size)
        state.penalties = np.inf * np.ones(population_size)
        state.augmented_objectives = np.inf * np.ones(population_size)
        state.ranks = None

        state.convergence_index = np.inf

        state.augmented_objective_mean_trend = []
        state.augmented_objective_best_trend = []
        state.augmented_objective_stdev_trend = []
        state.step_size_trend = []
        state.condition_number_trend = []
        state.internal_regulator_trend = []
        state.external_regulator_trend = []
        state.dispersion_trend = []
        state.dispersion_trend_reference = []
        state.convergence_index_trend = []

        state.internal_regulator = 1.0
        state.external_regulator = option.external_regulator
        state.dispersion = 1.0
        state.dispersion_reference = np.trace(state.covariance_with_step_size)

        state.evolution_path = np.zeros(dimension)
        state.conjugate_evolution_path = np.zeros(dimension)

        self.__current_state = state
        self.__previous_state = copy.deepcopy(self.__current_state)

    def solve(self) -> dict:
        # Create aliases to member objects.
        current_state = self.__current_state
        cma = self.__cma
        option = self.__option

        self.__seed(option.seed)
        self.__set_start_time()
        self.__reset_function_call()
        self.__reset_iteration()

        if option.verbose:
            self.__print_state_head()

        is_state_printed_in_last_iteration = False
        while True:
            self.__update_elapsed_time()

            satisfy_terminating_condition = False

            # Terminate the loop if the elapsed time reaches the specified limit.
            if (option.time_max is not None
                    and current_state.elapsed_time >= option.time_max):
                satisfy_terminating_condition = True

            # Terminate the loop if the iteration reaches the specified limit.
            if (option.iteration_max is not None
                    and current_state.iteration >= option.iteration_max):
                satisfy_terminating_condition = True

            # Terminate the loop if the convergence index reaches within the
            # specified tolerance.
            if current_state.convergence_index < option.convergence_tolerance:
                satisfy_terminating_condition = True

            # Terminate the loop if the number of function calls reaches the
            # specified limit.
            if (option.function_call_max is not None
                and current_state.function_call
                    > option.function_call_max - cma.population_size):
                satisfy_terminating_condition = True

            if satisfy_terminating_condition:
                if not is_state_printed_in_last_iteration and option.verbose:
                    self.__print_state_body()
                break

            self.__update_sample()
            self.__update_objective()
            self.__update_ranks()
            self.__update_incumbent()
            self.__update_state()

            # Print log for the specified interval
            if current_state.iteration % option.log_interval == 0 \
                    or current_state.convergence_index < option.convergence_tolerance:
                if option.verbose:
                    self.__print_state_body()
                    is_state_printed_in_last_iteration = True
            else:
                is_state_printed_in_last_iteration = False

            self.__next_iteration()

        self.__set_end_time()
        self.__print_state_footer()

        RESULT: Final(dict) = self.__create_result()
        return RESULT

    def __update_state(self) -> None:
        # NOTE: Update the sampler state. The dependencies of sub-methods are
        # described in comment in each method.

        # Create an alias to member object.
        option = self.__option

        # Store the current state.
        self.__previous_state = copy.copy(self.__current_state)

        # Update conjugate_evolution_path.
        self.__update_conjugate_evolution_path()

        # Update the step_size
        if option.is_enabled_step_size_adaption:
            self.__update_step_size()

        # Update evolution_path.
        self.__update_evolution_path()

        # Update the center vector of the distribution.
        self.__update_center()

        # Update the covariance matrix of the distribution.
        self.__update_covariance()

        if (abs(np.log(self.__current_state.step_size))) \
                > option.step_size_and_covariance_normalize_threshold:
            self.__normalize_step_size_and_covariance()

        # Update the dispersion.
        self.__update_dispersion()

        # Update the the dispersion reference.
        self.__update_dispersion_reference()

        # Bound the covariance matrix.
        self.__bound_step_size_and_covariance()

        # Update the internal regulator.
        self.__update_internal_regulator()

        # Update the external regulator.
        if option.external_regulator_updater is not None:
            self.__update_external_regulator()

        # Update the convergence index.
        self.__update_convergence_index()

        # Update the trend.
        self.__update_trend()

    def __seed(self, seed) -> None:
        np.random.seed(seed)

    def __update_sample(self) -> None:
        # Create an alias to member object.
        current_state = self.__current_state

        (d, B) = np.linalg.eig(
            current_state.covariance_with_step_size +
            1E-16 * np.identity(current_state.dimension))

        D: Final(np.ndarray) = np.diag(np.sqrt([np.linalg.norm(v) for v in d]))
        L: Final(np.ndarray) = np.dot(B, D)

        for i in range(self.__cma.population_size):
            # Step 1: Sample random vectors from an appropriate standard
            # multivariate normal distribution
            current_state.raw_moves[i] = np.random.randn(
                current_state.dimension
            )

            # Step 2: Transform the random vectors with the prepared lower-
            # triangle matrix.
            current_state.transformed_moves[i] = L.dot(
                current_state.raw_moves[i]
            )

            # Step 3: Shift and scale the transformed random vectors. The
            # obtained vector could be final sample to be evaluated at each
            # iteration.
            current_state.solutions[i] = (
                current_state.center
                + np.sqrt(current_state.external_regulator *
                          current_state.internal_regulator)
                * current_state.transformed_moves[i]
            )

    def __update_objective(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # Create aliases to member objects.
        current_state = self.__current_state
        option = self.__option

        # Compute the bounded solution.
        for i in range(self.__cma.population_size):
            current_state.solutions_evaluate[i] \
                = current_state.solutions[i].copy()

            if option.lower_bounds is not None:
                current_state.solutions_evaluate[i] = (
                    np.maximum(current_state.solutions_evaluate[i],
                               np.array(option.lower_bounds))
                )
            if option.upper_bounds is not None:
                current_state.solutions_evaluate[i] = (
                    np.minimum(current_state.solutions_evaluate[i],
                               np.array(option.upper_bounds))
                )

        # Compute the objective function value for each sample.
        if self.__pool:
            current_state.objectives = self.__pool.map(
                self.__fun, current_state.solutions_evaluate)
        else:
            for i in range(self.__cma.population_size):
                current_state.objectives[i] = self.__fun(
                    current_state.solutions_evaluate[i]
                )

        # Compute the penalty value for each sample.
        for i in range(self.__cma.population_size):
            violation = (
                current_state.solutions[i] -
                current_state.solutions_evaluate[i]
            )
            current_state.penalties[i] = violation.dot(violation)

        # Compute the augmented objective function value for each sample.
        for i in range(self.__cma.population_size):
            current_state.augmented_objectives[i] = (
                current_state.objectives[i]
                + option.penalty_coefficient * current_state.penalties[i]
            )

        self.__update_function_call(self.__cma.population_size)

    def __update_ranks(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()

        # Create aliases to member objects.
        current_state = self.__current_state

        # Argsort by ascending order of objective function values.
        current_state.ranks = np.argsort(current_state.augmented_objectives)

    def __update_incumbent(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()

        # Create aliases to member objects.
        current_state = self.__current_state

        # Update the incumbent solution if the current best solution improves it.
        best_index = current_state.ranks[0]

        if current_state.augmented_objectives[best_index] < current_state.incumbent_objective:
            current_state.incumbent_objective = current_state.augmented_objectives[best_index]
            current_state.incumbent_solution = (
                current_state.solutions[best_index].copy()
            )

    def __update_evolution_path(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()

        # Create aliases to member objects.
        current_state = self.__current_state
        previous_state = self.__previous_state
        cma = self.__cma

        # Compute the weighted center of y.
        weighted_center_transformed_move = np.sum(
            [cma.weights[i] * current_state.transformed_moves[current_state.ranks[i]]
             for i in range(cma.half_population_size)], axis=0
        )

        # Compute the mix-in ratios for previous and current values.
        mixin_ratio_previous = (
            1.0 - cma.learning_rate_evolution_path
        )

        mixin_ratio_current = np.sqrt(
            cma.learning_rate_evolution_path
            * (2.0 - cma.learning_rate_evolution_path)
            * cma.effective_half_population_size
        )

        # Update the evolution path with the mix-in ratios.
        current_state.evolution_path = (
            mixin_ratio_previous * previous_state.evolution_path
            + mixin_ratio_current * weighted_center_transformed_move
            / previous_state.step_size
        )

    def __update_conjugate_evolution_path(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()

        # Create aliases to member objects.
        current_state = self.__current_state
        previous_state = self.__previous_state
        cma = self.__cma

        # Compute the weighted center of z.
        weighted_center_raw_move = np.sum(
            [cma.weights[i] * current_state.raw_moves[current_state.ranks[i]]
             for i in range(cma.half_population_size)], axis=0
        )

        # Compute the mix-in ratios for previous and current values.
        mixin_ratio_previous = (
            1.0 - cma.learning_rate_conjugate_evolution_path
        )

        mixin_ratio_current = np.sqrt(
            cma.learning_rate_conjugate_evolution_path
            * (2.0 - cma.learning_rate_conjugate_evolution_path)
            * cma.effective_half_population_size
        )

        # Update the conjugate evolution path with the mix-in ratios.
        current_state.conjugate_evolution_path = (
            mixin_ratio_previous * previous_state.conjugate_evolution_path
            + mixin_ratio_current * weighted_center_raw_move
        )

    def __update_step_size(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()
        # * __update_conjugate_evolution_path()

        # Create aliases to member objects.
        current_state = self.__current_state
        cma = self.__cma

        # Update the step size with the updated conjugate evolution path.
        current_state.step_size *= np.exp(
            (cma.learning_rate_conjugate_evolution_path / cma.step_size_damper)
            * (np.linalg.norm(current_state.conjugate_evolution_path) / cma.chi_n - 1.0)
        )

    def __update_center(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()

        # Create aliases to member objects.
        current_state = self.__current_state
        cma = self.__cma

        # Compute the weighted center of x.
        weighted_center_solution = np.sum(
            [cma.weights[i] * current_state.solutions[current_state.ranks[i]]
             for i in range(cma.half_population_size)], axis=0
        )

        # Update the center vector of the sampler with the mix-in ratios.
        current_state.center = (
            (1-cma.learning_rate_center) * current_state.center
            + cma.learning_rate_center * weighted_center_solution
        )

    def __update_covariance(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()
        # * __update_evolution_path()
        # * __update_conjugate_evolution_path()
        # * __update_step_size()

        # Create aliases to member objects.
        current_state = self.__current_state
        previous_state = self.__previous_state
        cma = self.__cma

        def self_dyad(X): return np.outer(X, X)

        # Compute the mix-in ratios for rank-one, rank-mu, and previous values.
        mixin_ratio_rank_one = (
            self_dyad(current_state.evolution_path)
        )

        mixin_ratio_rank_mu = (np.sum(
            [cma.weights[i]
             * self_dyad(current_state.transformed_moves[current_state.ranks[i]])
             for i in range(cma.half_population_size)], axis=0)
            / pow(previous_state.step_size, 2.0)
        )

        mixin_ratio_previous = (
            (1.0 - cma.learning_rate_covariance_rank_one -
             cma.learning_rate_covariance_rank_mu)
        )

        # Update the covariance matrix of the sampler with the mix-in ratios.
        current_state.covariance = (
            mixin_ratio_previous * previous_state.covariance
            + mixin_ratio_rank_one * cma.learning_rate_covariance_rank_one
            + mixin_ratio_rank_mu * cma.learning_rate_covariance_rank_mu
        )

        # Update the covariance matrix of considering the step size.
        current_state.covariance_with_step_size = (
            current_state.covariance * pow(current_state.step_size, 2.0)
        )

    def __normalize_step_size_and_covariance(self) -> None:
        # Create aliases to member objects.
        current_state = self.__current_state

        current_state.covariance = (
            current_state.covariance * pow(current_state.step_size, 2.0)
        )
        current_state.step_size = 1.0
        current_state.covariance_with_step_size = current_state.covariance

        current_state.conjugate_evolution_path = np.zeros(
            current_state.dimension
        )

        current_state.evolution_path = np.zeros(current_state.dimension)

    def __update_dispersion(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()
        # * __update_evolution_path()
        # * __update_conjugate_evolution_path()
        # * __update_step_size()
        # * __update_covariance()

        # Create aliases to member objects.
        current_state = self.__current_state
        previous_state = self.__previous_state
        option = self.__option

        # Update the dispersion.
        current_state.dispersion = (
            option.external_regulator * previous_state.internal_regulator
            * np.trace(previous_state.covariance_with_step_size)
            * np.exp(np.trace(np.cov(current_state.transformed_moves.T))
                     / np.trace(previous_state.covariance_with_step_size) - 1.0)
        )

    def __update_dispersion_reference(self) -> None:
        # Create aliases to member objects.
        current_state = self.__current_state
        option = self.__option

        # Update the theoretical dispersion.
        current_state.dispersion_reference *= option.external_regulator

    def __bound_step_size_and_covariance(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()
        # * __update_evolution_path()
        # * __update_conjugate_evolution_path()
        # * __update_step_size()
        # * __update_covariance()
        # * __update_dispersion()

        # Create aliases to member objects.
        current_state = self.__current_state
        option = self.__option

        # Bound the covariance matrix.
        U: Final(float) = np.trace(current_state.covariance_with_step_size)
        V: Final(float) = current_state.dispersion / U

        if abs(np.log(V)) > option.delta_limit:
            if np.log(V) > 0:
                L: Final(float) = np.exp(-option.delta_limit)
            else:
                L: Final(float) = np.exp(option.delta_limit)
            CORRECTOR: Final(float) = np.sqrt(V * L)

            if option.is_enabled_step_size_adaption:
                current_state.covariance *= CORRECTOR
                current_state.step_size *= np.sqrt(CORRECTOR)
            else:
                current_state.covariance *= CORRECTOR**2

            current_state.covariance_with_step_size = (
                current_state.covariance * pow(current_state.step_size, 2.0)
            )

    def __update_internal_regulator(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()
        # * __update_evolution_path()
        # * __update_conjugate_evolution_path()
        # * __update_step_size()
        # * __update_covariance()
        # * __update_dispersion()
        # * __bound_step_size_and_covariance()

        # Create aliases to member objects.
        current_state = self.__current_state
        option = self.__option

        # Update the internal regulator.
        current_state.internal_regulator = (
            pow(current_state.dispersion
                / np.trace(current_state.covariance_with_step_size),
                1.0 - option.delay_factor)
            * pow(current_state.internal_regulator, option.delay_factor)
        )

    def __update_external_regulator(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()
        # * __update_objective()
        # * __update_ranks()
        # * __update_evolution_path()
        # * __update_conjugate_evolution_path()
        # * __update_step_size()
        # * __update_covariance()
        # * __update_dispersion()
        # * __bound_step_size_and_covariance()

        # Create aliases to member objects.
        current_state = self.__current_state
        option = self.__option

        # Update the internal regulator.
        current_state.external_regulator = \
            option.external_regulator_updater(current_state)

    def __update_convergence_index(self) -> None:
        # This method must be called after following methods at each iteration:
        # * __update_sample()

        # Create an alias to member object.
        current_state = self.__current_state

        # Update the convergence index.
        current_state.convergence_index = np.mean(
            np.var(current_state.solutions, axis=0)
        )

    def __update_trend(self) -> None:
        # Create an alias to member object.
        current_state = self.__current_state

        current_state.augmented_objective_best_trend.append(
            current_state.augmented_objectives[current_state.ranks[0]]
        )

        current_state.augmented_objective_mean_trend.append(
            np.mean(current_state.augmented_objectives)
        )

        current_state.augmented_objective_stdev_trend.append(
            np.std(current_state.augmented_objectives)
        )

        current_state.step_size_trend.append(
            current_state.step_size
        )

        current_state.condition_number_trend.append(
            np.linalg.cond(current_state.covariance)
        )

        current_state.internal_regulator_trend.append(
            current_state.internal_regulator
        )

        current_state.external_regulator_trend.append(
            current_state.external_regulator
        )

        current_state.dispersion_trend.append(
            current_state.dispersion
        )

        current_state.dispersion_trend_reference.append(
            current_state.dispersion_reference
        )

        current_state.convergence_index_trend.append(
            current_state.convergence_index
        )

    def __set_start_time(self) -> None:
        self.__current_state.start_time = datetime.now()

    def __set_end_time(self) -> None:
        self.__current_state.end_time = datetime.now()
        self.__update_elapsed_time()

    def __update_elapsed_time(self) -> None:
        self.__current_state.elapsed_time = (
            (datetime.now() - self.__current_state.start_time).total_seconds()
        )

    def __reset_function_call(self) -> None:
        self.__current_state.function_call = 0
        self.__previous_state.function_call = 0

    def __update_function_call(self, count) -> None:
        self.__current_state.function_call += count

    def __reset_iteration(self) -> None:
        self.__current_state.iteration = 0
        self.__previous_state.iteration = 0

    def __next_iteration(self) -> None:
        self.__current_state.iteration += 1

    def __create_result(self) -> dict:
        # Create an alias to member object.
        current_state = self.__current_state

        # Create the result dictionary.
        RESULT: Final(dict) = {
            'status': {
                'start_time': current_state.start_time.isoformat(),
                'end_time': current_state.end_time.isoformat(),
                'elapsed_time': current_state.elapsed_time,
                'function call': current_state.function_call,
            },
            'incumbent': {
                'solution': current_state.incumbent_solution.tolist(),
                'objective': current_state.incumbent_objective
            },
            'last': {
                'solution': current_state.center.tolist(),
                'objective': np.mean(current_state.augmented_objectives)
            }
        }
        return RESULT

    def __print_state_head(self) -> None:
        print('-----+-------------------+---------+-------------------+-------------------+---------')
        print('     |        Current    |  Total  |     Regulator     |      Variance     |  Conv.')
        print('ITER.|     mean     stdev|   min   | external  internal| stepsize    covar.|  index')
        print('-----+-------------------+---------+-------------------+-------------------+---------')

    def __print_state_body(self) -> None:
        # Create aliases to member objects.
        current_state = self.__current_state
        option = self.__option

        print('%05d|%9.2e %9.2e|%9.2e|%9.2e %9.2e|%9.2e %9.2e|%9.2e' % (
            current_state.iteration,
            current_state.augmented_objective_mean_trend[-1],
            current_state.augmented_objective_stdev_trend[-1],
            current_state.incumbent_objective,
            current_state.external_regulator,
            current_state.internal_regulator,
            current_state.step_size,
            current_state.covariance.trace(),
            current_state.convergence_index)
        )

    def __print_state_footer(self) -> None:
        print('------+------------------+---------+-------------------+-------------------+---------')

    def plot_trend(self, output_file_name='result.html'):
        # Create an alias to member object.
        current_state = self.__current_state
        colors = bokeh.palettes.brewer['YlGnBu'][4]

        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
        ]

        iterations = np.arange(
            len(current_state.augmented_objective_best_trend))

        # Objective
        fig_objective = bokeh.plotting.figure(
            tooltips=TOOLTIPS,
            title='Objective',
            x_axis_label='Iteration',
            y_axis_label='Objective',
            x_range=bokeh.models.DataRange1d(start=0),
            y_range=bokeh.models.DataRange1d(start=0),
            plot_width=500,
            plot_height=300)

        fig_objective.circle(
            x=iterations,
            y=current_state.augmented_objective_stdev_trend,
            legend_label='Stdev',
            width=3,
            fill_alpha=0.8,
            color=colors[1])

        fig_objective.circle(
            x=iterations,
            y=current_state.augmented_objective_mean_trend,
            legend_label='Mean',
            width=3,
            fill_alpha=0.8,
            color=colors[0])

        fig_objective.legend.visible = True

        # Step Size
        fig_step_size = bokeh.plotting.figure(
            tooltips=TOOLTIPS,
            title='Step Size',
            x_axis_label='Iteration',
            y_axis_label='Step Size',
            x_range=bokeh.models.DataRange1d(start=0),
            plot_width=500,
            plot_height=300,
            y_axis_type='log')

        fig_step_size.line(
            x=iterations,
            y=current_state.step_size_trend,
            width=3,
            color=colors[0])

        # Condition Number
        fig_condition_number = bokeh.plotting.figure(
            tooltips=TOOLTIPS,
            title='Condition Number',
            x_axis_label='Iteration',
            y_axis_label='Condition Number',
            x_range=bokeh.models.DataRange1d(start=0),
            plot_width=500,
            plot_height=300,
            y_axis_type='log')

        fig_condition_number.line(
            x=iterations,
            y=current_state.condition_number_trend,
            width=3,
            color=colors[0])

        # Regulator
        fig_regulator = bokeh.plotting.figure(
            tooltips=TOOLTIPS,
            title='Regulator',
            x_axis_label='Iteration',
            y_axis_label='Regulator',
            x_range=bokeh.models.DataRange1d(start=0),
            plot_width=500,
            plot_height=300,
            y_axis_type='log')

        fig_regulator.line(
            x=iterations,
            y=current_state.internal_regulator_trend,
            legend_label='Internal',
            width=3,
            color=colors[0])

        fig_regulator.line(
            x=iterations,
            y=current_state.external_regulator_trend,
            legend_label='External',
            width=3,
            color=colors[1])

        fig_regulator.legend.visible = True

        # Dispersion
        fig_dispersion = bokeh.plotting.figure(
            tooltips=TOOLTIPS,
            title='Dispersion',
            x_axis_label='Iteration',
            y_axis_label='Dispersion',
            x_range=bokeh.models.DataRange1d(start=0),
            plot_width=500,
            plot_height=300,
            y_axis_type='log')

        fig_dispersion.line(
            x=iterations,
            y=current_state.dispersion_trend,
            legend_label='Actual',
            width=3,
            color=colors[0])

        fig_dispersion.line(
            x=iterations,
            y=current_state.dispersion_trend_reference,
            legend_label='Theoretical Reference',
            width=3,
            color=colors[1])

        fig_dispersion.legend.visible = True

        # Convergence Index
        fig_convergence_index = bokeh.plotting.figure(
            tooltips=TOOLTIPS,
            title='Convergence Index',
            x_axis_label='Iteration',
            y_axis_label='Convergence Index',
            x_range=bokeh.models.DataRange1d(start=0),
            plot_width=500,
            plot_height=300,
            y_axis_type='log')

        fig_convergence_index.line(
            x=iterations,
            y=current_state.convergence_index_trend,
            width=3,
            color=colors[0])

        grid = bokeh.layouts.gridplot(
            [[fig_objective, fig_step_size],
             [fig_condition_number, fig_regulator],
             [fig_dispersion, fig_convergence_index]])
        bokeh.plotting.output_file(output_file_name, title='RegCMA Trend')
        bokeh.plotting.save(grid)


def solve(fun, x0, option=None, plot=False):
    """ RegCMA solver Interface

    Parameters
    ----------
    fun : Function object
        An objective function to be minimized which can compute objective
        function values with the usage of fun(x), where x denotes a real-
        value vector with appropriate dimension.
    x0 : ndarray
        Initial solution with appropriate dimension.
    **kwargs: dict
        Pairs of option key/value to configure the solver.

    Returns
    ------
    dict
        optimization result

    """
    solver = RegCMA(fun, x0, option)
    result = solver.solve()
    if plot:
        solver.plot_trend()

    return result

################################################################################
# END
################################################################################
