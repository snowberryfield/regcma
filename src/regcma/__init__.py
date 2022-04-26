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

from .regcma import solve

VERSION = '0.0.1'
