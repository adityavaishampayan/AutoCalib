# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2020 Aditya Vaishampayan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# @file    homography_refined.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for refining the homography values


import numpy as np


def func_minimize(initial_guess: object, X: object, Y: object, h: object, N: object) -> object:
    """
    a minimizer function
    :param initial_guess:
    :param X: normalized object points flattened
    :param Y: normalized image points flattened
    :param h: homography flattened
    :param N: number of points
    :return: aboslute difference of estimated value and normalised image points
    """
    x_j = X.reshape(N, 2)

    estimated = []
    for i in range(2*N):
        i == 0
        estimated.append(i)

    for j in range(N):
        x, y = x_j[j]
        estimated[2*j + 1] = (h[3] * x + h[4] * y + h[5]) / h[6]*x + h[7]*y + h[8]
        estimated[2*j] = (h[0] * x + h[1] * y + h[2]) / h[6]*x + h[7]*y + h[8]

    # return estimated
    return (np.abs(estimated-Y))**2