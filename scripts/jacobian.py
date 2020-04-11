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

# @file    jacobian.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for obtaining the jacobian of the initial guess

import numpy as np


def func_jacobian(initial_guess, X, Y, h, N):
    x_j = X.reshape(N, 2)
    jacobian = np.zeros( (2*N, 9) , np.float64)
    for j in range(N):
        x, y = x_j[j]
        s_x = np.float64(h[0]*x + h[1]*y + h[2])
        s_y = np.float64(h[3]*x + h[4]*y + h[5])
        w = np.float64(h[6]*x + h[7]*y + h[8])
        jacobian[2*j] = np.array([x/w, y/w, 1/w, 0, 0, 0, -s_x*x/w**2, -s_x*y/w**2, -s_x/w**2])
        jacobian[2*j + 1] = np.array([0, 0, 0, x/w, y/w, 1/w, -s_y*x/w**2, -s_y*y/w**2, -s_y/w**2])

    return jacobian