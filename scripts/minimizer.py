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


def minimizer_func(initial_guess, X, Y, h, N):
    # X : normalized object points flattened
    # Y : normalized image points flattened
    # h : homography flattened
    # N : number of points
    #
    x_j = X.reshape(N, 2)
    # Y = Y.reshape(N, 2)
    # h = h.reshape(3, 3)
    projected = [0 for i in range(2*N)]
    for j in range(N):
        x, y = x_j[j]
        w = h[6]*x + h[7]*y + h[8]
        # pts = np.matmul(np.array([ [h[0], h[1], h[2]] , [h[3], h[4], h[5]]]), np.array([ [x] , [y] , [1.]]))
        # pts = pts/float(w)
        # u, v = pts[0][0], pts[1][0]
        projected[2*j] = (h[0] * x + h[1] * y + h[2]) / w
        projected[2*j + 1] = (h[3] * x + h[4] * y + h[5]) / w

    # return projected
    return (np.abs(projected - Y))**2