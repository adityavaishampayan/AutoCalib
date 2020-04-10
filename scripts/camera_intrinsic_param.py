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

# @file    camera_intrinsic_param.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for estimating camera intrinsic matrix

import numpy as np


def get_intrinsic_parameters(H_r):
    M = len(H_r)
    V = np.zeros((2*M, 6), np.float64)

    def v_pq(p, q, H):
        v = np.array([
                H[0, p]*H[0, q],
                H[0, p]*H[1, q] + H[1, p]*H[0, q],
                H[1, p]*H[1, q],
                H[2, p]*H[0, q] + H[0, p]*H[2, q],
                H[2, p]*H[1, q] + H[1, p]*H[2, q],
                H[2, p]*H[2, q]
            ])
        return v

    for i in range(M):
        H = H_r[i]
        V[2*i] = v_pq(p=0, q=1, H=H)
        V[2*i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))

    # solve V.b = 0
    u, s, vh = np.linalg.svd(V)
    # print(u, "\n", s, "\n", vh)
    b = vh[np.argmin(s)]
    print("V.b = 0 Solution : ", b.shape)

    # according to zhangs method
    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) *(beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    print([vc,
            l,
            alpha,
            beta,
            gamma,
        uc])

    A = np.array([
            [alpha, gamma, uc],
            [0, beta, vc],
            [0, 0, 1.0],
        ])
    print("Intrinsic Camera Matrix is :")
    print(A)
    return A