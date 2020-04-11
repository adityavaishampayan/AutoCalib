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


def vpq(p, q, homography):
    v = np.array([homography[0, p] * homography[0, q],
                  homography[0, p] * homography[1, q] + homography[1, p] * homography[0, q],
                  homography[1, p] * homography[1, q],
                  homography[2, p] * homography[0, q] + homography[0, p] * homography[2, q],
                  homography[2, p] * homography[1, q] + homography[1, p] * homography[2, q],
                  homography[2, p] * homography[2, q]])
    return v


def get_intrinsic_parameters(r_h):
    M = len(r_h)
    V = np.zeros((2*M, 6), np.float64)

    for i in range(M):
        H = r_h[i]
        V[2*i] = vpq(p=0, q=1, homography=H)
        V[2*i + 1] = np.subtract(vpq(p=0, q=0, homography=H), vpq(p=1, q=1, homography=H))

    u, s, v_h = np.linalg.svd(V)
    b = v_h[np.argmin(s)]

    # according to zhangs equations
    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) * (beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    print([vc, l, alpha, beta, gamma, uc])

    A = np.array([[alpha, gamma, uc],
                  [0,     beta,  vc],
                  [0,      0,   1.0]])
    return A