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

# @file    homography.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief wrapper file for calling the functions in scripts folder

import numpy as np


def compute_view_based_homography(correspondence):

    N_u_inv = correspondence[6]
    norm_img_pts = correspondence[2]
    img_pts = correspondence[0]
    obj_pts = correspondence[1]
    norm_obj_pts = correspondence[3]
    N_x = correspondence[5]
    N = len(img_pts)
    M = np.zeros((2 * N, 9), dtype=np.float64)

    for i in range(len(img_pts)):
        # obtaining the normalised image points
        u, v = norm_img_pts[i]
        # obtaining the normalised object points
        x, y = norm_obj_pts[i]
        r1 = np.array([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        r2 = np.array([0, 0, 0, -x, -y, -1, x * v, y * v, v])
        M[2 * i] = r1
        M[(2 * i) + 1] = r2

    # M.h  = 0 . Solve the homogeneous system (e. g. by singular value decomposition):
    u, s, v_h = np.linalg.svd(M)

    # obtaining the minimum eigen value
    h_norm = v_h[np.argmin(s)]
    h_norm = h_norm.reshape(3, 3)

    # de normalize equation 68 of Burger – Zhang’s Camera Calibration Algorithm
    h = np.matmul(np.matmul(N_u_inv, h_norm), N_x)

    # if abs(h[2, 2]) > 10e-8:
    h = h[:, :] / h[2, 2]

    reprojection_error = 0
    for i in range(len(img_pts)):
        t1 = np.array([[obj_pts[i][0]], [obj_pts[i][1]], [1.0]])
        t = np.matmul(h, t1).reshape(1, 3)
        t = t / t[0][-1]
        reprojection_error += np.sum(np.abs(img_pts[i] - t[0][:-1]))

    reprojection_error = np.sqrt(reprojection_error / N)
    print("Reprojection error : ", reprojection_error)

    return h