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

# @file    approx_distortion.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief Because we assumed that the camera has minimal distortion we can assume that kc=[0,0]T for a good initial estimate.

import numpy as np
import cv2


def estimateReprojectionErrorDistortion(calib_matrix, extrinsic, imgpoints, objpoints):
    err = []
    reproject_points = []

    u0, v0 = calib_matrix[0, 2], calib_matrix[1, 2]

    for impt, objpt in zip(imgpoints, objpoints):
        model = np.array([[objpt[0]], [objpt[1]], [0], [1]])

        proj_point = np.dot(extrinsic, model)
        proj_point = proj_point / proj_point[2]
        x, y = proj_point[0], proj_point[1]

        U = np.dot(calib_matrix, proj_point)
        U = U / U[2]
        u, v = U[0], U[1]
        k1 = 0.04909
        k2 = -0.3353
        t = x ** 2 + y ** 2
        u_cap = u + (u - u0) * (k1 * t + k2 * (t ** 2))
        v_cap = v + (v - v0) * (k1 * t + k2 * (t ** 2))

        reproject_points.append([u_cap, v_cap])

        err.append(np.sqrt((impt[0] - u_cap) ** 2 + (impt[1] - v_cap) ** 2))

    return np.mean(err), reproject_points
