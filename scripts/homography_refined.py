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
from scipy import optimize as opt

from scripts.jacobian import jac_function
from scripts.minimizer import minimizer_func

def refine_homographies(H, correspondences, skip=False):
    if skip:
        return H

    image_points = correspondences[0]
    object_points = correspondences[1]
    normalized_image_points = correspondences[2]
    normalized_object_points = correspondences[3]
    N_u = correspondences[4]
    N_x = correspondences[5]
    N_u_inv = correspondences[6]
    N_x_inv = correspondences[7]

    N = normalized_object_points.shape[0]
    X = object_points.flatten()
    Y = image_points.flatten()
    h = H.flatten()
    h_prime = opt.least_squares(fun=minimizer_func, x0=h, jac=jac_function, method="lm", args=[X, Y, h, N], verbose=0)

    if h_prime.success:
        H = h_prime.x.reshape(3, 3)
    H = H / H[2, 2]
    return H