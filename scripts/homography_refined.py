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

from scripts.jacobian import func_jacobian
from scripts.minimizer import func_minimize


def h_refined(H, correspondences):
    """
    a function to return the refined homography
    :param H: homography matrix
    :param correspondences: coreesponding points
    :return: refined homography matrix
    """
    # flateening the h matrix
    h = H.flatten()
    # flattening the image points
    img_points = correspondences[0]
    y = img_points.flatten()
    # flattening the objects points
    obj_points = correspondences[1]
    X = obj_points.flatten()
    # normalised object points
    norm_obj_points = correspondences[3]
    n = norm_obj_points.shape[0]

    h_prime = opt.least_squares(fun=func_minimize, x0=h, jac=func_jacobian, method="lm", args=[X, y, h, n], verbose=0)

    if h_prime.success:
        H = h_prime.x.reshape(3, 3)
    H = H / H[2, 2]
    return H