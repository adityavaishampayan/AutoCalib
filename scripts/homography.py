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


def compute_view_based_homography(correspondence, reproj=False):
    """
    correspondence = (imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv)
    """
    image_points = correspondence[0]
    object_points = correspondence[1]
    normalized_image_points = correspondence[2]
    normalized_object_points = correspondence[3]
    N_u = correspondence[4]
    N_x = correspondence[5]
    N_u_inv = correspondence[6]
    N_x_inv = correspondence[7]

    N = len(image_points)
    print("Number of points in current view : ", N)

    M = np.zeros((2 * N, 9), dtype=np.float64)
    print("Shape of Matrix M : ", M.shape)

    print("N_model\n", N_x)
    print("N_observed\n", N_u)

    # create row wise allotment for each 0-2i rows
    # that means 2 rows..
    for i in range(N):
        X, Y = normalized_object_points[i]  # A
        u, v = normalized_image_points[i]  # B

        row_1 = np.array([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u])
        row_2 = np.array([0, 0, 0, -X, -Y, -1, X * v, Y * v, v])
        M[2 * i] = row_1
        M[(2 * i) + 1] = row_2

        print("p_model {0} \t p_obs {1}".format((X, Y), (u, v)))

    # M.h  = 0 . solve system of linear equations using SVD
    u, s, vh = np.linalg.svd(M)
    print("Computing SVD of M")
    # print("U : Shape {0} : {1}".format(u.shape, u))
    # print("S : Shape {0} : {1}".format(s.shape, s))
    # print("V_t : Shape {0} : {1}".format(vh.shape, vh))
    # print(s, np.argmin(s))

    h_norm = vh[np.argmin(s)]
    h_norm = h_norm.reshape(3, 3)
    # print("Normalized Homography Matrix : \n" , h_norm)
    print(N_u_inv)
    print(N_x)
    # h = h_norm
    h = np.matmul(np.matmul(N_u_inv, h_norm), N_x)

    # if abs(h[2, 2]) > 10e-8:
    h = h[:, :] / h[2, 2]

    print("Homography for View : \n", h)

    if reproj:
        reproj_error = 0
        for i in range(len(image_points)):
            t1 = np.array([[object_points[i][0]], [object_points[i][1]], [1.0]])
            t = np.matmul(h, t1).reshape(1, 3)
            t = t / t[0][-1]
            formatstring = "Imp {0} | ObjP {1} | Tx {2}".format(image_points[i], object_points[i], t)
            print(formatstring)
            reproj_error += np.sum(np.abs(image_points[i] - t[0][:-1]))
        reproj_error = np.sqrt(reproj_error / N) / 100.0
        print("Reprojection error : ", reproj_error)

    return h