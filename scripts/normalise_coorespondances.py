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

# @file    normalise_coorespondances.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief wrapper file for normalising the corresponding object and image points of the chessboard

import numpy as np


def get_normalization_matrix(pts, name="A"):
    pts = pts.astype(np.float64)
    x_mean, y_mean = np.mean(pts, axis=0)
    var_x, var_y = np.var(pts, axis=0)

    s_x, s_y = np.sqrt(2 / var_x), np.sqrt(2 / var_y)

    print("Matrix: {4} : meanx {0}, meany {1}, varx {2}, vary {3}, sx {5}, sy {6} ".format(x_mean, y_mean, var_x,
                                                                                           var_y, name, s_x, s_y))

    n = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])
    # print(n)

    n_inv = np.array([[1. / s_x, 0, x_mean], [0, 1. / s_y, y_mean], [0, 0, 1]])
    return n.astype(np.float64), n_inv.astype(np.float64)


def normalize_points(chessboard_correspondences):
    views = len(chessboard_correspondences)

    ret_correspondences = []
    for i in range(views):
        imp, objp = chessboard_correspondences[i]
        N_x, N_x_inv = get_normalization_matrix(objp, "A")
        N_u, N_u_inv = get_normalization_matrix(imp, "B")
        # print(N_x)
        # print(N_u)
        # convert imp, objp to homogeneous
        # hom_imp = np.array([np.array([[each[0]], [each[1]], [1.0]]) for each in imp])
        # hom_objp = np.array([np.array([[each[0]], [each[1]], [1.0]]) for each in objp])
        hom_imp = np.array([[[each[0]], [each[1]], [1.0]] for each in imp])
        hom_objp = np.array([[[each[0]], [each[1]], [1.0]] for each in objp])

        normalized_hom_imp = hom_imp
        normalized_hom_objp = hom_objp

        for i in range(normalized_hom_objp.shape[0]):
            # 54 points. iterate one by onea
            # all points are homogeneous
            n_o = np.matmul(N_x, normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o / n_o[-1]

            n_u = np.matmul(N_u, normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u / n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:, :-1]
        normalized_imp = normalized_imp[:, :-1]

        # print(normalized_imp)

        ret_correspondences.append((imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv))

    return ret_correspondences