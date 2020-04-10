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
# @brief file for normalising the corresponding object and image points of the chessboard

import numpy as np


def get_normalization_matrix(pts: object, name: object = "A") -> object:
    """
    A function to obtain the normalisation matrix
    :param pts: points whose normalisation matrix needs to be obtained
    :param name: name of the points e.g. objects points or image points
    :return: normalisation and inverse normalisation matrix
    """
    # changing the type of points
    pts = pts.astype(np.float64)

    # obtaining the mean of the points
    x_mean, y_mean = np.mean(pts, axis=0)
    print("x mean: ", x_mean)
    print("y mean: ", y_mean)

    # obtaining the variance of the points
    x_var, y_var = np.var(pts, axis=0)
    print("variance in x: ", x_var)
    print("variance in y: ", y_var)

    # scaling the points so that the average distance of the points from the origin is equal to sqrt(2)
    s_x = np.sqrt(2 / x_var)
    s_y = np.sqrt(2 / y_var)
    print("standard deviation in x: ", s_x)
    print("standard deviation in y: ", s_y)

    # normalisation matrix as shown in eq 181 pg 27 of the paper Burger – Zhang’s Camera Calibration Algorithm
    n_matrix = np.array([[s_x,   0,  -s_x * x_mean],
                         [0,   s_y,  -s_y * y_mean],
                         [0,     0,              1]])

    # inverse of normalisation matrix as shown in eq 181 pg 27 of the paper Burger – Zhang’s Camera Calibration
    # Algorithm
    n_matrix_inv = np.array([[1.0/s_x,     0,          x_mean],
                             [0,            1.0/s_y,   y_mean],
                             [0,            0,              1]])

    return n_matrix.astype(np.float64), n_matrix_inv.astype(np.float64)


def normalize_points(chessboard_correspondences: object) -> object:
    """
    a function to normalise the correspondences of image and object points
    :param chessboard_correspondences: corresponding points
    :return:
    """

    ret_correspondences = []

    for i in range(len(chessboard_correspondences)):

        image_points, object_points = chessboard_correspondences[i]

        # image points
        homogenous_image_pts = np.array([[[pt[0]], [pt[1]], [1.0]] for pt in image_points])
        normalized_hom_imp = homogenous_image_pts
        n_matrix_imp, n_matrix_imp_inv = get_normalization_matrix(image_points, "image points")

        # object points
        homogenous_object_pts = np.array([[[pt[0]], [pt[1]], [1.0]] for pt in object_points])
        normalized_hom_objp = homogenous_object_pts
        n_matrix_obp, n_matrix_obp_inv = get_normalization_matrix(object_points, "object points")

        for i in range(normalized_hom_objp.shape[0]):

            # normalising the object points by multiplying with normalisation matrix
            n_o = np.matmul(n_matrix_obp, normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o / n_o[-1]

            # normalising the image points by multiplying with normalisation matrix
            n_u = np.matmul(n_matrix_imp, normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u / n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:, :-1]
        normalized_imp = normalized_imp[:, :-1]

        ret_correspondences.append((image_points, object_points, normalized_imp, normalized_objp, n_matrix_imp,
                                    n_matrix_obp, n_matrix_imp_inv, n_matrix_obp_inv))

    return ret_correspondences