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

# @file    camera_extrinsic_param.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for estimating camera extrinsic parameters
import numpy as np


def estimateExtrinsicParams(K, H):
    """
    a function for estimating the camera extrinsic parameters
    :param K: calibration matrix K
    :param H: homogrpahy
    :return: extrinsic matrix containing r and t
    """
    inv_k = np.linalg.inv(K)
    rot_1 = np.dot(inv_k, H[:, 0])
    lamda = np.linalg.norm(rot_1, ord=2)

    rot_1 = rot_1 / lamda
    rot_2 = np.dot(inv_k, H[:, 1])
    rot_2 = rot_2 / lamda
    rot_3 = np.cross(rot_1, rot_2)
    t = np.dot(inv_k, H[:, 2]) / lamda
    R = np.asarray([rot_1, rot_2, rot_3])
    R = R.T
    extrinsic = np.zeros((3, 4))
    extrinsic[:, :-1] = R
    extrinsic[:, -1] = t

    return extrinsic