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
# @brief file for estimating camera extrinsic matrix
import numpy as np

def estimateExtrinsicParams(K, H):
    K_inv = np.linalg.inv(K)

    # Rotation vectors

    r1 = np.dot(K_inv, H[:, 0])
    lamda = np.linalg.norm(r1, ord=2)
    r1 = r1 / lamda

    r2 = np.dot(K_inv, H[:, 1])
    r2 = r2 / lamda

    r3 = np.cross(r1, r2)

    # Translation vectors

    t = np.dot(K_inv, H[:, 2]) / lamda

    R = np.asarray([r1, r2, r3])
    R = R.T

    extrinsic = np.zeros((3, 4))
    extrinsic[:, :-1] = R
    extrinsic[:, -1] = t

    return extrinsic