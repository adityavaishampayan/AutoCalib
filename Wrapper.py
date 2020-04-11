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

# @file    wrapper.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief wrapper file for calling the functions in scripts folder

from scripts.chess_board_corners import getChessboardCorners
from scripts.normalise_coorespondances import normalize_points
from scripts.homography import compute_view_based_homography
from scripts.homography_refined import h_refined
from scripts.camera_intrinsic_param import get_intrinsic_parameters
from scripts.camera_extrinsic_param import estimateExtrinsicParams
from scripts.approx_distortion import estimateReprojectionErrorDistortion
from scripts.visualisation import visualize_pts

import glob
import numpy as np

def main():

    # obtaining the chessboard correspondences between object points and image points
    chessboard_correspondences = getChessboardCorners(images=None, visualize = True)

    # Normalizing the chessboard points for better results of homography
    chessboard_correspondences_normalized = normalize_points(chessboard_correspondences)

    # obtaining homography
    homography = []
    for correspondence in chessboard_correspondences_normalized:
        homography.append(compute_view_based_homography(correspondence))

    # refining the obtsined homography
    homography_refined = []
    for i in range(len(homography)):
        h_opt = h_refined(homography[i], chessboard_correspondences_normalized[i])
        homography_refined.append(h_opt)

    # obtaining the calibration matrix
    K = get_intrinsic_parameters(homography_refined)
    print("camera calibration matrix: ", K)

    extrinsic_para = []
    for i in range(len(homography_refined)):
        extrinsic = estimateExtrinsicParams(K, homography_refined[i])
        extrinsic_para.append(extrinsic)
        print("extrinsic params: ", extrinsic)

    optpoints = []
    for i in range(len(chessboard_correspondences)):
        image_points, object_points = chessboard_correspondences[i]
        points = estimateReprojectionErrorDistortion(K, extrinsic_para[i], image_points, object_points)
        optpoints.append(points)

    optpoints = np.array(optpoints)
    DATA_DIR = r'/home/aditya/hw2/AutoCalib/dataset/Calibration_Imgs/'

    images = [each for each in glob.glob(DATA_DIR + "*.jpg")]
    images = sorted(images)
    for i in range(len(optpoints)):
        image_points, object_points = chessboard_correspondences[i]
        visualize_pts(image_points, optpoints, images[i], i)


if __name__ == '__main__':
    main()




