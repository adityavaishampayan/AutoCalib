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
from scripts.homography_refined import refine_homographies
from scripts.camera_intrinsic_param import get_intrinsic_parameters


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
        h_opt = refine_homographies(homography[i], chessboard_correspondences_normalized[i], skip=False)
        homography_refined.append(h_opt)

    # obtaining the calibration matrix
    K = get_intrinsic_parameters(homography_refined)

    print("camera calibration matrix: ", K)


if __name__ == '__main__':
    main()




