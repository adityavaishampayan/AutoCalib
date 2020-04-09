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

# @file    chess_board_corners.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for obtaining the chess board corners


import numpy as np
import cv2
import glob

DATA_DIR = r'/home/aditya/hw2/AutoCalib/dataset/Calibration_Imgs/'
DEBUG_DIR = r'/home/aditya/hw2/AutoCalib/Debug/'
PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 1.0


def get_camera_images():
    images = [each for each in glob.glob(DATA_DIR + "*.jpg")]
    images = sorted(images)
    for each in images:
        yield (each, cv2.imread(each, 0))


def getChessboardCorners(images=None, visualize=True):

    objp = np.zeros((PATTERN_SIZE[1] * PATTERN_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    chessboard_corners = []
    image_points = []
    object_points = []
    correspondences = []
    ctr = 0
    for (path, each) in get_camera_images():  # images:
        print("Processing Image : ", path)
        ret, corners = cv2.findChessboardCorners(each, patternSize=PATTERN_SIZE)
        if ret:
            print("Chessboard Detected ")
            corners = corners.reshape(-1, 2)
            if corners.shape[0] == objp.shape[0]:
                # print(objp[:,:-1].shape)
                image_points.append(corners)
                object_points.append(objp[:,:-1])
                # append only World_X, World_Y. Because World_Z is ZERO. Just a simple modification for get_normalization_matrix
                correspondences.append([corners.astype(np.int), objp[:, :-1].astype(np.int)])
            if visualize:
                # Draw and display the corners
                ec = cv2.cvtColor(each, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(ec, PATTERN_SIZE, corners, ret)
                cv2.imwrite(DEBUG_DIR + str(ctr) + ".png", ec)

        else:
            print("Error in detection points", ctr)

        ctr += 1

    return correspondences