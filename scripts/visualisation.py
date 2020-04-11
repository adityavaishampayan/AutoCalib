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

# @file    visualisation.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for outputting the rectified images

import cv2

def visualize_pts(imgpoints, optpoints, image, number):
    """
    a function to visualize the reporjected points
    :param imgpoints: image points
    :param optpoints: output points
    :param image: image
    :param number: image number
    """
    img = cv2.imread(image)
    for i in imgpoints:
        x = i[0]
        y = i[1]
        # x_correct = optpoints[i][0]
        # y_correct = optpoints[i][1]
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        # cv2.rectangle(img, (x_correct - 5, y_correct - 5), (x_correct + 5, y_correct + 5), (0, 255, 0), -1)
        cv2.imwrite("Output/reproj_{}.jpg".format(number), img)
