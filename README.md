# AutoCalib
Estimating camera parameters like focal length, distortion coefficients and principal point automatically

# Procedure Summary
The recommended calibration procedure is as follows:
1. Print a pattern and attach it to a planar surface;
2. Take a few images of the model plane under different orientations by moving either the plane
or the camera;
3. Detect the feature points in the images;
4. Estimate the five intrinsic parameters and all the extrinsic parameters using the closed-form
solution.
5. Estimate the coefficients of the radial distortion by solving the linear least-squares;
6. Refine all parameters by minimiziation.


# Distortion
Radial distortion causes straight lines to appear curved. Radial distortion becomes larger the farther points are from the center of the image.

tangential distortion occurs because the image-taking lense is not aligned perfectly parallel to the imaging plane. So, some areas in the image may look nearer than expected.

 Intrinsic parameters are specific to a camera. They include information like focal length ( fx,fy) and optical centers ( cx,cy). The focal length and optical centers can be used to create a camera matrix, which can be used to remove distortion due to the lenses of a specific camera. The camera matrix is unique to a specific camera, so once calculated, it can be reused on other images taken by the same camera. It is expressed as a 3x3 matrix
 
 Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system.

For stereo applications, these distortions need to be corrected first. To find these parameters, we must provide some sample images of a well defined pattern (e.g. a chess board). We find some specific points of which we already know the relative positions (e.g. square corners in the chess board). We know the coordinates of these points in real world space and we know the coordinates in the image, so we can solve for the distortion coefficients. For better results, we need at least 10 test patterns.

# Calibration matrix

camera calibration matrix:  [[ 2.05759563e+03 -6.68390012e-01  7.64017398e+02]
 [ 0.00000000e+00  2.04457744e+03  1.36076094e+03]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]