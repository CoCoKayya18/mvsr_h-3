# Overview
This project consists of three main programs: Preprocessing, SiftMatching, and PoseEstimation. These programs are designed for feature extraction, matching, and pose estimation using SIFT features in images and videos. The necessary data files, including images, videos, and CSV files, are organized within the data directory. Camera calibration data is provided in the CameraStuff directory. Demo-Videos for the last two programs are available in the folder data/videos/demo_videos.

## 1. Preprocessing
Responsible for:

Extracting keypoints and descriptors from the image.
Saving keypoints and descriptors to a CSV file.
Extracting selected keypoints (hardcoded in a vector) and saving the selected keypoint IDs in the activeSet CSV.
Saving an image with the selected keypoints.

## 2. SiftMatching
Responsible for:

Detecting keypoints in a video and matching them with the image keypoints.
Calculating the keypoint variance to select good features with low variance.
Saving an image with the best keypoints to compare with the preprocessing image.
Note: The written index on the image is not correct, but the positions of the features are correct.
Includes a PnP solver for experimental purposes.

## 3. PoseEstimation
Responsible for:

Estimating the position of the image using the PnP solver.
Reading real-world coordinates from the activeSet_XYZ CSV.
Matching real-world coordinates with 2D coordinates.
Retrieving all necessary information from the CSV files in the data folder.

## Data Organization
Videos: Located in the data/videos folder.

Images: Located in the data/images folder.

CSV Files:
Main CSV files are located in the data folder.

Debugging CSV files are saved in the data/csvs_for_debugging folder.

Camera Calibration: Calibration information is located in the CameraStuff folder under the file CameraCalibration.yml. Python scripts for accessing the surface front camera and starting the ros_camera_calibration node are also included in this folder.
