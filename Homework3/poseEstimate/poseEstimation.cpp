#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

class PoseEstimator
{
    public:
        PoseEstimator(cv::Mat InputImage, std::string InputVideoPath):inputImage(InputImage), videoPath(InputVideoPath)
        {
            // Setup for Calibration Matrix
            cv::FileStorage fs("/root/Hausaufgabe3/CameraStuff/CameraCalibration.yml", cv::FileStorage::READ);
            if (!fs.isOpened())
            {
                std::cerr << "Failed to open calibration file." << std::endl;
            }

            // Load the camera matrix and distortion coefficients
            fs["camera_matrix"] >> this->cameraMatrix;
            fs["distortion_coefficients"] >> this->distCoeffs;

            fs.release();

            this->readKeypointsAndDescriptors();
        };
        ~PoseEstimator(){};

        struct coordinates
        {
            int ID;
            float x_position;
            float y_position;
            float z_position;
        };

        void readKeypointsAndDescriptors()
        {
            // Read keypoints from CSV file
            std::ifstream keypointsFile("../data/keypoints.csv");
            if (!keypointsFile.is_open()) {
                std::cerr << "Error: Could not open keypoints file." << std::endl;
                return;
            }

            std::string line;
            std::getline(keypointsFile, line); // Skip the header line
            while (std::getline(keypointsFile, line)) {
                std::istringstream ss(line);
                std::string token;
                cv::KeyPoint keypoint;

                std::getline(ss, token, ','); // Skip the number field
                std::getline(ss, token, ',');
                keypoint.pt.x = std::stof(token);
                std::getline(ss, token, ',');
                keypoint.pt.y = std::stof(token);
                std::getline(ss, token, ',');
                keypoint.size = std::stof(token);
                std::getline(ss, token, ',');
                keypoint.angle = std::stof(token);
                std::getline(ss, token, ',');
                keypoint.response = std::stof(token);

                this->readKeypoints.push_back(keypoint);
            }
            keypointsFile.close();

            // Read descriptors from CSV file
            std::ifstream descriptorsFile("../data/descriptors.csv");
            if (!descriptorsFile.is_open()) {
                std::cerr << "Error: Could not open descriptors file." << std::endl;
                return;
            }

            std::vector<std::vector<float>> descriptorsList;
            std::getline(descriptorsFile, line); 
            while (std::getline(descriptorsFile, line)) {
                std::istringstream ss(line);
                std::string token;
                std::vector<float> descriptor;

                std::getline(ss, token, ',');

                int token_count = 0; // To count the number of tokens in each line
                while (std::getline(ss, token, ',')) {
                    descriptor.push_back(std::stof(token));
                    ++token_count;
                }

                descriptorsList.push_back(descriptor);
            }

            descriptorsFile.close();

            // Convert the descriptors to cv::Mat
            int rows = descriptorsList.size();
            int cols = descriptorsList[0].size();
            this->readDescriptors = cv::Mat(rows, cols, CV_32F);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    this->readDescriptors.at<float>(i, j) = descriptorsList[i][j];
                }
            }

            // Read keypointIDs from csv
            std::ifstream keypointsIDFile("../data/activeSet.csv");

            if (!keypointsIDFile.is_open()) {
                std::cerr << "Error: Could not open activeSet file." << std::endl;
                return;
            }

            std::string lineForSelected;
            std::getline(keypointsIDFile, lineForSelected); // Skip the header line
            std::vector<int> selectedIndices;
            while (std::getline(keypointsIDFile, lineForSelected)) {
                int keypointID = std::stoi(lineForSelected) - 1; 
                selectedIndices.push_back(keypointID);
            }

            keypointsIDFile.close();

            // Extract selected keypoints and descriptors
            for (int idx : selectedIndices) {
                if (idx >= 0 && idx < this->readKeypoints.size()) {
                    this->selectedKeypoints.push_back(this->readKeypoints[idx]);
                    this->selectedDescriptors.push_back(this->readDescriptors.row(idx));
                } else {
                    std::cerr << "Invalid keypoint index: " << idx << std::endl;
                }
            }

            // Convert selected descriptors to cv::Mat
            if (!selectedIndices.empty()) {
                this->selectedDescriptors = cv::Mat(selectedIndices.size(), this->readDescriptors.cols, this->readDescriptors.type());
                for (size_t i = 0; i < selectedIndices.size(); ++i) {
                    this->readDescriptors.row(selectedIndices[i]).copyTo(this->selectedDescriptors.row(i));
                }
            }

            // Read world coordinates from the csv
            std::ifstream threeDCoordinatesFile("../data/activeSet_XYZ.csv");
            std::string lineForCoordinates;
            std::getline(threeDCoordinatesFile, lineForCoordinates); // Skip the header line

            while (std::getline(threeDCoordinatesFile, lineForCoordinates)) 
            {
                std::istringstream ss(lineForCoordinates);
                std::string token;
                coordinates tempCoords;
                
                std::getline(ss, token, ',');
                tempCoords.ID = std::stoi(token);
                std::getline(ss, token, ',');
                tempCoords.x_position = std::stof(token);
                std::getline(ss, token, ',');
                tempCoords.y_position = std::stof(token);
                std::getline(ss, token, ',');
                tempCoords.z_position = std::stof(token);
                this->selectedCoordinates.push_back(tempCoords);
            }

            threeDCoordinatesFile.close();
        };

        void doSomePoseEstimation()
        {

            cv::VideoCapture cap(this->videoPath);
            if (!cap.isOpened())
            {
                std::cerr << "Error: Could not open video file." << std::endl;
                return;
            }

            cv::namedWindow("Pose Estimation", cv::WINDOW_NORMAL);

            while (cap.read(this->selectedFrame))
            {
                // Undistort the frame
                cv::undistort(this->selectedFrame, this->selectedUndistortedFrame, this->cameraMatrix, this->distCoeffs);

                // Extract SIFT features
                auto detector = cv::SIFT::create();
                detector->detectAndCompute(this->selectedUndistortedFrame, cv::noArray(), this->selectedFrameKeypoints, this->selectedFrameDescriptors);

                // Match features with a brute force matcher
                cv::BFMatcher matcher(cv::NORM_L2, true);
                matcher.match(this->selectedDescriptors, this->selectedFrameDescriptors, this->selectedMatches);

                this->selectedImgMatches = cv::Mat::zeros(this->height, this->width * 2, this->inputImage.type());
                cv::drawMatches(this->inputImage, this->selectedKeypoints, this->selectedUndistortedFrame, this->selectedFrameKeypoints, this->selectedMatches, this->selectedImgMatches);

                cv::resize(this->selectedImgMatches, this->selectedImgMatches, cv::Size(this->width, this->height));

                // Display the matches
                cv::imshow("Pose Estimation", this->selectedImgMatches);


                // Extract corresponding 2D-3D data structures
                std::vector<cv::Point3f> objectPoints;
                std::vector<cv::Point2f> imagePoints;
                for (const auto &match : selectedMatches)
                {
                    imagePoints.push_back(this->selectedFrameKeypoints[match.trainIdx].pt);
                    objectPoints.push_back(cv::Point3f(this->selectedCoordinates[match.trainIdx].x_position, this->selectedCoordinates[match.trainIdx].y_position, this->selectedCoordinates[match.trainIdx].z_position));
                }

                if (objectPoints.size() < 4 || imagePoints.size() < 4)
                {
                    std::cerr << "Not enough points to solve PnP. Object Point Size: " << objectPoints.size() << " Image Point Size: " << imagePoints.size() << std::endl;
                    continue;
                }

                // Calculate translation and rotation using solvePnP
                cv::Mat rvec, tvec;
                cv::solvePnP(objectPoints, imagePoints, this->cameraMatrix, this->distCoeffs, rvec, tvec);

                // Print the translation and rotation vector
                std::cout << "Translation: " << tvec.t() << std::endl;
                std::cout << "Rotation: " << rvec.t() << std::endl;

                if (cv::waitKey(1) == 'q') 
                {
                    break;
                }
            }

            cap.release();
        };

    private:
        cv::Mat inputImage;
        std::string videoPath;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;

        std::vector<cv::KeyPoint> readKeypoints;
        cv::Mat readDescriptors;

        cv::Mat selectedFrame;
        cv::Mat selectedUndistortedFrame;
        cv::Mat selectedImgMatches;

        std::vector<cv::KeyPoint> selectedKeypoints;
        cv::Mat selectedDescriptors;
        std::vector<cv::KeyPoint> selectedFrameKeypoints;
        cv::Mat selectedFrameDescriptors;

        std::vector<coordinates> selectedCoordinates;
        std::vector<cv::DMatch> selectedMatches;

        int width = 1200;
        int height = 500;


};


int main(int argc, char* argv[])
{
    cv::Mat image = cv::imread("/root/Hausaufgabe3/data/images/myImage.jpg");
    // std::string videoPath = "/root/Hausaufgabe3/data/videos/Just_Straight.mp4";
    // std::string videoPath = "/root/Hausaufgabe3/data/videos/Movement.mp4";
    std::string videoPath = "/root/Hausaufgabe3/data/videos/More_Movement.mp4";

    PoseEstimator poseEstimator2000(image, videoPath); 
    poseEstimator2000.doSomePoseEstimation();

    return 0;
}