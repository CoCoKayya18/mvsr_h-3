#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

class SiftMatcher
{
    public:
        SiftMatcher(cv::Mat InputImage, std::string InputVideoPath):inputImage(InputImage), videoPath(InputVideoPath)
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
        ~SiftMatcher(){};

        struct KeypointTracker 
        {
            cv::KeyPoint keypoint;
            std::vector<cv::Point2f> positions;
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

                // Read each field from the CSV line
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

            std::ifstream keypointsIDFile("../data/activeSet.csv");

            if (!keypointsIDFile.is_open()) {
                std::cerr << "Error: Could not open activeSet file." << std::endl;
                return;
            }

            std::string lineForSelected;
            std::getline(keypointsIDFile, lineForSelected); // Skip the header line
            std::vector<int> selectedIndices;
            while (std::getline(keypointsIDFile, lineForSelected)) {
                int keypointID = std::stoi(lineForSelected);
                selectedIndices.push_back(keypointID);
            }

            keypointsIDFile.close();

            // Extract selected keypoints and descriptors
            for (int idx : selectedIndices) {
                if (idx >= 0 && idx < this->readKeypoints.size()) {
                    this->selectedKeypoints.push_back(this->readKeypoints[idx]);
                    // std::cout << "Saved ID " << idx << std::endl; 
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

            // Save all Keypoints for initialization of the tracking
            for (const auto &kp : this->readKeypoints)
            {
                KeypointTracker tracker;
                tracker.keypoint = kp;
                keypointTrackers.push_back(tracker);
            }

        };

        void matchEveryKeypoint()
        {
            cv::VideoCapture cap(this->videoPath);
            if (!cap.isOpened())
            {
                std::cerr << "Error: Could not open video file." << std::endl;
                return;
            }

            int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
            std::cout << "Total number of frames: " << totalFrames << std::endl;

            cv::namedWindow("Matches", cv::WINDOW_NORMAL);

            while (cap.read(this->frame))
            {
                // Undistort the frame
                cv::undistort(this->frame, this->undistortedFrame, this->cameraMatrix, this->distCoeffs);

                // Extract SIFT features
                auto detector = cv::SIFT::create();
                detector->detectAndCompute(this->undistortedFrame, cv::noArray(), this->frameKeypoints, this->frameDescriptors);

                // Match features with a brute force matcher
                cv::BFMatcher matcher(cv::NORM_L2, true);
                matcher.match(this->readDescriptors, this->frameDescriptors, this->matches);

                this->imgMatches = cv::Mat::zeros(this->height, this->width * 2, this->inputImage.type());
                cv::drawMatches(this->inputImage, this->readKeypoints, this->undistortedFrame, this->frameKeypoints, this->matches, this->imgMatches);

                cv::resize(this->imgMatches, this->imgMatches, cv::Size(this->width, this->height));

                // Display the matches
                cv::imshow("Matches", this->imgMatches);

                // Extract corresponding 2D-3D data structures
                std::vector<cv::Point3f> objectPoints;
                std::vector<cv::Point2f> imagePoints;
                for (const auto &match : matches)
                {
                    imagePoints.push_back(this->frameKeypoints[match.trainIdx].pt);
                    objectPoints.emplace_back(this->readKeypoints[match.queryIdx].pt.x, this->readKeypoints[match.queryIdx].pt.y, 0.0f);
                }

                if (objectPoints.size() < 4 || imagePoints.size() < 4)
                {
                    std::cerr << "Not enough points to solve PnP" << std::endl;
                    continue;
                }

                // Calculate translation and rotation using solvePnP, just testing for later real implementation 
                cv::Mat rvec, tvec;
                cv::solvePnP(objectPoints, imagePoints, this->cameraMatrix, this->distCoeffs, rvec, tvec);

                trackKeypoints(this->matches, this->frameKeypoints);

                if (cv::waitKey(1) == 'q') {
                    break;
                }
            }

            evaluateKeypointStability();

            cap.release();
        };

        void matchSelectedKeypoints()
        {
            cv::VideoCapture cap(this->videoPath);
            if (!cap.isOpened())
            {
                std::cerr << "Error: Could not open video file." << std::endl;
                return;
            }

            cv::namedWindow("Selected Matches", cv::WINDOW_NORMAL);

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
                cv::imshow("Selected Matches", this->selectedImgMatches);

                // Extract corresponding 2D-3D data structures
                std::vector<cv::Point3f> objectPoints;
                std::vector<cv::Point2f> imagePoints;
                for (const auto &match : selectedMatches)
                {
                    imagePoints.push_back(this->selectedFrameKeypoints[match.trainIdx].pt);
                    objectPoints.emplace_back(this->selectedKeypoints[match.queryIdx].pt.x, this->selectedKeypoints[match.queryIdx].pt.y, 0.0f);
                }

                if (objectPoints.size() < 4 || imagePoints.size() < 4)
                {
                    std::cerr << "Not enough points to solve PnP. Object Point Size: " << objectPoints.size() << " Image Point Size: " << imagePoints.size() << std::endl;
                    continue;
                }

                // Calculate translation and rotation using solvePnP
                cv::Mat rvec, tvec;
                cv::solvePnP(objectPoints, imagePoints, this->cameraMatrix, this->distCoeffs, rvec, tvec);

                trackKeypoints(this->selectedMatches, this->selectedFrameKeypoints);

                if (cv::waitKey(1) == 'q') 
                {
                    break;
                }
            }

            evaluateKeypointStability();

            cap.release();
        };

        void trackKeypoints(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& frameKeypoints)
        {
            for (const auto& match : matches)
            {
                // Find the corresponding keypoint tracker
                auto& tracker = keypointTrackers[match.queryIdx];
                
                // Add the matched keypoint position from the current frame
                tracker.positions.push_back(frameKeypoints[match.trainIdx].pt);
            }
        }

        void evaluateKeypointStability()
        {
            std::vector<std::pair<int, double>> keypointVariances;
            std::ofstream outputFile("/root/Hausaufgabe3/data/csvs_for_debugging/KeypointTracking.csv");

            // Write the header for the CSV file
            outputFile << "KeypointIndex,FrameIndex,X,Y,Variance\n";

            for (size_t i = 0; i < keypointTrackers.size(); ++i)
            {
                const auto& tracker = keypointTrackers[i];
                if (tracker.positions.size() < 2)
                {
                    continue; // Skip keypoints with less than 2 positions
                }

                // Calculate the mean position
                cv::Point2f mean(0, 0);
                for (const auto& pos : tracker.positions)
                {
                    mean += pos;
                }
                mean *= (1.0 / tracker.positions.size());

                // Calculate the variance
                double variance = 0;
                for (const auto& pos : tracker.positions)
                {
                    variance += cv::norm(pos - mean) * cv::norm(pos - mean);
                }
                variance /= tracker.positions.size();

                if (!std::isfinite(variance))
                {
                    std::cerr << "Invalid variance for keypoint " << i << ": " << variance << std::endl;
                    continue; // Skip invalid variances
                }

                // Write tracking information for each keypoint with variance
                for (size_t j = 0; j < tracker.positions.size(); ++j)
                {
                    outputFile << i << "," << j << "," << tracker.positions[j].x << "," << tracker.positions[j].y << "," << variance << "\n";
                }

                keypointVariances.emplace_back(i, variance);
            }

            outputFile.close();

            // Sort keypoints by variance
            std::sort(keypointVariances.begin(), keypointVariances.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b)
            {
                return a.second < b.second;
            });

            std::ofstream outFile("/root/Hausaufgabe3/data/csvs_for_debugging/keypointTrackingSorted.csv");
            outFile << "Keypoint,Variance\n";
            for (const auto& kv : keypointVariances) {
                outFile << kv.first << "," << kv.second << "\n";
            }

            // Select the top 10 keypoints with the least variamake
            std::vector<std::pair<int, double>> selectedKeypointsVariances;
            for (size_t i = 0; i < std::min<size_t>(10, keypointVariances.size()); ++i)
            {
                selectedKeypointsVariances.push_back(keypointVariances[i]);
            }

            std::ofstream sift_keypoints_file("/root/Hausaufgabe3/data/csvs_for_debugging/sift_keypoints.csv"); 
            sift_keypoints_file << "Number, x, y, size, angle, response\n"; 
            selectedKeypoints.clear();
            int i = 0;
            std::vector<int> keypointID;

            for (const auto& kv : selectedKeypointsVariances)
            {
                std::cout << "Selected Keypoint: " << kv.first << " Variance: " << kv.second << std::endl;

                keypointID.push_back(kv.first);

                const auto& tracker = keypointTrackers[kv.first - 1];

                sift_keypoints_file << (i + 1) << "," << tracker.keypoint.pt.x << "," << tracker.keypoint.pt.y << ","
                        << tracker.keypoint.size << "," << tracker.keypoint.angle << "," << tracker.keypoint.response << "\n";

                if (!tracker.positions.empty()) {
                    selectedKeypoints.push_back(tracker.keypoint); 
                }

                ++i;
            }

            sift_keypoints_file.close();

            // Saving image for comparison with the image from preprocessing, however, keypoint numbers dont align, but thats just a drawing bug, not important for keypoint selection
            cv::Mat selectedOutput;
            cv::drawKeypoints(this->inputImage, selectedKeypoints, selectedOutput);

            for (size_t i = 0; i < selectedKeypoints.size(); ++i) {

                std::string number = std::to_string(keypointID[i]);
                cv::Point2f point = selectedKeypoints[i].pt;
                cv::putText(selectedOutput, number, point, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                cv::circle(selectedOutput, point, 3, cv::Scalar(255, 0, 0), -1);
                cv::circle(selectedOutput, point, 10, cv::Scalar(0, 255, 0), 2);
            }

            // Save the image with the selected keypoints
            cv::imwrite("/root/Hausaufgabe3/data/images/selected_keypoints_after_variance.jpg", selectedOutput);
        }




    private:
        cv::Mat inputImage;
        std::string videoPath;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;

        cv::Mat frame;
        cv::Mat undistortedFrame;
        cv::Mat imgMatches;

        std::vector<cv::KeyPoint> readKeypoints;
        cv::Mat readDescriptors;
        std::vector<cv::KeyPoint> frameKeypoints;
        cv::Mat frameDescriptors;

        cv::Mat selectedFrame;
        cv::Mat selectedUndistortedFrame;
        cv::Mat selectedImgMatches;

        std::vector<cv::KeyPoint> selectedKeypoints;
        cv::Mat selectedDescriptors;
        std::vector<cv::KeyPoint> selectedFrameKeypoints;
        cv::Mat selectedFrameDescriptors;

        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> selectedMatches;

        int width = 1200;
        int height = 500;

        std::vector<KeypointTracker> keypointTrackers;
};


int main(int argc, char* argv[])
{
    cv::Mat image = cv::imread("/root/Hausaufgabe3/data/images/myImage.jpg");
    std::string videoPath = "/root/Hausaufgabe3/data/videos/Just_Straight.mp4";

    SiftMatcher FeatureMatcher(image, videoPath); 
    FeatureMatcher.matchEveryKeypoint();
    // FeatureMatcher.matchSelectedKeypoints();

    return 0;
}