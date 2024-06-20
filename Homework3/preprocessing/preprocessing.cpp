#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

class Sift
{
    public:
        Sift(cv::Mat InputImage):inputImage(InputImage){};
        ~Sift(){};

        void getKeypoints()
        {
            auto detector = cv::SIFT::create();
            detector->detectAndCompute(this->inputImage, cv::noArray(), this->keypoints, this->descriptors);

            cv::Mat output;
            cv::drawKeypoints(this->inputImage, this->keypoints, output);

            // Number the features and draw them on the image
            for (int i = 0; i < this->keypoints.size(); ++i) {
                std::string number = std::to_string(i+1);
                cv::Point2f point = this->keypoints[i].pt;
                cv::putText(output, number, point, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
            }

            // Save keypoints to a CSV file
            std::ofstream keypointsFile;
            keypointsFile.open("/root/Hausaufgabe3/data/keypoints.csv");
            keypointsFile << "Number, x, y, size, angle, response\n"; 
            for (int i = 0; i < this->keypoints.size(); ++i) {
                const cv::KeyPoint& kpt = this->keypoints[i];
                keypointsFile << (i+1) << "," << kpt.pt.x << "," << kpt.pt.y << ","
                            << kpt.size << "," << kpt.angle << "," << kpt.response << "\n";
            }
            keypointsFile.close();

            // Save descriptors to a CSV file
            std::ofstream descriptorsFile;
            descriptorsFile.open("/root/Hausaufgabe3/data/descriptors.csv");
            
            descriptorsFile << "Number";
            for (int j = 0; j < descriptors.cols; ++j) {
                descriptorsFile << ",Descriptor-" << j;
            }
            descriptorsFile << "\n";

            for (int i = 0; i < descriptors.rows; ++i) {
                descriptorsFile << (i + 1);
                for (int j = 0; j < descriptors.cols; ++j) {
                    descriptorsFile << "," << descriptors.at<float>(i, j);
                }
                descriptorsFile << "\n";
            }

            descriptorsFile.close();

            // Add results to image and save.
            cv::imwrite("/root/Hausaufgabe3/data/images/sift_result.jpg", output);

            // Change depending on which keypoints are good features
            std::vector<int> selectedIndices = {1024, 888, 690, 1399, 1517, 151, 2272, 1305, 1375, 1337};

            // Extract the corresponding keypoints
            std::vector<cv::KeyPoint> selectedKeypoints;
            for (int idx : selectedIndices)
            {
                if (idx < this->keypoints.size())
                {
                    selectedKeypoints.push_back(this->keypoints[idx-1]);
                }
            }

            // Save selected keypoints to a CSV file
            std::ofstream preprocessing_keypoints;
            preprocessing_keypoints.open("/root/Hausaufgabe3/data/csvs_for_debugging/preprocessing_keypoints.csv");
            preprocessing_keypoints << "Number, x, y, size, angle, response\n";
            for (int i = 0; i < selectedKeypoints.size(); ++i)
            {
                const cv::KeyPoint &kpt = selectedKeypoints[i];
                preprocessing_keypoints << (i + 1) << "," << kpt.pt.x << "," << kpt.pt.y << ","
                            << kpt.size << "," << kpt.angle << "," << kpt.response << "\n";
            }
            preprocessing_keypoints.close();

            std::ofstream activeSetFile;
            activeSetFile.open("/root/Hausaufgabe3/data/activeSet.csv");
            activeSetFile << "ID\n";
            for (int idx : selectedIndices)
            {
                activeSetFile << idx << "\n";
            }
            activeSetFile.close();

            // Draw and save the image with the selected keypoints
            cv::Mat selectedOutput;
            cv::drawKeypoints(this->inputImage, selectedKeypoints, selectedOutput);

            // Number the selected features
            for (int i = 0; i < selectedKeypoints.size(); ++i)
            {
                std::string number = std::to_string(selectedIndices[i]);
                cv::Point2f point = selectedKeypoints[i].pt;
                cv::putText(selectedOutput, number, point, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                cv::circle(selectedOutput, point, 3, cv::Scalar(255, 0, 0), -1);
                cv::circle(selectedOutput, point, 10, cv::Scalar(0, 255, 0), 2);
            }

            // Save the image with the selected keypoints
            cv::imwrite("/root/Hausaufgabe3/data/images/selected_keypoints.jpg", selectedOutput);

        };

    private:
        cv::Mat inputImage;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;

};


int main(int argc, char* argv[])
{
    cv::Mat image = cv::imread("/root/Hausaufgabe3/data/images/myImage.jpg");

    Sift sift(image);
    sift.getKeypoints();

    return 0;
}