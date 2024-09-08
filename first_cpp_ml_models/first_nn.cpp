#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
 
#include <iostream>
 
using namespace cv;
 
int main() {
    std::string image_path = samples::findFile("/Users/justicecalderon/Desktop/swe-portfolio/justice-educative-courses/ml-engineering/ml-projects/first_cpp_ml_models/justice.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
 
    if(img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    std::cout << img.size() << std::endl;
 
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
 
    if(k == 's') {
        imwrite("/Users/justicecalderon/Desktop/swe-portfolio/justice-educative-courses/ml-engineering/ml-projects/first_cpp_ml_models/justice.jpg", img);
    }
 
    return 0;
}