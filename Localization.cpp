#include "localization.h"


//---------------- Name space ---------------------------------------
using namespace cv;
using namespace std;

//---------------- Definition ---------------------------------------
// Constructor
Localization::Localization(Mat gray, Mat histimage){
    this->grayimag = gray;
    this->histimage = histimage;

}

// Destructor
Localization::~Localization(){
}

void Localization::Canny() {
    namedWindow("Canny image", cv::WINDOW_OPENGL); // Create Window
    cv::Canny(this->grayimag,            // input image
              this->cannyimage,                    // output image
              50,                        // low threshold
              10,3);                        // high threshold
    imshow( "Canny image", this->cannyimage );
    cv::waitKey(0);
}

void Localization::distancetransform() {
    Mat kernel = (Mat_<float>(3,3) <<
                                   1,  1, 1,
            1, -8, 1,
            1,  1, 1);
    Mat imgLaplacian;
    filter2D(this->histimage, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    this->histimage.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow( "New Sharped Image", imgResult );
    // Perform the distance transform algorithm
    distanceTransform(this->cannyimage, this->dist, DIST_L2, 3);
    // Normalize the this->distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(this->dist, this->dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", this->dist);
    waitKey();
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    cv::threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
}

void Localization::dilation() {
    Mat kernel1 = Mat::ones(19, 19, CV_8U);
    dilate(this->dist, this->dist, kernel1);
    imshow("Peaks", this->dist);
    waitKey();
}

vector<vector<Point>> Localization::findcontours() {
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    this->dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, this->hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    this->markers = Mat::zeros(this->dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(this->markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    circle(this->markers, Point(5,5), 3, Scalar(255), -1);
    Mat markers8u;
    this->markers.convertTo(markers8u, CV_8U, 10);
    return contours;
}