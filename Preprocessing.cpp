#include "preprocessing.h"


//---------------- Name space ---------------------------------------
using namespace cv;
using namespace std;

//---------------- Definition ---------------------------------------


// Constructor
Preprocessing::Preprocessing(Mat image){
    this->img = image;
}

// Destructor
Preprocessing::~Preprocessing(){
}

Mat Preprocessing::Blurred() {
    cv::medianBlur(this->histimage,
                   this->blurrimag,
                   21

    );
    cv::GaussianBlur(this->blurrimag,            // input image
                     this->blurrimag,                            // output image
                     cv::Size(21, 21),                        // smoothing window width and height in pixels
                     2.5,
                     0,
                     cv::BORDER_DEFAULT);
    return this->blurrimag;
}



std::vector<cv::Mat>  Preprocessing::CalculateHistogram()
{
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    std::vector<cv::Mat>  hist;
    split( this->img, this->bgr_planes );
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist( &this->bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &this->bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &this->bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    hist.push_back(b_hist);
    hist.push_back(g_hist);
    hist.push_back(r_hist);
    return bgr_planes;
}
vector<cv::Mat> Preprocessing::CalculateEqualizedHistogram()
{
    std::vector<cv::Mat> eqhist;
    cv::Mat b, g, r;
    cv::equalizeHist(this->bgr_planes[0], b);
    cv::equalizeHist(this->bgr_planes[1], g);
    cv::equalizeHist(this->bgr_planes[2], r);
    eqhist.push_back(b);
    eqhist.push_back(g);
    eqhist.push_back(r);
    return eqhist;
}


Mat Preprocessing::Equalization() {
    std::vector<cv::Mat> bgr_planes = CalculateHistogram();
    cv::merge(bgr_planes, this->histimage);
    namedWindow("Original Image Histogram", cv::WINDOW_AUTOSIZE); // Create Window
    cv::imshow("Original Image Histogram", this->histimage);
    cv::waitKey(0);

    std::vector<cv::Mat> eqhist = CalculateEqualizedHistogram();
    cv::merge(eqhist, this->histimage);
    namedWindow("Equalized Original Image", cv::WINDOW_AUTOSIZE); // Create Window
    cv::imshow("Equalized Original Image", this->histimage);
    CalculateHistogram();
    cv::waitKey(0);
    return this->histimage;
}