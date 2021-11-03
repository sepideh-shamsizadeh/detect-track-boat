

#ifndef HISTOGRAM_PREPROCESSING_H
#define HISTOGRAM_PREPROCESSING_H

#endif //HISTOGRAM_PREPROCESSING_H
//---------------- Head  File ---------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
//---------------- Name space ---------------------------------------
using namespace cv;
using namespace std;

class Preprocessing{
public:
    Mat img;
    Mat blurrimag;
    Mat histimage;
    vector<Mat> bgr_planes;
    vector<Mat> hists;

public:
    Preprocessing(Mat image);													// Constructor
    ~Preprocessing();													// Destructor
    Mat Blurred();
    Mat Equalization();
    vector<Mat> CalculateEqualizedHistogram();
    vector<Mat> CalculateHistogram();
};


