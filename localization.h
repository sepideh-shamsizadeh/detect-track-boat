//
// Created by sepideh on 23/07/21.
//

#ifndef HISTOGRAM_LOCALIZATION_H
#define HISTOGRAM_LOCALIZATION_H

#endif //HISTOGRAM_LOCALIZATION_H



//---------------- Head  File ---------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
//---------------- Name space ---------------------------------------
using namespace cv;
using namespace std;

class Localization{
public:
    Mat grayimag;
    Mat cannyimage;
    Mat histimage;
    Mat dist;
    Mat markers;
    vector<Vec4i> hierarchy;

public:
    Localization(Mat gray, Mat histimage);													// Constructor
    ~Localization();													// Destructor
    void distancetransform();
    void Canny();
    void dilation();
    vector<vector<Point> > findcontours();
};
