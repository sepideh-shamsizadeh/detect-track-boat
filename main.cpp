#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include "MeanShift.cpp"
#include "Preprocessing.cpp"
#include "Localization.cpp"


using namespace cv;
using namespace std;
using namespace cv::dnn;

String nn_test_path = "/home/sepideh/workspace/ComputerVision/boatTracking/pythonProject/frozen_models/simple_frozen_graph.pb";
RNG rng(12345);

double theta,rho;
cv::Mat imgGrayscale,imgBlurred,imgCanny,cannycolor;
cv::Mat img_final;
std::vector<cv::Vec2f> lines;
String nn_path;
Net model;
vector<string> class_names;
void showHistogram(std::vector<cv::Mat>& hists)
{
    // Min/Max computation
    double hmax[3] = {0,0,0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    std::string wname[3] = { "blue", "green", "red" };
    cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                             cv::Scalar(0,0,255) };

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
        {
            cv::line(
                    canvas[i],
                    cv::Point(j, rows),
                    cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
                    hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
                    1, 8, 0
            );
        }
        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}


Mat segment_meanshift(Mat image, bool preview=false){
    cvtColor(image, image, COLOR_RGB2Lab);

    MeanShift MSProc(8, 16);
    MSProc.MSSegmentation(image);

    cout<<"the Spatial Bandwith is "<<MSProc.hs<<endl;
    cout<<"the Color Bandwith is "<<MSProc.hr<<endl;

    cvtColor(image, image, COLOR_Lab2RGB);
    if(preview==true){
        imshow("MS Picture", image);
        waitKey();
    }
    return image;
}



void predict(Mat proposal,Net model, Mat image,Point pt1, Point pt2){
    Mat blob = blobFromImage(proposal, 0.1, Size(96, 96
    ), Scalar(127, 127, 127));
    model.setInput(blob);
    Mat outputs = model.forward();
    Point classIdPoint;
    double final_prob;
    minMaxLoc(outputs.reshape(1, 1), 0, &final_prob, 0, &classIdPoint);
    int label_id = classIdPoint.x;

    // put the class name text on top of the image
    String ss=class_names[label_id].c_str();
    int res = ss.compare("boat");
    final_prob= 1-final_prob;

if(final_prob>=0.6) {
    string out_text = format("boat");
    cout << final_prob << endl;
//            imshow("proposal", proposal);
//            waitKey();
    putText(image, out_text, Point(pt1.x + 25, pt1.y + 34), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
}




}

float intersection_over_union(Rect rect1, Rect rect2){
    int rect1_x_min = rect1.x;
    int rect1_x_max = rect1.x + rect1.width;
    int rect1_y_min = rect1.y;
    int rect1_y_max = rect1.y + rect1.height;

    int rect2_x_min = rect2.x;
    int rect2_x_max = rect2.x + rect2.width;
    int rect2_y_min = rect2.y;
    int rect2_y_max = rect2.y + rect2.height;

    if(rect1_x_max < rect2_x_min || rect2_x_max < rect1_x_min || rect1_y_max < rect2_y_min || rect2_y_max < rect1_y_min){
        return 0;
    }else{
        float x1_inner = max(rect1_x_min, rect2_x_min);
        float y1_inner = max(rect1_y_min, rect2_y_min);
        float x2_inner = min(rect1_x_max, rect2_x_max);
        float y2_inner = min(rect1_y_max, rect2_y_max);

        float inner_area = (y2_inner - y1_inner) * (x2_inner - x1_inner);

        float rect1_area = (rect1_y_max - rect1_y_min) * (rect1_x_max - rect1_x_min);
        float rect2_area = (rect2_y_max - rect2_y_min) * (rect2_x_max - rect2_x_min);
        float union_area = rect1_area + rect2_area - inner_area;
        cout<<"rect1_area:"<<rect1_area<<"rect2_area:"<<rect2_area;
        cout<<"inner_area:"<<inner_area<<"union_area:"<<union_area;
        return inner_area / union_area;
    }
}

int main() {
    cv::Mat image= cv::imread("../test/00.png");
    imshow("orginal image",image);
    waitKey();
    Preprocessing process = Preprocessing(image);
    Mat hist, blur;
    hist = process.Equalization();
    blur = process.Blurred();
    imshow("Blurred image", blur);
    waitKey();
    cv::cvtColor(blur, imgGrayscale, cv::COLOR_BGR2GRAY);// convert to grayscale
    Localization local= Localization(imgGrayscale, hist);
    local.Canny();
    local.distancetransform();
    local.dilation();
    vector<vector<Point>> contours =  local.findcontours();
    //    Mat image_segment = segment_meanshift(imgBlurred, false);
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(local.markers.size(), CV_8UC3);

    // Fill labeled objects with random colors
    for (int i = 0; i < local.markers.rows; i++)
    {
        for (int j = 0; j < local.markers.cols; j++)
        {
            int index = local.markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i,j) = colors[index-1];

            }
        }

    }
    model = readNetFromTensorflow(nn_test_path);
    class_names = {"boat", "not"};
    vector<Mat> proposals;
    for(size_t i = 0; i< contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(dst, contours, (int) i, color, 2, LINE_8, local.hierarchy, 0);
        if (contours[i].size() > 10) {
            int x_min = image.size().width, y_min = image.size().height, x_max = 0, y_max = 0;
            for (int j = 0; j < contours[i].size(); j++) {
                if (contours[i][j].x <= x_min)
                    x_min = contours[i][j].x;
                if (contours[i][j].x >= x_max)
                    x_max = contours[i][j].x;

                if (contours[i][j].y <= y_min)
                    y_min = contours[i][j].y;
                if (contours[i][j].y >= y_max)
                    y_max = contours[i][j].y;
            }
            Point pt1(x_min, y_min);
            Point pt2(x_max, y_max);
            cv::Rect myROI(x_min, y_min, x_max-x_min, y_max-y_min);
            cv::Mat croppedImage = image(myROI);
            predict(croppedImage,model,image,pt1,pt2);
        }
    }

    for (int j = 0; j+64 < image.rows; j += 64)
        for (int i = 0; i+64 < image.cols; i += 64)
        {
            {
                cv::Rect myROI(i, j, 64, 64);
                cv::Mat croppedImage = blur(myROI);
                Point pt1(i, j);
                Point pt2(i+64, j+64);
                predict(croppedImage,model,image,pt1,pt2);
            }
        }

    // Visualize the final image
    imshow("Final Result", dst);
    imshow("FinResult", image);
    waitKey();
    return 0;
}
