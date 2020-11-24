#ifndef RETINAFACE
#define RETINAFACE
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "net.h"

using namespace std;
using namespace cv;


struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};
int detect_retinaface(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects, AAssetManager *mgr);
void draw_faceobjects(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects);

#endif