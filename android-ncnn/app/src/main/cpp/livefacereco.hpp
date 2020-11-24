//
// Created by Xinghao Chen 2020/7/27
//
#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include "stdlib.h"
#include <iostream>
#include <array>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "arcface.h"

#define PI 3.14159265
using namespace std;
using namespace cv;

// Adjustable Parameters
const bool largest_face_only=true;
const bool record_face=false;
const int distance_threshold = 90;
const float face_thre=0.40;
const float true_thre=0.89;
const int jump=10;
const int input_width = 320;
const int input_height = 240;
const int output_width = 320;
const int output_height = 240;
const string project_path="";
//end

const cv::Size frame_size = Size(output_width,output_height);
const float ratio_x = (float)output_width/ input_width;
const float ratio_y = (float)output_height/ input_height;

Mat Zscore(const Mat &fc);
inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2) {
    double dot = v1.dot(v2);  // <v1, v2>
    double denom_v1 = norm(v1);  // |v1|
    double denom_v2 = norm(v2);  // |v2|
    return dot / (denom_v1 * denom_v2);
}

/**
 * Calculating the turning angle of face
 *  */
inline double count_angle(float landmark[5][2]) {
    double a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2;
    double b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2;
    double angle = atan(abs(b) / a) * 180 / PI;
    return angle;
}


/**
 * Formatting output structure
 */
inline cv::Mat draw_conclucion(String intro, double input, cv::Mat result_cnn, int position) {
    char string[10];
    sprintf(string, "%.2f", input);
    std::string introString(intro);
    introString += string;
    putText(result_cnn, introString, cv::Point(5, position), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255),2);
    return result_cnn;
}
int MTCNNDetection();


