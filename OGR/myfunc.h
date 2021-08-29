#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

class DigitalRec {
public:
	Mat binaryProc(Mat& image);
	void colorReverse(Mat& image);
	int cutRows(Mat& image, Mat& topImg, Mat& bottomImg);
	int cutCols(Mat& image, Mat& leftImg, Mat& rightImg);
	int imgMatch(Mat& image);
};