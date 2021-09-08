#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

class MyDemo {
public:
	void colorSpace_Demo(Mat& image);
	void matCreation_Demo();
	void pixelVisit_Demo(Mat& image);
	void operators_Demo(Mat& image);
	void bitWise_Demo(Mat& image);
	void checkBar_Demo(Mat& image);
	void key_Demo(Mat& image);
	void colorStyle_Demo(Mat& image);
	void channels_Demo(Mat& image);
	void inRange_Demo(Mat& image);
	void pixelStatistic_Demo(Mat& image);
	void drawing_Demo(Mat& image);
	void random_Demo();
	void polyDrawing_Demo();
	void mouseDrawing_Demo(Mat& image);
	void normalize_Demo(Mat& image);
	void resize_Demo(Mat& image);
	void flip_Demo(Mat& image);
	void rotate_Demo(Mat& image);
	void video_Demo(Mat& image);
	void histShow_Demo(Mat& image);
	void histShow2_Demo(Mat& image);
	void blur_Demo(Mat& image);
	void gaussianBlur_Demo(Mat& image);
	void faceDetector_Demo();
};