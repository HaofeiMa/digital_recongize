#include "myfunc.h"
#include <opencv2/dnn.hpp>

//using namespace cv;
//using namespace std;


//ͼ���ֵ��
Mat DigitalRec::binaryProc(Mat& image) {
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);	//�Ҷȴ���
	threshold(grayImage, grayImage, 100, 255, THRESH_BINARY);	//��ֵ��������ֵ100����Χ255
	return grayImage;
}

//��ɫ��ת
void DigitalRec::colorReverse(Mat& image) {
	for (int row = 0; row < image.rows; row++) {
		uchar* current_pixel = image.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			*current_pixel++ = 255 - *current_pixel;	//ָ��������ص㷴ת��ɫ
		}
	}
}


//��������غ�
int getColSum(Mat image, int col)
{
	int sum = 0;
	int height = image.rows;
	int width = image.cols;
	for (int i = 0; i < height; i++)
	{
		sum = sum + image.at <uchar>(i, col);
	}
	return sum;
}


//��������غ�
int getRowSum(Mat image, int row)
{
	int sum = 0;
	int height = image.rows;
	int width = image.cols;
	for (int i = 0; i < width; i++)
	{
		sum += image.at <uchar>(row, i);
	}
	return sum;
}


//��ȡ�������ص��
int getPixelSum(Mat& image)
{
	int a = 0;
	for (int row = 0; row < image.rows; row++) {
		uchar* current_pixel = image.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			a+=*current_pixel++;	//ָ��������ص㷴ת��ɫ
		}
	}
	return a;
}


//�и���������
int DigitalRec::cutRows(Mat& image, Mat& topImg, Mat& bottomImg) {
	int top = 0;
	int bottom = image.rows;
	int i = 0;

	do {
		//����ϱ߽�
		for (; i < image.rows; i++) {
			if (getRowSum(image, i) > 0) {
				top = i;
				break;
			}
		}
		if (top == 0) {
			return 1;
		}

		//����±߽�
		for (; i < image.rows; i++) {
			if (getRowSum(image, i) == 0) {
				bottom = i;
				break;
			}
		}
	} while (bottom - top < 80);

	//�и�ͼ��
	Rect rectTop(0, top, image.cols, bottom - top);
	topImg = image(rectTop).clone();
	Rect rectBottom(0, bottom, image.cols, image.rows - bottom);
	bottomImg = image(rectBottom).clone();
	return 0;
}


//�и���������
int DigitalRec::cutCols(Mat& image, Mat& leftImg, Mat& rightImg) {
	int left = 0;
	int right = image.cols;
	int i = 0;

	do {
		//�����߽�
		for (; i < image.cols; i++) {
			if (getColSum(image, i) > 0) {
				left = i;
				break;
			}
		}
		if (left == 0) {
			return 1;
		}

		//����ұ߽�
		for (; i < image.cols; i++) {
			if (getColSum(image, i) == 0) {
				right = i;
				break;
			}
		}
	} while (right - left < 50);

	//�и�ͼ��
	Rect rectLeft(left, 0, right - left, image.rows);
	leftImg = image(rectLeft).clone();
	Rect rectRight(right, 0, image.cols - right, image.rows);
	rightImg = image(rectRight).clone();
	return 0;
}


//ͼƬƥ��
int DigitalRec::imgMatch(Mat& image) {
	Mat imgSub;
	int min = 10e6;
	int num = 0;
	
	for (int i = 0; i < 10; i++) {
		Mat templatimg = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/" + std::to_string(i) + ".jpg",IMREAD_GRAYSCALE);
		resize(image, image, Size(32, 48), 0, 0, cv::INTER_LINEAR);
		resize(templatimg, templatimg, Size(32, 48), 0, 0, cv::INTER_LINEAR);
		absdiff(templatimg, image, imgSub);
		int pixelSum = getPixelSum(imgSub);
		if (pixelSum < min) {
			min = pixelSum;
			num = i;
		}
	}
	//std::cout << "ƥ�䣺" << num << "\tdiff��" << pixelSum << std::endl;
	return num;
}