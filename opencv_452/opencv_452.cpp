#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>


using namespace cv;
using namespace std;

void colorReverse(Mat& image);	//颜色反转
int getColSum(Mat src, int col);//获得列像素点和
int getRowSum(Mat src, int row);//获得行像素点和
int getPixelSum(Mat& image);	//获得所有像素点和
int imgMatch(Mat& image, int& rate, int& num);		//模板匹配
int cutLeft(Mat& src, Mat& leftImg, Mat& rightImg);	//左右切割
int cutTop(Mat& src, Mat& dstImg, Mat& bottomImg);	//上下切割
int getSubtract(Mat& src);		//两张图片相减


int main()
{

	Mat src = imread("E:/Program/OpenCV/vcworkspaces/ogr_test/images/txt.jpg",IMREAD_GRAYSCALE);
    Mat grayImage;					//定义Mat对象用于存储每一帧数据

    //cvtColor(src, grayImage, COLOR_BGR2GRAY);                  //转换灰度图
    threshold(src, grayImage, 100, 255, THRESH_BINARY_INV);     //转换二值图，设置阈值，高于50认为255
	//colorReverse(grayImage);

	imshow("grayimg", grayImage);

    Mat leftImg, rightImg, topImg, bottomImg;
    int topRes = cutTop(grayImage, topImg, bottomImg);
    int matchNum = -1, matchRate = 10e6;
	while (topRes == 0)
    {
        int leftRes = cutLeft(topImg, leftImg, rightImg);
        while (leftRes == 0) {
            imgMatch(leftImg, matchNum, matchRate);//数字识别
			//getSubtract(topImg);
            imshow("num", leftImg);
            if (matchRate < 300000) {
                cout << "识别数字：" << matchNum << "\t\t匹配度：" << matchRate << endl;
                //imwrite(to_string(matchingNum) + ".jpg", num[j]);
            }
			Mat srcTmp = rightImg.clone();
            leftRes = cutLeft(srcTmp, leftImg, rightImg);
        }
        Mat srcTmp = bottomImg.clone();
        topRes = cutTop(srcTmp, topImg, bottomImg);
    }

    



	waitKey(0);
	destroyAllWindows();;
	return 0;
}


//颜色反转
void colorReverse(Mat& image) {
	for (int row = 0; row < image.rows; row++) {
		uchar* current_pixel = image.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			*current_pixel++ = 255 - *current_pixel;	//指针遍历像素点反转颜色
		}
	}
}

//获得列像素和
int getColSum(Mat src, int col)
{
	int sum = 0;
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
	{
		sum = sum + src.at <uchar>(i, col);
	}
	return sum;
}

//获得行像素和
int getRowSum(Mat src, int row)
{
	int sum = 0;
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < width; i++)
	{
		sum += src.at <uchar>(row, i);
	}
	return sum;
}

//获取所有像素点和
int getPixelSum(Mat& image) {
	int a = 0;
	for (int row = 0; row < image.rows; row++) {
		uchar* current_pixel = image.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			a += *current_pixel++;	//指针遍历像素点反转颜色
		}
	}
	return a;
}


//模板匹配
int imgMatch(Mat& image, int& rate, int& num) {
	Mat imgSub;
	double min = 10e6;
	num = 0;
	rate = 0;

	for (int i = 0; i < 10; i++) {
		Mat templatimg = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/" + std::to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		resize(image, image, Size(64, 108), 0, 0, cv::INTER_LINEAR);
		resize(templatimg, templatimg, Size(64, 108), 0, 0, cv::INTER_LINEAR);
		absdiff(templatimg, image, imgSub);
		rate = getPixelSum(imgSub);
		if (rate < min) {
			min = rate;
			num = i;
		}
	}
	return num;
}

//数字切割
int cutLeft(Mat& src, Mat& leftImg, Mat& rightImg)
{
	int left = 0, right = src.cols;

	int i;
	for (i = 0; i < src.cols; i++)
	{
		int colValue = getColSum(src, i);
		//cout <<i<<" th "<< colValue << endl;
		if (colValue > 0)
		{
			left = i;
			break;
		}
	}
	if (left == 0)
	{
		return 1;
	}


	for (; i < src.cols; i++)
	{
		int colValue = getColSum(src, i);
		//cout << i << " th " << colValue << endl;
		if (colValue == 0)
		{
			right = i;
			break;
		}
	}
	int width = right - left;
	Rect rectLeft(left, 0, width, src.rows);
	leftImg = src(rectLeft).clone();
	Rect rectRight(right, 0, src.cols - right, src.rows);
	rightImg = src(rectRight).clone();
	return 0;
}

int cutTop(Mat& src, Mat& topImg, Mat& bottomImg)//上下切割
{
	int top = 0, bottom = src.rows;

	int i;
	for (i = 0; i < src.rows; i++)
	{
		int colValue = getRowSum(src, i);
		//cout <<i<<" th "<< colValue << endl;
		if (colValue > 0)
		{
			top = i;
			break;
		}
	}
	if (top == 0)
	{
		return 1;
	}

	for (; i < src.rows; i++)
	{
		int colValue = getRowSum(src, i);
		//cout << i << " th " << colValue << endl;
		if (colValue == 0)
		{
			bottom = i;
			break;
		}
	}
	int height = bottom - top;
	Rect rectTop(0, top, src.cols, height);
	topImg = src(rectTop).clone();
	Rect rectBottom(0, bottom, src.cols, src.rows - bottom);
	bottomImg = src(rectBottom).clone();
	return 0;
}


//图像相减
int getSubtract(Mat& src) //两张图片相减
{
	Mat img_result;
	int min = 1000000;
	int serieNum = 0;
	for (int i = 0; i < 10; i++) {
		Mat Template = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/" + std::to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		threshold(Template, Template, 100, 255, THRESH_BINARY);
		threshold(src, src, 100, 255, THRESH_BINARY);
		resize(src, src, Size(32, 48), 0, 0, INTER_LINEAR);
		resize(Template, Template, Size(32, 48), 0, 0, INTER_LINEAR);//调整尺寸		
		//imshow(name, Template);
		absdiff(Template, src, img_result);//
		int diff = getPixelSum(img_result);
		if (diff < min)
		{
			min = diff;
			serieNum = i;
		}
	}
	printf("最小距离是%d ", min);
	printf("匹配到第%d个模板匹配的数字是%d\n", serieNum, serieNum);
	return serieNum;
}