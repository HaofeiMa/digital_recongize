#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void binaryProc(Mat& image);	//图像二值化
void colorReverse(Mat& image);	//颜色反转
int getPixelSum(Mat& image);	//获得所有像素点和
int imgMatch(Mat& image, int& rate, int& num);	//模板匹配

int main() {
	Mat src = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/photo.jpg");	//读取图片

	//摄像头初始化
	VideoCapture capture(0);	//创建VideoCapture类
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//获取摄像头的宽、高、帧数、FPS
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);

	
	//摄像头读取
	Mat frame;
	while (capture.isOpened()) {
		capture.read(frame);	//逐帧读取视频
		if (frame.empty()) {	//如果视频结束或未检测到摄像头则跳出循环
			break;
		}

		Mat grayImg = frame.clone();
		binaryProc(grayImg);	//二值化处理
		colorReverse(grayImg);	//颜色反转

		vector<vector<Point>> contours;  //定义轮廓和层次结构
		vector<Vec4i> hierarchy;
		findContours(grayImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE); //寻找轮廓
		int i = 0;
		Point2f pp[5][4];   //定义点集
		vector<vector<Point>>::iterator It;
		Rect rect[15];
		for (It = contours.begin(); It < contours.end(); It++) {                        //画出包围数字的最小矩形
			rect[i].x = (float)boundingRect(*It).tl().x;
			rect[i].y = (float)boundingRect(*It).tl().y;
			rect[i].width = (float)boundingRect(*It).br().x - (float)boundingRect(*It).tl().x;
			rect[i].height = (float)boundingRect(*It).br().y - (float)boundingRect(*It).tl().y;
			if ((rect[i].height > 80) && (rect[i].width > 50)&& (rect[i].height < 300) && (rect[i].width < 300)) {
				rectangle(frame, rect[i], Scalar(0, 0, 255), 2, 8, 0);
				rectangle(grayImg, rect[i], Scalar(0, 0, 0), 0, 8, 0);
				i++;
			}
		}


		Mat num[15];
		int matchingNum = 0;
		int matchingRate = 0;
		for (int j = 0; j < i; j++) {
			grayImg(rect[j]).copyTo(num[j]);
			imshow("num", num[j]);
			imgMatch(num[j], matchingRate, matchingNum);
			if (matchingRate < 200000) {
				cout << "识别数字：" << matchingNum << "\t\t匹配度：" << matchingRate << endl;
				//imwrite(to_string(matchingNum) + ".jpg", num[j]);
			}		
		}


		imshow("Video", frame);
		char k = waitKey(33);	//两帧读取的间隔时间
		if (k == 'q') {			//按下q键退出循环
			break;
		}
	}
	capture.release();			//释放视频
	

	waitKey(0);
	destroyAllWindows();;
	return 0;
}


//图像二值化
void binaryProc(Mat& image) {
	cvtColor(image, image, COLOR_BGR2GRAY);	//灰度处理
	threshold(image, image, 100, 255, THRESH_BINARY);	//二值化处理，阈值100，范围255
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


//获取所有像素点和
int getPixelSum(Mat& image){
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
	int min = 10e6;
	num = 0;
	rate = 0;

	for (int i = 0; i < 10; i++) {
		Mat templatimg = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/" + std::to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		resize(image, image, Size(32, 48), 0, 0, cv::INTER_LINEAR);
		resize(templatimg, templatimg, Size(32, 48), 0, 0, cv::INTER_LINEAR);
		absdiff(templatimg, image, imgSub);
		rate = getPixelSum(imgSub);
		if (rate < min) {
			min = rate;
			num = i;
		}
	}
}