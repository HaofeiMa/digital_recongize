#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "myfunc.h"

using namespace cv;
using namespace std;

int main() {
	Mat src = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/photo.jpg");	//读取图片
	DigitalRec dr;

	//摄像头初始化
	VideoCapture capture(0);	//创建VideoCapture类
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//获取摄像头的宽、高、帧数、FPS
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);

	
	//摄像头读取
	Mat frame;
	while (capture.isOpened()) {
		capture.read(frame);	//逐帧读取视频
		flip(frame, frame, 1);	//将读取的视频左右反转
		if (frame.empty()) {	//如果视频结束或未检测到摄像头则跳出循环
			break;
		}
		imshow("Video", frame);	//每次循环显示一帧图像


		//二值化处理
		Mat grayImg = dr.binaryProc(frame);

		//颜色反转
		dr.colorReverse(grayImg);
		imshow("GrayVideo", grayImg);	//每次循环显示一帧图像

		//区域切割
		Mat topImg, bottomImg;
		int rowFlag = dr.cutRows(grayImg, topImg, bottomImg);
		for (int i = 0; rowFlag == 0; i++) {
			Mat leftImg, rightImg;
			int colFlag = dr.cutCols(topImg, leftImg, rightImg);
			for (int j = 0; colFlag == 0; j++) {
				imshow(to_string(i * 10 + j), leftImg);
				cout << dr.imgMatch(leftImg) << endl;
				//imwrite(to_string(i*10+j) + ".jpg", leftImg);
				Mat imageTemp = rightImg;
				colFlag = dr.cutCols(imageTemp, leftImg, rightImg);
			}
			Mat imageTemp = bottomImg;
			rowFlag = dr.cutRows(imageTemp, topImg, bottomImg);
		}


		char k = waitKey(33);	//两帧读取的间隔时间
		if (k == 'q') {			//按下q键退出循环
			break;
		}
	}
	capture.release();			//释放视频



	/*
	
	*/
	
	
	

	waitKey(0);
	destroyAllWindows();;
	return 0;
}