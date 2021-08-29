#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "myfunc.h"

using namespace cv;
using namespace std;

int main() {
	Mat src = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/photo.jpg");	//��ȡͼƬ
	DigitalRec dr;

	//����ͷ��ʼ��
	VideoCapture capture(0);	//����VideoCapture��
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//��ȡ����ͷ�Ŀ��ߡ�֡����FPS
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);

	
	//����ͷ��ȡ
	Mat frame;
	while (capture.isOpened()) {
		capture.read(frame);	//��֡��ȡ��Ƶ
		flip(frame, frame, 1);	//����ȡ����Ƶ���ҷ�ת
		if (frame.empty()) {	//�����Ƶ������δ��⵽����ͷ������ѭ��
			break;
		}
		imshow("Video", frame);	//ÿ��ѭ����ʾһ֡ͼ��


		//��ֵ������
		Mat grayImg = dr.binaryProc(frame);

		//��ɫ��ת
		dr.colorReverse(grayImg);
		imshow("GrayVideo", grayImg);	//ÿ��ѭ����ʾһ֡ͼ��

		//�����и�
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


		char k = waitKey(33);	//��֡��ȡ�ļ��ʱ��
		if (k == 'q') {			//����q���˳�ѭ��
			break;
		}
	}
	capture.release();			//�ͷ���Ƶ



	/*
	
	*/
	
	
	

	waitKey(0);
	destroyAllWindows();;
	return 0;
}