#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void binaryProc(Mat& image);	//ͼ���ֵ��
void colorReverse(Mat& image);	//��ɫ��ת
int getPixelSum(Mat& image);	//����������ص��
int imgMatch(Mat& image, int& rate, int& num);	//ģ��ƥ��

int main() {
	Mat src = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/photo.jpg");	//��ȡͼƬ

	//����ͷ��ʼ��
	VideoCapture capture(0);	//����VideoCapture��
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//��ȡ����ͷ�Ŀ��ߡ�֡����FPS
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);

	
	//����ͷ��ȡ
	Mat frame;
	while (capture.isOpened()) {
		capture.read(frame);	//��֡��ȡ��Ƶ
		if (frame.empty()) {	//�����Ƶ������δ��⵽����ͷ������ѭ��
			break;
		}

		Mat grayImg = frame.clone();
		binaryProc(grayImg);	//��ֵ������
		colorReverse(grayImg);	//��ɫ��ת

		vector<vector<Point>> contours;  //���������Ͳ�νṹ
		vector<Vec4i> hierarchy;
		findContours(grayImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE); //Ѱ������
		int i = 0;
		Point2f pp[5][4];   //����㼯
		vector<vector<Point>>::iterator It;
		Rect rect[15];
		for (It = contours.begin(); It < contours.end(); It++) {                        //������Χ���ֵ���С����
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
				cout << "ʶ�����֣�" << matchingNum << "\t\tƥ��ȣ�" << matchingRate << endl;
				//imwrite(to_string(matchingNum) + ".jpg", num[j]);
			}		
		}


		imshow("Video", frame);
		char k = waitKey(33);	//��֡��ȡ�ļ��ʱ��
		if (k == 'q') {			//����q���˳�ѭ��
			break;
		}
	}
	capture.release();			//�ͷ���Ƶ
	

	waitKey(0);
	destroyAllWindows();;
	return 0;
}


//ͼ���ֵ��
void binaryProc(Mat& image) {
	cvtColor(image, image, COLOR_BGR2GRAY);	//�Ҷȴ���
	threshold(image, image, 100, 255, THRESH_BINARY);	//��ֵ��������ֵ100����Χ255
}


//��ɫ��ת
void colorReverse(Mat& image) {
	for (int row = 0; row < image.rows; row++) {
		uchar* current_pixel = image.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			*current_pixel++ = 255 - *current_pixel;	//ָ��������ص㷴ת��ɫ
		}
	}
}


//��ȡ�������ص��
int getPixelSum(Mat& image){
	int a = 0;
	for (int row = 0; row < image.rows; row++) {
		uchar* current_pixel = image.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			a += *current_pixel++;	//ָ��������ص㷴ת��ɫ
		}
	}
	return a;
}


//ģ��ƥ��
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