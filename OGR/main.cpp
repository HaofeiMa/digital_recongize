#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>

using namespace cv;
using namespace std;


void binaryProc(Mat& image);	//ͼ���ֵ��
void morphTreat(Mat& image);	//��̬ѧ����
void colorReverse(Mat& image);	//��ɫ��ת
int getColSum(Mat src, int col);//��������ص��
int getRowSum(Mat src, int row);//��������ص��
int getPixelSum(Mat& image);	//����������ص��
int imgMatch(Mat& image, int& rate, int& num);		//ģ��ƥ��
int cutLeft(Mat& src, Mat& leftImg, Mat& rightImg);	//�����и�
int cutTop(Mat& src, Mat& dstImg, Mat& bottomImg);	//�����и�
int getSubtract(Mat& src);		//����ͼƬ���
void rotate_Demo(Mat& image, double angle);	//ͼ����ת


int main() {
	//����ͷ��ʼ��
	VideoCapture capture(1);	//����VideoCapture�࣬ʹ����������ͷ��1��
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//��ȡ����ͷ�Ŀ��ߡ�֡����FPS
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	

	//����ͷ��ȡ
	Mat frame;
	while (capture.isOpened()) {
		capture.read(frame);	//��֡��ȡ��Ƶ
		if (frame.empty()) {	//�����Ƶ������δ��⵽����ͷ������ѭ��
			break;
		}
		resize(frame, frame, Size(640, 640 * frame.rows / frame.cols));
		Mat binImg;
		frame.copyTo(binImg);  //��ֵ��֮���ͼ��
		binaryProc(binImg);	//��ֵ������
		morphTreat(binImg);			//��̬ѧ����
		imshow("binImg", binImg);


		/************************** ��ȡA4ֽ����ʶ������ *****************************/
		double length, area, rectArea;     //���������ܳ�����������������
		double long2Short = 0.0;           //��̬��=����/�̱�
		Rect rect;           //������
		RotatedRect box, boxTemp;  //��Ӿ���
		CvPoint2D32f pt[4];    //���ζ������
		Mat pts;    //���ζ������
		double axisLong = 0.0, axisShort = 0.0;        //���εĳ��ߺͶ̱�
		double axisLongTemp = 0.0, axisShortTemp = 0.0;//���εĳ��ߺͶ̱�
		double LengthTemp;     //�м����
		float  angle = 0;      //��¼��б�Ƕ�
		float  angleTemp = 0;
		double location_x = 0.0;
		double location_y = 0.0;
		vector<vector<Point>> contours;
		vector<Vec4i>hierarchy;
		findContours(binImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i < contours.size(); i++)
		{
			//������������С��Ӿ���  
			length = arcLength(contours[i], true);      //��ȡ�����ܳ�
			area = contourArea(contours[i]);			//��ȡ�������
			if (area > 2000 && area < 300000)			//�������������С�жϣ����������ļ���
			{
				rect = boundingRect(contours[i]);		//������α߽�
				boxTemp = minAreaRect(contours[i]);		//��ȡ�����ľ���
				boxPoints(boxTemp, pts);				//��ȡ�����ĸ��������꣨���ϣ����ϣ����£����£�
				for (int row = 0; row < pts.rows; row++) {		//���б������ζ����ĸ���������
					pt[row].x = pts.at<uchar>(row, 0);
					pt[row].y = pts.at<uchar>(row, 1);
				}
				angleTemp = boxTemp.angle;              //�õ���б�Ƕ�
				if (angleTemp > 45) {					//������ʱ��ƫת���������б�Ƕ�Ϊ-(90-angle)
					angleTemp = angleTemp - 90;
				}
				
				axisLongTemp = sqrt(pow(pt[1].x - pt[0].x, 2) + pow(pt[1].y - pt[0].y, 2));  //���㳤�ᣨ���ɶ���
				axisShortTemp = sqrt(pow(pt[2].x - pt[1].x, 2) + pow(pt[2].y - pt[1].y, 2)); //������ᣨ���ɶ���


				if (axisShortTemp > axisLongTemp)		//���������ڳ��ᣬ��������
				{
					LengthTemp = axisLongTemp;
					axisLongTemp = axisShortTemp;
					axisShortTemp = LengthTemp;
				}

				rectArea = axisLongTemp * axisShortTemp;	//������ε�ʵ�����

				long2Short = axisLongTemp / axisShortTemp;	//���㳤���
				
				// �����A4ֽΪ1.414�����ó���ȡ���������Ͷ̱߳�����Ϊ��������
				if (long2Short > 1 && long2Short < 1.8  && rectArea > 5000 && rectArea < 300000 && axisShortTemp > 50)
				{
					rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);		//������ͷͼ���л�����������
					if (rect.width > 100 && rect.height > 100 && axisShortTemp>100) {	//��С���η�Χ����������ʶ��
						rect.x += 40;
						rect.y += 40;
						rect.width -= 40;
						rect.height -= 40;
					}
					imshow("Video", frame);					//��ʾ����ͷ���㻭��
					location_x = rect.x + rect.width / 2;	//��þ����������꣬��A4ֽ��������
					location_y = rect.y + rect.height / 2;
					Mat a4Img = frame(rect);				//��ȡA4ֽ����
					Mat a4binImg;
					cvtColor(a4Img, a4binImg, CV_BGR2GRAY);   //��A4ֽ����ת��Ϊ�Ҷ�ͼ��
					threshold(a4binImg, a4binImg, 120, 255, THRESH_BINARY); //�Ҷ�ͼ���ֵ��
					colorReverse(a4binImg);					//��ɫ��ת
					rotate_Demo(a4binImg, angleTemp);		//����ǰ������Ƕȣ���ͼ�������ת����֤����ˮƽ����
					imshow("A4", a4binImg);

					/********************  ����ʶ�𷽷�1��������ȡ��  *********************/
					
					//�ҵ����������������������ƾ���
					vector<vector<Point>> contours_rec;  //���������Ͳ�νṹ
					vector<Vec4i> hierarchy_rec;
					findContours(a4binImg, contours_rec, hierarchy_rec, RETR_EXTERNAL, CHAIN_APPROX_NONE); //Ѱ������
					int i = 0;
					vector<vector<Point>>::iterator It;
					Rect a4rect[15];
					for (It = contours_rec.begin(); It < contours_rec.end(); It++) {                        //������Χ���ֵ���С����
						a4rect[i].x = (float)boundingRect(*It).tl().x;
						a4rect[i].y = (float)boundingRect(*It).tl().y;
						a4rect[i].width = (float)boundingRect(*It).br().x - (float)boundingRect(*It).tl().x;
						a4rect[i].height = (float)boundingRect(*It).br().y - (float)boundingRect(*It).tl().y;
						if ((a4rect[i].height > 80) && (a4rect[i].width > 50) && (a4rect[i].height < 300) && (a4rect[i].width < 300)) {
							rectangle(a4Img, a4rect[i], Scalar(0, 0, 255), 2, 8, 0);
							rectangle(a4binImg, a4rect[i], Scalar(0, 0, 0), 0, 8, 0);
							i++;
						}
					}

					//�������ҵ���������һʶ��
					Mat num[20];
					int output1[3] = {-1,-1,-1};	//�ֱ�洢ʶ�����������
					int output1_flag = 0;
					int matchingNum = 0;
					int matchingRate = 0;
					int matchflag = 0;
					for (int j = 0; j < i; j++) {
						a4binImg(a4rect[j]).copyTo(num[j]);		//��ȡ��Χ���ֵľ���������num[j]
						imshow("num", num[j]);
						imgMatch(num[j], matchingRate, matchingNum);	//����ƥ��
						if (matchingRate < 400000) {
							output1[output1_flag++] = matchingNum;	//��ʶ�𵽵�����˳���������
							if (output1_flag>=3) {		//��ʶ��ȫ����������ʱ��ӡ���
								cout << "ʶ�����֣�" << output1[0] * 100 + output1[1] * 10 + output1[2] << "\t����λ�ã�[" << location_x << ", " << location_y << "]" << endl;
								output1_flag = 0;
								output1[0] = -1;
								output1[1] = -1;
								output1[2] = -1;
								matchflag = 1;
							}
							//imwrite(to_string(matchingNum) + ".jpg", num[j]);
						}
					}
					/*****************************************************/
							


					/********************  ����ʶ�𷽷�1�����з�  *********************/

					//�����һ�ַ���û��ʶ��ɹ�
					if (matchflag == 0) {
						Mat leftImg, rightImg, topImg, bottomImg;
						int topRes = cutTop(a4binImg, topImg, bottomImg);	//����ȡ��A4ֽ��������ɨ�裬���������֮��>550�Ĳ���topImg���Լ�ʣ�ಿ��bottomImg
						int matchNum = -1, matchRate = 10e6;
						int output2[3] = { -1,-1,-1 };	//�ֱ�洢��������
						int output2_flag = 0;
						while (topRes == 0)		//���Դ��������غ�>550�Ĳ���ʱ
						{
							int leftRes = cutLeft(topImg, leftImg, rightImg);	//��������֮��>550�Ĳ���topImg����ɨ�裬���������֮��>550�Ĳ���leftImg���Լ�ʣ�ಿ��rightImg
							while (leftRes == 0) {	//���Դ��������غ�>550�Ĳ���ʱ
								imgMatch(leftImg, matchRate, matchNum);	//����ʶ��
								if (matchRate < 200000) {
									output2[output2_flag++] = matchNum;
									if (output2_flag >= 3) {
										cout << "ʶ�����֣�" << output2[0] * 100 + output2[1] * 10 + output2[2] << "\t����λ�ã�[" << location_x << ", " << location_y << "]" << endl;
										output2_flag = 0;
										output2[0] = -1;
										output2[1] = -1;
										output2[2] = -1;
									}
								}
								Mat srcTmp = rightImg.clone();
								leftRes = cutLeft(srcTmp, leftImg, rightImg);	//��ʣ�ಿ��rightImg��������ɨ��
							}
							Mat srcTmp = bottomImg.clone();
							topRes = cutTop(srcTmp, topImg, bottomImg);			//��ʣ�ಿ��bottomImg��������ɨ��
							output2_flag = 0;
							output2[0] = -1;
							output2[1] = -1;
							output2[2] = -1;
						}
					}
					/*****************************************************/
					

					box = boxTemp;
					angle = angleTemp;
					axisLong = axisLongTemp;
					axisShort = axisShortTemp;
				}
			}
		}

		//imshow("Video", frame);
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
	//��ȡBGR��ͨ�����ֵ
	int w = image.cols;
	int h = image.rows;
	int rmax = 0, gmax = 0, bmax = 0;
	for (int row = 0; row < h; row++) {
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++) {
			if (*current_row > bmax) {
				bmax = *current_row++;
			}
			if (*current_row > gmax) {
				gmax = *current_row++;
			}
			if (*current_row > rmax) {
				rmax = *current_row++;
			}
		}
	}


	//��ֵ������
	unsigned char pixelB, pixelG, pixelR;  //��¼��ͨ��ֵ
	unsigned char DifMax = 40;             //������ɫ���ֵ���ֵ����
	unsigned char WhiteMax = 50;		   //�жϰ�ɫ
	unsigned char B = 215, G = 215, R = 215; //��ͨ������ֵ�趨�������A4ֽ
	int i = 0, j = 0;
	for (i = 0; i < image.rows; i++)   //ͨ����ɫ������ͼƬ���ж�ֵ������
	{
		for (j = 0; j < image.cols; j++)
		{
			pixelB = image.at<Vec3b>(i, j)[0]; //��ȡͼƬ����ͨ����ֵ
			pixelG = image.at<Vec3b>(i, j)[1];
			pixelR = image.at<Vec3b>(i, j)[2];

			if ((abs(B - pixelB) < DifMax) && (abs(G - pixelG) < DifMax) && (abs(R - pixelR) < DifMax) && abs(pixelB - pixelG) < WhiteMax && abs(pixelG - pixelR) < WhiteMax && abs(pixelB - pixelR) < WhiteMax)
			{                                           //������ͨ����ֵ�͸���ͨ����ֵ���бȽ�
				image.at<Vec3b>(i, j)[0] = 255;     //������ɫ��ֵ��Χ�ڵ����óɰ�ɫ
				image.at<Vec3b>(i, j)[1] = 255;
				image.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				image.at<Vec3b>(i, j)[0] = 0;        //��������ɫ��ֵ��Χ�ڵ�����Ϊ��ɫ
				image.at<Vec3b>(i, j)[1] = 0;
				image.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	
	//cvtColor(image, binImg, COLOR_BGR2GRAY);	//�Ҷȴ���
	//threshold(binImg, binImg, 170, 255, THRESH_BINARY);	//��ֵ��������ֵ100����Χ255
	//imshow("������ɫ��Ϣ��ֵ��", binImg);        //��ʾ��ֵ������֮���ͼ��
}



//��̬ѧ����
void morphTreat(Mat& binImg) {
	Mat BinOriImg;     //��̬ѧ������ͼ��
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); //������̬ѧ�����Ĵ�С
	GaussianBlur(binImg, binImg, Size(5, 5), 11, 11);
	dilate(binImg, binImg, element);     //���ж�����Ͳ���
	dilate(binImg, binImg, element);
	dilate(binImg, binImg, element);
	dilate(binImg, binImg, element);
	dilate(binImg, binImg, element);

	erode(binImg, binImg, element);      //���ж�θ�ʴ����
	erode(binImg, binImg, element);
	erode(binImg, binImg, element);
	erode(binImg, binImg, element);
	erode(binImg, binImg, element);
	//imshow("��̬ѧ�����", BinOriImg);        //��ʾ��̬ѧ����֮���ͼ��
	cvtColor(binImg, binImg, CV_BGR2GRAY);   //����̬ѧ����֮���ͼ��ת��Ϊ�Ҷ�ͼ��
	threshold(binImg, binImg, 100, 255, THRESH_BINARY); //�Ҷ�ͼ���ֵ��
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


//��������غ�
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


//��������غ�
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
	double min = 10e6;
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


//���������и�
int cutLeft(Mat& src, Mat& leftImg, Mat& rightImg)
{
	int left = 0, right = src.cols;

	int i;
	for (i = 0; i < src.cols; i++)
	{
		int colValue = getColSum(src, i);
		//cout <<i<<" th "<< colValue << endl;
		if (colValue > 550)
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
		if (colValue < 550)
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


//�����и�
int cutTop(Mat& src, Mat& topImg, Mat& bottomImg)
{
	int top = 0, bottom = src.rows;

	int i;
	for (i = 0; i < src.rows; i++)
	{
		int colValue = getRowSum(src, i);
		//cout <<i<<" th "<< colValue << endl;
		if (colValue > 550)
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
		if (colValue < 550)
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


//ͼ�����
int getSubtract(Mat& src) //����ͼƬ���
{
	Mat img_result;
	int min = 1000000;
	int serieNum = 0;
	for (int i = 0; i < 10; i++) {
		Mat Template = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/" + std::to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		threshold(Template, Template, 100, 255, CV_THRESH_BINARY);
		threshold(src, src, 100, 255, CV_THRESH_BINARY);
		resize(src, src, Size(32, 48), 0, 0, CV_INTER_LINEAR);
		resize(Template, Template, Size(32, 48), 0, 0, CV_INTER_LINEAR);//�����ߴ�		
		//imshow(name, Template);
		absdiff(Template, src, img_result);//
		int diff = getPixelSum(img_result);
		if (diff < min)
		{
			min = diff;
			serieNum = i;
		}
	}
	printf("��С������%d ", min);
	printf("ƥ�䵽��%d��ģ��ƥ���������%d\n", serieNum, serieNum);
	return serieNum;
}


//ͼ����ת
void rotate_Demo(Mat& image, double angle) {
	Mat M;
	int h = image.rows;
	int w = image.cols;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), angle, 1.0);	//����任����M
	double cos = abs(M.at<double>(0, 0));	//��cosֵ
	double sin = abs(M.at<double>(0, 1));	//��sinֵ
	int nw = abs(cos * w - sin * h) / abs(cos * cos - sin * sin);		//�����µĳ�����
	int nh = abs(cos * h - sin * w) / abs(cos * cos - sin * sin);
	M.at<double>(0, 2) += (nw / 2 - w / 2);		//�����µ�����
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	warpAffine(image, image, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(0, 0, 0));
	//imshow("Rotation", dst);
}
