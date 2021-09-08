/*
尚存问题：
1. 数字1轮廓无法正常提取
*/

#include <opencv2/opencv.hpp>
//#include <opencv2/cv.h>
//#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>

using namespace cv;
using namespace std;

void binaryProc(Mat& image);	//图像二值化
void morphTreat(Mat& image);	//形态学处理
void colorReverse(Mat& image);	//颜色反转
int getColSum(Mat src, int col);//获得列像素点和
int getRowSum(Mat src, int row);//获得行像素点和
int getPixelSum(Mat& image);	//获得所有像素点和
int imgMatch(Mat& image, int& rate, int& num);		//模板匹配
int cutLeft(Mat& src, Mat& leftImg, Mat& rightImg);	//左右切割
int cutTop(Mat& src, Mat& dstImg);	//上下切割
int getSubtract(Mat& src);		//两张图片相减
void rotate_Demo(Mat& image, double angle);	//图像旋转


int main() {
	Mat src = imread("E:/Program/OpenCV/vcworkspaces/OGR/images/photo.jpg");	//读取图片

	//摄像头初始化
	VideoCapture capture(1);	//创建VideoCapture类
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//获取摄像头的宽、高、帧数、FPS
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);

	
	//摄像头读取
	Mat frame;
	while (capture.isOpened()) {
		capture.read(frame);	//逐帧读取视频
		if (frame.empty()) {	//如果视频结束或未检测到摄像头则跳出循环
			break;
		}
		resize(frame, frame, Size(640, 640 * frame.rows / frame.cols));
		Mat binImg;
		frame.copyTo(binImg);  //二值化之后的图像
		binaryProc(binImg);	//二值化处理
		morphTreat(binImg);			//形态学处理
		imshow("binImg", binImg);
		//colorReverse(grayImg);	//颜色反转

		/*******************************************************/
		double length, area, rectArea;     //定义轮廓周长、面积、外界矩形面积
		double long2Short = 0.0;           //体态比=长边/短边
		Rect rect;           //外界矩形
		RotatedRect box, boxTemp;  //外接矩形
		CvPoint2D32f pt[4];    //矩形定点变量
		Mat pts;    //矩形定点变量
		double axisLong = 0.0, axisShort = 0.0;        //矩形的长边和短边
		double axisLongTemp = 0.0, axisShortTemp = 0.0;//矩形的长边和短边
		double LengthTemp;     //中间变量
		float  angle = 0;      //记录倾斜角度
		float  angleTemp = 0;
		vector<vector<Point>> contours;
		vector<Vec4i>hierarchy;
		findContours(binImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i < contours.size(); i++)
		{
			//绘制轮廓的最小外结矩形  
			//RotatedRect rect1 = minAreaRect(contours[i]);
			length = arcLength(contours[i], true);       //获取轮廓周长
			area = contourArea(contours[i]);       //获取轮廓面积
			if (area > 2000 && area < 300000)     //矩形区域面积大小判断
			{
				rect = boundingRect(contours[i]);//计算矩形边界
				boxTemp = minAreaRect(contours[i]);  //获取轮廓的矩形
				boxPoints(boxTemp, pts);              //获取矩形四个顶点坐标（左上，右上，右下，左下）
				for (int row = 0; row < pts.rows; row++) {
					pt[row].x = pts.at<uchar>(row, 0);
					pt[row].y = pts.at<uchar>(row, 1);
				}
				angleTemp = boxTemp.angle;                 //得到倾斜角度
				if (angleTemp > 45) {
					angleTemp = angleTemp - 90;
				}
				
				axisLongTemp = sqrt(pow(pt[1].x - pt[0].x, 2) + pow(pt[1].y - pt[0].y, 2));  //计算长轴（勾股定理）
				axisShortTemp = sqrt(pow(pt[2].x - pt[1].x, 2) + pow(pt[2].y - pt[1].y, 2)); //计算短轴（勾股定理）


				if (axisShortTemp > axisLongTemp)   //短轴大于长轴，交换数据
				{
					LengthTemp = axisLongTemp;
					axisLongTemp = axisShortTemp;
					axisShortTemp = LengthTemp;
				}

				rectArea = axisLongTemp * axisShortTemp;  //计算矩形的面积

				long2Short = axisLongTemp / axisShortTemp; //计算长宽比
				if (long2Short > 1 && long2Short < 1.8  && rectArea > 5000 && rectArea < 300000 && axisShortTemp > 50)
				{
					rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);
					if (rect.width > 100 && rect.height > 100 && axisShortTemp>100) {
						rect.x += 30;
						rect.y += 30;
						rect.width -= 30;
						rect.height -= 30;
					}
					imshow("Video", frame);    //显示最终结果图

					Mat a4Img = frame(rect);
					Mat a4binImg;
					cvtColor(a4Img, a4binImg, CV_BGR2GRAY);   //将形态学处理之后的图像转化为灰度图像
					threshold(a4binImg, a4binImg, 120, 255, THRESH_BINARY); //灰度图像二值化
					colorReverse(a4binImg);
					rotate_Demo(a4binImg, angleTemp);
					//imshow("A4", a4binImg);
					/********************  数字识别  *********************/
					vector<vector<Point>> contours_rec;  //定义轮廓和层次结构
					vector<Vec4i> hierarchy_rec;
					findContours(a4binImg, contours_rec, hierarchy_rec, RETR_EXTERNAL, CHAIN_APPROX_NONE); //寻找轮廓
					int i = 0;
					Point2f pp[5][4];   //定义点集
					vector<vector<Point>>::iterator It;
					Rect a4rect[15];
					for (It = contours_rec.begin(); It < contours_rec.end(); It++) {                        //画出包围数字的最小矩形
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

					imshow("A4", a4binImg);

					Mat num[20];
					int output[3] = {-1,-1,-1};
					int output_flag = 0;
					int matchingNum = 0;
					int matchingRate = 0;
					for (int j = 0; j < i; j++) {
						a4binImg(a4rect[j]).copyTo(num[j]);
						imshow("num", num[j]);
						imgMatch(num[j], matchingRate, matchingNum);
						if (matchingRate < 400000) {
							output[output_flag] = matchingNum;
							output_flag++;
							if (output[0] >= 0 && output[1] >= 0 && output[2] >= 0) {
								cout << output[0] * 100 + output[1] * 10 + output[2] << endl;
								output_flag = 0;
								output[0] = -1;
								output[1] = -1;
								output[2] = -1;
							}
							//imwrite(to_string(matchingNum) + ".jpg", num[j]);
						}
					}
					cout << endl;
					/*****************************************************/

					//内部再识别
					/*
					resize(a4binImg, a4binImg, Size(640, 640 * a4binImg.rows / a4binImg.cols));
					double length_inside, area_inside, rectArea_inside;     //定义轮廓周长、面积、外界矩形面积
					double long2Short_inside = 0.0;           //体态比=长边/短边
					Rect rect_inside;           //外界矩形
					RotatedRect box_inside, boxTemp_inside;  //外接矩形
					CvPoint2D32f pt_inside[4];    //矩形定点变量
					Mat pts_inside;    //矩形定点变量
					double axisLong_inside = 0.0, axisShort_inside = 0.0;        //矩形的长边和短边
					double axisLongTemp_inside = 0.0, axisShortTemp_inside = 0.0;//矩形的长边和短边
					double LengthTemp_inside;     //中间变量
					vector<vector<Point>> contours_inside;
					vector<Vec4i>hierarchy_inside;
					findContours(a4binImg, contours_inside, hierarchy_inside, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
					for (int i = 0; i < contours_inside.size(); i++)
					{
						//绘制轮廓的最小外结矩形  
						//RotatedRect rect1 = minAreaRect(contours[i]);
						length_inside = arcLength(contours_inside[i], true);       //获取轮廓周长
						area_inside = contourArea(contours_inside[i]);       //获取轮廓面积
						if (area_inside > 2000 && area_inside < 300000)     //矩形区域面积大小判断
						{
							rect_inside = boundingRect(contours_inside[i]);//计算矩形边界
							boxTemp_inside = minAreaRect(contours_inside[i]);  //获取轮廓的矩形
							boxPoints(boxTemp_inside, pts_inside);              //获取矩形四个顶点坐标（左上，右上，右下，左下）
							for (int row = 0; row < pts_inside.rows; row++) {
								pt_inside[row].x = pts_inside.at<uchar>(row, 0);
								pt_inside[row].y = pts_inside.at<uchar>(row, 1);
							}

							axisLongTemp_inside = sqrt(pow(pt_inside[1].x - pt_inside[0].x, 2) + pow(pt_inside[1].y - pt_inside[0].y, 2));  //计算长轴（勾股定理）
							axisShortTemp_inside = sqrt(pow(pt_inside[2].x - pt_inside[1].x, 2) + pow(pt_inside[2].y - pt_inside[1].y, 2)); //计算短轴（勾股定理）


							if (axisShortTemp_inside > axisLongTemp_inside)   //短轴大于长轴，交换数据
							{
								LengthTemp_inside = axisLongTemp_inside;
								axisLongTemp_inside = axisShortTemp_inside;
								axisShortTemp_inside = LengthTemp_inside;
							}

							rectArea_inside = axisLongTemp_inside * axisShortTemp_inside;  //计算矩形的面积

							long2Short_inside = axisLongTemp_inside / axisShortTemp_inside; //计算长宽比
							if (long2Short_inside > 1 && long2Short_inside < 1.8 && rectArea_inside > 5000 && rectArea_inside < 300000 && axisShortTemp_inside > 50)
							{
								rectangle(a4binImg, rect_inside, Scalar(0, 0, 255), 2, 8, 0);
								if (rect_inside.width > 100 && rect_inside.height > 100 && axisShortTemp_inside > 100) {
									rect_inside.x += 30;
									rect_inside.y += 30;
									rect_inside.width -= 30;
									rect_inside.height -= 30;
								}
								imshow("a4binImg", a4binImg);    //显示最终结果图

								Mat a4binImg_inside = a4binImg(rect_inside);
								imshow("a4binImg_inside", a4binImg_inside);
								
								//数字识别
								vector<vector<Point>> contours_rec;  //定义轮廓和层次结构
								vector<Vec4i> hierarchy_rec;
								findContours(a4binImg_inside, contours_rec, hierarchy_rec, RETR_EXTERNAL, CHAIN_APPROX_NONE); //寻找轮廓
								int i = 0;
								Point2f pp[5][4];   //定义点集
								vector<vector<Point>>::iterator It;
								Rect a4rect[15];
								for (It = contours_rec.begin(); It < contours_rec.end(); It++) {                        //画出包围数字的最小矩形
									a4rect[i].x = (float)boundingRect(*It).tl().x;
									a4rect[i].y = (float)boundingRect(*It).tl().y;
									a4rect[i].width = (float)boundingRect(*It).br().x - (float)boundingRect(*It).tl().x;
									a4rect[i].height = (float)boundingRect(*It).br().y - (float)boundingRect(*It).tl().y;
									if ((a4rect[i].height > 80) && (a4rect[i].width > 50) && (a4rect[i].height < 300) && (a4rect[i].width < 300)) {
										rectangle(a4binImg_inside, a4rect[i], Scalar(0, 0, 0), 0, 8, 0);
										i++;
									}
								}

								//imshow("A4", a4binImg_inside);

								Mat num[20];
								int output[3] = { -1,-1,-1 };
								int output_flag = 0;
								int matchingNum = 0;
								int matchingRate = 0;
								for (int j = 0; j < i; j++) {
									a4binImg_inside(a4rect[j]).copyTo(num[j]);
									imshow("num", num[j]);
									imgMatch(num[j], matchingRate, matchingNum);
									if (matchingRate < 400000) {
										output[output_flag] = matchingNum;
										output_flag++;
										if (output[0] >= 0 && output[1] >= 0 && output[2] >= 0) {
											cout << output[0] * 100 + output[1] * 10 + output[2] << endl;
											output_flag = 0;
											output[0] = -1;
											output[1] = -1;
											output[2] = -1;
										}
										//imwrite(to_string(matchingNum) + ".jpg", num[j]);
									}
								}
								cout << endl;




								box_inside = boxTemp_inside;
								axisLong_inside = axisLongTemp_inside;
								axisShort_inside = axisShortTemp_inside;
								//cout << "倾斜角度：" << angle << endl;
							}


						}
					}
					*/

					//行切识别数字
					/*
					Mat leftImg, rightImg, topImg, bottomImg;
					int leftRes = cutLeft(a4binImg, leftImg, rightImg);
					int matchNum = 0, matchRate = 0;
					cout << "begin cutting" << leftRes << endl;
					while (leftRes == 0)
					{
						cout << "leftCutting" << endl;
						//	char nameLeft[10];
						//	sprintf(nameLeft, "%dLeft", i);
						//	char nameRight[10];
						//	sprintf(nameRight, "%dRight", i);
						//	i++;
						//imshow(nameLeft, leftImg);
						//	stringstream ss;
						//	ss << nameLeft;
						//	imwrite("D:\\" + ss.str() + ".jpg", leftImg);
						//	ss >> nameLeft;
						int topRes = cutTop(rightImg, topImg, bottomImg);
						while (topRes == 0) {
							cout << "TopCutting" << endl;
							Mat srcTmp = topImg.clone();
							imgMatch(bottomImg, matchNum, matchRate);//数字识别
							imshow("num", bottomImg);
							if (matchRate < 300000) {
								cout << "识别数字：" << matchNum << "\t\t匹配度：" << matchRate << endl;
								//imwrite(to_string(matchingNum) + ".jpg", num[j]);
							}
							topRes = cutTop(srcTmp, topImg, bottomImg);
						}
						Mat srcTmp = leftImg.clone();
						leftRes = cutLeft(srcTmp, leftImg, rightImg);
					}
					*/
					


					box = boxTemp;
					angle = angleTemp;
					axisLong = axisLongTemp;
					axisShort = axisShortTemp;
				}
				

			}
		}

		//imshow("Video", frame);
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
	//获取BGR三通道最大值
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

	//二值化处理
	unsigned char pixelB, pixelG, pixelR;  //记录各通道值
	unsigned char DifMax = 40;             //基于颜色区分的阈值设置
	unsigned char WhiteMax = 50;		   //判断白色
	unsigned char B = 215, G = 215, R = 215; //各通道的阈值设定，针对与A4纸
	int i = 0, j = 0;
	for (i = 0; i < image.rows; i++)   //通过颜色分量将图片进行二值化处理
	{
		for (j = 0; j < image.cols; j++)
		{
			pixelB = image.at<Vec3b>(i, j)[0]; //获取图片各个通道的值
			pixelG = image.at<Vec3b>(i, j)[1];
			pixelR = image.at<Vec3b>(i, j)[2];

			if ((abs(B - pixelB) < DifMax) && (abs(G - pixelG) < DifMax) && (abs(R - pixelR) < DifMax) && abs(pixelB - pixelG) < WhiteMax && abs(pixelG - pixelR) < WhiteMax && abs(pixelB - pixelR) < WhiteMax)
			{                                           //将各个通道的值和各个通道阈值进行比较
				image.at<Vec3b>(i, j)[0] = 255;     //符合颜色阈值范围内的设置成白色
				image.at<Vec3b>(i, j)[1] = 255;
				image.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				image.at<Vec3b>(i, j)[0] = 0;        //不符合颜色阈值范围内的设置为黑色
				image.at<Vec3b>(i, j)[1] = 0;
				image.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	
	//cvtColor(image, binImg, COLOR_BGR2GRAY);	//灰度处理
	//threshold(binImg, binImg, 170, 255, THRESH_BINARY);	//二值化处理，阈值100，范围255
	//imshow("基于颜色信息二值化", binImg);        //显示二值化处理之后的图像
}



//形态学处理
void morphTreat(Mat& binImg) {
	Mat BinOriImg;     //形态学处理结果图像
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); //设置形态学处理窗的大小
	GaussianBlur(binImg, binImg, Size(5, 5), 11, 11);
	dilate(binImg, binImg, element);     //进行多次膨胀操作
	dilate(binImg, binImg, element);
	dilate(binImg, binImg, element);
	dilate(binImg, binImg, element);
	dilate(binImg, binImg, element);

	erode(binImg, binImg, element);      //进行多次腐蚀操作
	erode(binImg, binImg, element);
	erode(binImg, binImg, element);
	erode(binImg, binImg, element);
	erode(binImg, binImg, element);
	//imshow("形态学处理后", BinOriImg);        //显示形态学处理之后的图像
	cvtColor(binImg, binImg, CV_BGR2GRAY);   //将形态学处理之后的图像转化为灰度图像
	threshold(binImg, binImg, 100, 255, THRESH_BINARY); //灰度图像二值化
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
	cutTop(leftImg, leftImg);
	return 0;
}

int cutTop(Mat& src, Mat& dstImg)//上下切割
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
	Rect rect(0, top, src.cols, height);
	dstImg = src(rect).clone();
	//Rect rectTop(0, top, src.cols, height);
	//topImg = src(rectTop).clone();
	//Rect rectBottom(0, bottom, src.cols, src.rows-height);
	//bottomImg = src(rectBottom).clone();
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
		threshold(Template, Template, 100, 255, CV_THRESH_BINARY);
		threshold(src, src, 100, 255, CV_THRESH_BINARY);
		resize(src, src, Size(32, 48), 0, 0, CV_INTER_LINEAR);
		resize(Template, Template, Size(32, 48), 0, 0, CV_INTER_LINEAR);//调整尺寸		
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

//图像旋转
void rotate_Demo(Mat& image, double angle) {
	Mat M;
	int h = image.rows;
	int w = image.cols;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), angle, 1.0);	//定义变换矩阵M
	double cos = abs(M.at<double>(0, 0));	//求cos值
	double sin = abs(M.at<double>(0, 1));	//求sin值
	int nw = cos * w + sin * h;		//计算新的长、宽
	int nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);		//计算新的中心
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	warpAffine(image, image, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(0, 0, 0));
	//imshow("Rotation", dst);
}
