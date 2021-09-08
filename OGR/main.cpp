#include <opencv2/opencv.hpp>
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
int cutTop(Mat& src, Mat& dstImg, Mat& bottomImg);	//上下切割
int getSubtract(Mat& src);		//两张图片相减
void rotate_Demo(Mat& image, double angle);	//图像旋转


int main() {
	//摄像头初始化
	VideoCapture capture(1);	//创建VideoCapture类，使用外置摄像头（1）
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


		/************************** 提取A4纸区域并识别数字 *****************************/
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
		double location_x = 0.0;
		double location_y = 0.0;
		vector<vector<Point>> contours;
		vector<Vec4i>hierarchy;
		findContours(binImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i < contours.size(); i++)
		{
			//绘制轮廓的最小外接矩形  
			length = arcLength(contours[i], true);      //获取轮廓周长
			area = contourArea(contours[i]);			//获取轮廓面积
			if (area > 2000 && area < 300000)			//矩形区域面积大小判断，符合条件的继续
			{
				rect = boundingRect(contours[i]);		//计算矩形边界
				boxTemp = minAreaRect(contours[i]);		//获取轮廓的矩形
				boxPoints(boxTemp, pts);				//获取矩形四个顶点坐标（左上，右上，右下，左下）
				for (int row = 0; row < pts.rows; row++) {		//从列表中依次读出四个顶点坐标
					pt[row].x = pts.at<uchar>(row, 0);
					pt[row].y = pts.at<uchar>(row, 1);
				}
				angleTemp = boxTemp.angle;              //得到倾斜角度
				if (angleTemp > 45) {					//对于逆时针偏转的情况，倾斜角度为-(90-angle)
					angleTemp = angleTemp - 90;
				}
				
				axisLongTemp = sqrt(pow(pt[1].x - pt[0].x, 2) + pow(pt[1].y - pt[0].y, 2));  //计算长轴（勾股定理）
				axisShortTemp = sqrt(pow(pt[2].x - pt[1].x, 2) + pow(pt[2].y - pt[1].y, 2)); //计算短轴（勾股定理）


				if (axisShortTemp > axisLongTemp)		//如果短轴大于长轴，交换数据
				{
					LengthTemp = axisLongTemp;
					axisLongTemp = axisShortTemp;
					axisShortTemp = LengthTemp;
				}

				rectArea = axisLongTemp * axisShortTemp;	//计算矩形的实际面积

				long2Short = axisLongTemp / axisShortTemp;	//计算长宽比
				
				// 长宽比A4纸为1.414，利用长宽比、矩形面积和短边长度作为限制条件
				if (long2Short > 1 && long2Short < 1.8  && rectArea > 5000 && rectArea < 300000 && axisShortTemp > 50)
				{
					rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);		//在摄像头图像中画出矩形区域
					if (rect.width > 100 && rect.height > 100 && axisShortTemp>100) {	//缩小矩形范围，便于数字识别
						rect.x += 40;
						rect.y += 40;
						rect.width -= 40;
						rect.height -= 40;
					}
					imshow("Video", frame);					//显示摄像头拍摄画面
					location_x = rect.x + rect.width / 2;	//获得矩形中心坐标，即A4纸中心坐标
					location_y = rect.y + rect.height / 2;
					Mat a4Img = frame(rect);				//提取A4纸区域
					Mat a4binImg;
					cvtColor(a4Img, a4binImg, CV_BGR2GRAY);   //将A4纸区域转化为灰度图像
					threshold(a4binImg, a4binImg, 120, 255, THRESH_BINARY); //灰度图像二值化
					colorReverse(a4binImg);					//颜色反转
					rotate_Demo(a4binImg, angleTemp);		//根据前所计算角度，对图像进行旋转，保证数字水平存在
					imshow("A4", a4binImg);

					/********************  数字识别方法1：轮廓提取法  *********************/
					
					//找到各数字所在轮廓，并绘制矩形
					vector<vector<Point>> contours_rec;  //定义轮廓和层次结构
					vector<Vec4i> hierarchy_rec;
					findContours(a4binImg, contours_rec, hierarchy_rec, RETR_EXTERNAL, CHAIN_APPROX_NONE); //寻找轮廓
					int i = 0;
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

					//对所有找到的轮廓逐一识别
					Mat num[20];
					int output1[3] = {-1,-1,-1};	//分别存储识别的三个数字
					int output1_flag = 0;
					int matchingNum = 0;
					int matchingRate = 0;
					int matchflag = 0;
					for (int j = 0; j < i; j++) {
						a4binImg(a4rect[j]).copyTo(num[j]);		//提取包围数字的矩形区域至num[j]
						imshow("num", num[j]);
						imgMatch(num[j], matchingRate, matchingNum);	//数字匹配
						if (matchingRate < 400000) {
							output1[output1_flag++] = matchingNum;	//将识别到的数字顺序存入数组
							if (output1_flag>=3) {		//当识别到全部三个数字时打印输出
								cout << "识别数字：" << output1[0] * 100 + output1[1] * 10 + output1[2] << "\t坐标位置：[" << location_x << ", " << location_y << "]" << endl;
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
							


					/********************  数字识别方法1：行切法  *********************/

					//如果上一种方法没有识别成功
					if (matchflag == 0) {
						Mat leftImg, rightImg, topImg, bottomImg;
						int topRes = cutTop(a4binImg, topImg, bottomImg);	//对提取的A4纸区域逐行扫描，获得行像素之和>550的部分topImg，以及剩余部分bottomImg
						int matchNum = -1, matchRate = 10e6;
						int output2[3] = { -1,-1,-1 };	//分别存储三个数字
						int output2_flag = 0;
						while (topRes == 0)		//当仍存在行像素和>550的部分时
						{
							int leftRes = cutLeft(topImg, leftImg, rightImg);	//对行像素之和>550的部分topImg逐列扫描，获得列像素之和>550的部分leftImg，以及剩余部分rightImg
							while (leftRes == 0) {	//当仍存在列像素和>550的部分时
								imgMatch(leftImg, matchRate, matchNum);	//数字识别
								if (matchRate < 200000) {
									output2[output2_flag++] = matchNum;
									if (output2_flag >= 3) {
										cout << "识别数字：" << output2[0] * 100 + output2[1] * 10 + output2[2] << "\t坐标位置：[" << location_x << ", " << location_y << "]" << endl;
										output2_flag = 0;
										output2[0] = -1;
										output2[1] = -1;
										output2[2] = -1;
									}
								}
								Mat srcTmp = rightImg.clone();
								leftRes = cutLeft(srcTmp, leftImg, rightImg);	//对剩余部分rightImg继续逐列扫描
							}
							Mat srcTmp = bottomImg.clone();
							topRes = cutTop(srcTmp, topImg, bottomImg);			//对剩余部分bottomImg继续逐行扫描
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


//数字左右切割
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


//上下切割
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
	int nw = abs(cos * w - sin * h) / abs(cos * cos - sin * sin);		//计算新的长、宽
	int nh = abs(cos * h - sin * w) / abs(cos * cos - sin * sin);
	M.at<double>(0, 2) += (nw / 2 - w / 2);		//计算新的中心
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	warpAffine(image, image, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(0, 0, 0));
	//imshow("Rotation", dst);
}
