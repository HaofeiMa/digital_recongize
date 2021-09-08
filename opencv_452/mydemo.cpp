#include "mydemo.h"
#include <opencv2/dnn.hpp>

//using namespace cv;
//using namespace std;


void MyDemo::colorSpace_Demo(Mat& image) {
	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("HSV Image", hsv);
	imshow("Gray Image", gray);
	imwrite("E:/Program/OpenCV/vcworkspaces/opencv_452/img/hsv.png", hsv);
	imwrite("E:/Program/OpenCV/vcworkspaces/opencv_452/img/gray.png", gray);
}

void MyDemo::matCreation_Demo() {

	//�����հ�ͼ��
	Mat m_new = Mat::ones(Size(8, 8),CV_8UC3);
	m_new = Scalar(66, 66, 66);
	imshow("new image",m_new);

}

/*
void MyDemo::pixelVisit_Demo(Mat& image) {
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {

			//�Ҷ�ͼ��
			if (dims == 1) {
				int pv = image.at<uchar>(row, col);
				image.at<uchar>(row, col) = 255 - pv;
			}

			//��ɫͼ��
			if (dims == 3) {
				Vec3b bgr = image.at<Vec3b>(row, col);
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}
	}
	imshow("Pixel Visit Demo", image);
}
*/

void MyDemo::pixelVisit_Demo(Mat& image) {
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
	for (int row = 0; row < h; row++) {
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++) {

			//�Ҷ�ͼ��
			if (dims == 1) {
				*current_row++ = 255 - *current_row;
			}

			//��ɫͼ��
			if (dims == 3) {
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}
	imshow("Pixel Visit Demo", image);
}

/*
void MyDemo::operators_Demo(Mat& image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	m = Scalar(20, 20, 20);

	//add(image, m, dst);
	//subtract(image, m, dst);
	multiply(image, m, dst);
	//divide(image, m, dst);

	imshow("operator",dst);

}
*/

void MyDemo::operators_Demo(Mat& image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	m = Scalar(20, 20, 20);
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();

	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {

			//�Ҷ�ͼ��
			if (dims == 1) {
				int pv = image.at<uchar>(row, col);
				image.at<uchar>(row, col) = 255 - pv;
			}

			//��ɫͼ��
			if (dims == 3) {
				Vec3b p1 = image.at<Vec3b>(row, col);
				Vec3b p2 = m.at<Vec3b>(row, col);
				dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(p1[0] * p2[0]);
				dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] * p2[1]);
				dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] * p2[2]);
			}
		}
	}
	imshow("operator",dst);
}


void MyDemo::bitWise_Demo(Mat& image) {
	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1, Rect(50, 50, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(m2, Rect(100, 100, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);
	Mat dst;
	bitwise_and(m1, m2, dst);
	imshow("bitWise", dst);
}


static void onTrack(int lightness, void* data) {
	Mat src = *(Mat*)data;
	Mat m = Mat::zeros(src.size(), src.type());
	Mat dst = Mat::zeros(src.size(), src.type());
	m = Scalar(lightness, lightness, lightness);

	add(src, m, dst);
	imshow("Change Lightness", dst);
}


void MyDemo::checkBar_Demo(Mat& image) {
	namedWindow("Change Lightness", WINDOW_AUTOSIZE);

	int lightness = 50;
	int max_value = 100;

	createTrackbar("Value Bar", "Change Lightness", &lightness, max_value, onTrack,(void *)&image);
	onTrack(lightness, &image);
}


void MyDemo::key_Demo(Mat& image) {
	Mat m = Mat::zeros(image.size(),image.type());
	m = Scalar(10, 10, 10);
	while(true) {
		char k = waitKey(10);
		if (k == 'q') {	// Quit
			break;
		}
		if (k == '1') {	//Key 1
			std::cout << "You enter key 1 - Lightness Up." << std::endl;
			add(image, m, image);
		}
		if (k == '2') {	//Key 2
			std::cout << "You enter key 2 - Lightness Down." << std::endl;
			subtract(image, m, image);
		}
		imshow("Key", image);
	}
}


void MyDemo::colorStyle_Demo(Mat& image) {
	int colormap[] = {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_JET,
		COLORMAP_WINTER,
		COLORMAP_RAINBOW,
		COLORMAP_OCEAN,
		COLORMAP_SUMMER,
		COLORMAP_SPRING,
		COLORMAP_COOL,
		COLORMAP_PINK,
		COLORMAP_HOT,
		COLORMAP_PARULA,
		COLORMAP_MAGMA,
		COLORMAP_INFERNO,
		COLORMAP_PLASMA,
		COLORMAP_VIRIDIS,
		COLORMAP_CIVIDIS,
		COLORMAP_TWILIGHT,
		COLORMAP_TWILIGHT_SHIFTED
	};
	Mat dst;
	int index = 0;
	while (true) {
		int k = waitKey(500);
		if (k == 'q') {
			break;
		}
		applyColorMap(image, dst, colormap[(index++)%19]);
		imshow("colorStyle", dst);
	}
}


void MyDemo::channels_Demo(Mat& image) {
	//std::vector<Mat> mv;
	//split(image, mv);
	Mat dst = Mat::zeros(image.size(), image.type());

	/*
	imshow("Blue Channel", mv[0]);
	imshow("Green Channel", mv[1]);
	imshow("Red Channel", mv[2]);
	
	Mat m1,m2,m3;
	mv[1] = 0;
	mv[2] = 0;
	merge(mv, m1);
	imshow("Blue Channel", m1);

	split(image, mv);
	mv[0] = 0;
	mv[2] = 0;
	merge(mv, m2);
	imshow("Green Channel", m2);

	split(image, mv);
	mv[0] = 0;
	mv[1] = 0;
	merge(mv, m3);
	imshow("Red Channel", m3);
	*/

	int ft[] = { 0,2,1,1,2,0 };
	mixChannels(&image,1, &dst,1, ft,3);
	imshow("Mix", dst);
	
}


void MyDemo::inRange_Demo(Mat& image) {

	//ת��ɫ�ʿռ�
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);

	//��ȡ��Ļ����
	Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
	imshow("mask", mask);

	//��ת��ȡ��������	
	bitwise_not(mask, mask);
	imshow("Tuened mask", mask);

	//���︴�Ƶ��±�����
	Mat bg = imread("E:/Program/OpenCV/vcworkspaces/opencv_452/img/plantbg.jpg");//����ͼƬ
	image.copyTo(bg, mask);
	imshow("Finished", bg);

}

void MyDemo::pixelStatistic_Demo(Mat& image) {
	double minv, maxv;
	Point minLoc, maxLoc;
	std::vector<Mat> mv;
	split(image, mv);

	for (int i = 0; i < mv.size(); i++) {	
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &minLoc);
		std::cout << "channel:" << i << "min value:" << minv << "maxvalue:" << maxv << std::endl;
	}

	Mat mean, stddev;
	meanStdDev(image, mean, stddev);
	std::cout << "means:" << mean << "stddev:" << stddev << std::endl;

}

void MyDemo::drawing_Demo(Mat& image) {
	//���ƾ���
	Rect rect;
	rect.x = 250;
	rect.y = 170;
	rect.width = 100;
	rect.height = 100;
	rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
	

	//����Բ
	circle(image, Point(250, 170), 20, Scalar(255, 0, 0), -1, 8, 0);

	//����ֱ��
	line(image, Point(250, 170), Point(350, 270), Scalar(0, 255, 0), 2, LINE_AA, 0);

	//������Բ
	RotatedRect rrt;
	rrt.center = Point(100, 200);
	rrt.size = Size(100, 200);
	rrt.angle = 0;
	ellipse(image, rrt, Scalar(0, 255, 255), 1, 8);

	imshow("Drawing", image);
}

void MyDemo::random_Demo() {
	Mat bg = Mat::zeros(Size(512, 512), CV_8UC3);
	int width = bg.cols;
	int height = bg.rows;
	RNG rng(666);
	while (true) {
		char k = waitKey(100);
		if(k == 'q')
		{
			break;
		}
		int x1 = rng.uniform(0, width);
		int y1 = rng.uniform(0, height);
		int x2 = rng.uniform(0, width);
		int y2 = rng.uniform(0, height);
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);

		line(bg, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 1, LINE_AA, 0);
		imshow("Randow image", bg);
	}
}

void MyDemo::polyDrawing_Demo() {
	Mat bg = Mat::zeros(Size(512, 512), CV_8UC3);
	Point p1(100, 100);
	Point p2(350, 100);
	Point p3(450, 300);
	Point p4(250, 450);
	Point p5(80, 200);

	std::vector<Point> pts;
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);

	//std::vector<Point> pts(5);
	//pts[0] = p1;
	//pts[1] = p2;
	//pts[2] = p3;
	//pts[3] = p4;
	//pts[4] = p5;

	//fillPoly(bg, pts, Scalar(255, 255, 0), 8, 0);
	//polylines(bg, pts, true, Scalar(0, 255, 255), 3, LINE_AA, 0);

	std::vector<std::vector<Point>> contours;
	contours.push_back(pts);
	drawContours(bg, contours, -1, Scalar(255, 255, 0), -1);
	polylines(bg, pts, true, Scalar(0, 255, 255), 3, LINE_AA, 0);
	imshow("Poly Drawing!", bg);
}


Point sp(-1, -1);	//��ʼ�㣨��ʼֵ-1��-1��
Point ep(-1, -1);	//�����㣨��ʼֵ-1��-1��
Mat temp;			//ԭͼ�Ŀ�¡������ʵʱˢ��ͼƬ

static void on_draw(int event, int x, int y, int flags, void* userdata) {
	Mat bg = *(Mat*)userdata;	//�ص�������������ͼ������
	if (event == EVENT_LBUTTONDOWN) {	//������������
		sp.x = x;	//�����������ʱ��xyֵ
		sp.y = y;
		std::cout << "Start point: " << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP) {//��������̧��
		ep.x = x;	//�������̧��ʱ��xyֵ
		ep.y = y;
		int dx = ep.x - sp.x;	//������γ���
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {	//�����γ���Ϊ����ʱ
			Rect box(sp.x, sp.y, dx, dy);
			rectangle(bg, box, Scalar(0, 0, 255), 2, 8, 0);	//���ƾ���
			imshow("Mouse Drawing", bg);
			imshow("ROI", temp(box));	//��ʾROI���򣨱���ѡ������
			sp.x = -1;	//��ʼ�����긴λ
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {		//����ʼ�����겻�ǳ�ʼֵ��������ƶ�ʱ
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(bg);	//ˢ����Ļ�������һѭ�����Ƶľ���
				rectangle(bg, box, Scalar(0, 0, 255), 2, 8, 0);	//�����¾���
				imshow("Mouse Drawing", bg);
			}
		}
	}
}


void MyDemo::mouseDrawing_Demo(Mat& image) {
	namedWindow("Mouse Drawing", WINDOW_AUTOSIZE);	//����һ������
	setMouseCallback("Mouse Drawing", on_draw,(void*)(&image));	//�������ص�����
	imshow("Mouse Drawing", image);
	temp = image.clone();
}


void MyDemo::normalize_Demo(Mat& image) {
	Mat dst;
	std::cout << image.type() << std::endl;	//CV_8UC3
	image.convertTo(image, CV_32F);			//��������ת��Ϊ����������
	std::cout << image.type() << std::endl;	//CV_32FC3
	normalize(image, dst, 0, 1.0, NORM_MINMAX);	//��һ��
	std::cout << dst.type() << std::endl;
	imshow("Normalize", dst);
}

void MyDemo::resize_Demo(Mat& image) {
	Mat zoomin, zoomout;	//�������ͼ��
	int h = image.rows;		//��ȡԭͼ��Ŀ��
	int w = image.cols;
	resize(image, zoomin, Size(w * 1.5, h * 1.5), 0, 0, INTER_LINEAR);	//ͼ��Ŵ�1.5��
	imshow("zoomin", zoomin);
	resize(image, zoomout, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);		//ͼ����С2��
	imshow("zoomout", zoomout);
}


void MyDemo::flip_Demo(Mat& image) {
	Mat dst;
	flip(image, dst, 0);	//���·�ת
	imshow("���·�ת", dst);
	flip(image, dst, 1);	//���ҷ�ת
	imshow("���ҷ�ת", dst);
	flip(image, dst, -1);	//�Խ��߷�ת��180����ת��
	imshow("�Խ��߷�ת��180����ת��", dst);
}

void MyDemo::rotate_Demo(Mat& image) {
	Mat dst, M;
	int h = image.rows;
	int w = image.cols;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0);	//����任����M
	double cos = abs(M.at<double>(0, 0));	//��cosֵ
	double sin = abs(M.at<double>(0, 1));	//��sinֵ
	int nw = cos * w + sin * h;		//�����µĳ�����
	int nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);		//�����µ�����
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	warpAffine(image, dst, M, Size(nw,nh), INTER_LINEAR,0,Scalar(255,255,255));
	imshow("Rotation", dst);
}

/*
void MyDemo::video_Demo(Mat& image) {
	VideoCapture capture("E:/Program/OpenCV/vcworkspaces/opencv_452/img/marvel.mp4");
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int count = capture.get(CAP_PROP_FRAME_COUNT);
	double fps = capture.get(CAP_PROP_FPS);
	std::cout << "Frame width:" << frame_width << std::endl;
	std::cout << "Frame height:" << frame_height << std::endl;
	std::cout << "Frame count:" << frame_height << std::endl;
	std::cout << "FPS:" << fps << std::endl;
	VideoWriter writer("E:/Program/OpenCV/vcworkspaces/opencv_452/img/test.mp4", capture.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);
	Mat frame;
	while (true) {
		capture.read(frame);
		flip(frame, frame, 1);
		if (frame.empty()) {
			break;
		}
		imshow("Video", frame);
		colorSpace_Demo(frame);
		writer.write(frame);
		int k = waitKey(10);
		if (k == 27) {
			break;
		}
	}
	capture.release();
	writer.release();
}
*/
void MyDemo::video_Demo(Mat& image) {
	VideoCapture capture(0);	//����VideoCapture��
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//��ȡ����ͷ�Ŀ��ߡ�֡����FPS
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	//int count = capture.get(CAP_PROP_FRAME_COUNT);
	//double fps = capture.get(CAP_PROP_FPS);
	//VideoWriter writer("E:/Program/OpenCV/vcworkspaces/opencv_452/img/test.mp4", capture.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);	//������Ƶ
	VideoWriter writer;
	int fourcc = writer.fourcc('X', 'V', 'I', 'D');
	writer.open("E:/Program/OpenCV/vcworkspaces/opencv_452/img/test.mp4", fourcc, 30, Size(frame_width, frame_height), true);	//������Ƶ

	Mat frame;					//����Mat�������ڴ洢ÿһ֡����
	while (capture.isOpened()) {
		capture.read(frame);	//��֡��ȡ��Ƶ
		flip(frame, frame, 1);	//����ȡ����Ƶ���ҷ�ת
		if (frame.empty()) {	//�����Ƶ������δ��⵽����ͷ������ѭ��
			break;
		}
		writer.write(frame);
		imshow("Video", frame);	//ÿ��ѭ����ʾһ֡ͼ��
		char k = waitKey(33);	//��֡��ȡ�ļ��ʱ��
		if (k == 'q') {			//����q���˳�ѭ��
			break;
		}
	}
	capture.release();			//�ͷ���Ƶ
	writer.release();
}

void MyDemo::histShow_Demo(Mat& image) {
	// ��ͨ�����룬���ڷֱ��������ͨ����ֱ��ͼ
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);
	// �����������
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	// ����Blue, Green, Redͨ����ֱ��ͼ
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	// ��ʾֱ��ͼ
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	// ��һ��ֱ��ͼ����
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// ����ֱ��ͼ����
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	// ��ʾֱ��ͼ
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
}


void MyDemo::histShow2_Demo(Mat& image) {
	// 2D ֱ��ͼ
	Mat hsv, hs_hist;
	cvtColor(image, hsv, COLOR_BGR2HSV);	//RGBתHSV
	int hbins = 30, sbins = 32;				//���ö�άֱ��ͼ��ֱ������
	int hist_bins[] = { hbins, sbins };
	float h_range[] = { 0, 180 };			//H��0-180
	float s_range[] = { 0, 256 };			//S��0-256
	const float* hs_ranges[] = { h_range, s_range };
	int hs_channels[] = { 0, 1 };			//ѡ��ͨ��0��ͨ��1
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);

	//���й�һ��
	double maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0, 0);	//�ҵ����ֵ
	int scale = 10;
	Mat hist2d_image = Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);
	for (int h = 0; h < hbins; h++) {
		for (int s = 0; s < sbins; s++)
		{
			float binVal = hs_hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(hist2d_image, Point(h * scale, s * scale),Point((h + 1) * scale - 1, (s + 1) * scale - 1),Scalar::all(intensity),-1);
		}
	}
	applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);
	imshow("H-S Histogram", hist2d_image);
}


void MyDemo::blur_Demo(Mat& image) {
	Mat dst;
	blur(image, dst, Size(1, 15), Point(-1, -1));
	imshow("Blur", dst);
}

void MyDemo::gaussianBlur_Demo(Mat& image) {
	Mat dst;
	GaussianBlur(image, dst, Size(5, 5), 15);
	imshow("GaussianBlur", dst);
}

void MyDemo::faceDetector_Demo() {
	//����VideoCapture��
	//VideoCapture capture(0);
	VideoCapture capture("E:/Program/OpenCV/vcworkspaces/opencv_452/img/test.mp4");
	Mat frame;

	////��ȡģ�ͺ����ò���
	std::string root_dir = "E:/Program/OpenCV/opencv/sources/samples/dnn/face_detector/";
	dnn::Net net = dnn::readNetFromTensorflow(root_dir+"opencv_face_detector_uint8.pb", root_dir+"opencv_face_detector.pbtxt");

	//ʵʱ���
	while (capture.isOpened()) {
		capture.read(frame);	//��֡��ȡ��Ƶ
		flip(frame, frame, 1);	//����ȡ����Ƶ���ҷ�ת
		if (frame.empty()) {	//�����Ƶ������δ��⵽����ͷ������ѭ��
			break;
		}

		//׼������
		Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);	
		//scalefactor=1.0��ʾͼ���ɫ�ʱ�����0��255֮�䣻size��mean����������models.yml�С�����false��ʾ����Ҫrgb��ת��Ҳ����Ҫ����
		net.setInput(blob);//�����ݶ���ģ���С���blob�����NCHW��N�Ǹ�����Cͨ������H�߶ȣ�W��ȣ�
		Mat probs = net.forward(); 
		//����ĵ�һ��γ������ͼ���У�ÿ��ͼ���index��
		//�ڶ�γ�ȣ���ǰͼ���ǵڼ�������batchid���ڼ���ͼimageid��
		//������γ�ȱ�ʾ�ж��ٸ���
		//���ĸ�γ�ȣ�ÿ�������߸�ֵ��ǰ���������ͺ�dst�������������Ŷȣ�����ĸ��Ǿ��ε����ϽǺ����Ͻ�
		Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr());
		//��ĸ���Ϊ������ÿ������߸�ֵΪÿ�е�Ԫ��

		//�������
		for (int i = 0; i < detectionMat.rows; i++) {
			float confidence = detectionMat.at<float>(i, 2);//ȡ��������ֵ�����Ŷ�
			if (confidence > 0.6) {
				//�ٳ���ͼ��Ŀ�Ȼ�߶Ȳ��ܱ�Ϊ��ʵ��
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		imshow("Face Dector", frame);

		char k = waitKey(33);	//��֡��ȡ�ļ��ʱ��
		if (k == 'q') {			//����q���˳�ѭ��
			break;
		}
	}
}
