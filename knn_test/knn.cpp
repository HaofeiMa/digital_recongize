//#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <iostream>
#include <ml.hpp>
#include <fstream>
using namespace std;
using namespace cv;
using namespace cv::ml;



int main()
{
    Ptr<KNearest> model;
    fstream file;
    string fileName = "num_knn_pixel.yml";
    file.open(fileName.c_str(), ios::in);

    if (file)
    {
        //训练结果存在，加载训练结果
        model = Algorithm::load<KNearest>("num_knn_pixel.yml");
    }
    else
    {	//训练结果不存在，重新训练
        Mat img = imread("E:/Program/OpenCV/vcworkspaces/knn_test/images/data/digits.png");
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        int b = 20;
        int m = gray.rows / b;   //原图为1000*2000
        int n = gray.cols / b;   //裁剪为5000个20*20的小图块
        Mat data, labels;   //特征矩阵
        for (int i = 0; i < n; i++)
        {
            int offsetCol = i * b; //列上的偏移量
            for (int j = 0; j < m; j++)
            {
                int offsetRow = j * b;  //行上的偏移量
                                      //截取20*20的小块
                Mat tmp;
                gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
                //reshape  0：通道不变  其他数字，表示要设置的通道数
                //reshape  表示矩阵行数，如果设置为0，则表示保持原有行数不变，如果设置为其他数字，表示要设置的行数
                data.push_back(tmp.reshape(0, 1));  //序列化后放入特征矩阵
                labels.push_back((int)j / 5);  //对应的标注
            }

        }
        data.convertTo(data, CV_32F); //uchar型转换为cv_32f
        int samplesNum = data.rows;
        int trainNum = 3000;
        Mat trainData, trainLabels;
        trainData = data(Range(0, trainNum), Range::all());   //前3000个样本为训练数据
        trainLabels = labels(Range(0, trainNum), Range::all());

        //使用KNN算法
        int K = 5;
        Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
        model = KNearest::create();
        model->setDefaultK(K);
        model->setIsClassifier(true);
        model->train(tData);

        //预测分类
        double train_hr = 0, test_hr = 0;
        Mat response;
        // compute prediction error on train and test data
        for (int i = 0; i < samplesNum; i++)
        {
            Mat sample = data.row(i);
            float r = model->predict(sample);   //对所有行进行预测
                                                //预测结果与原结果相比，相等为1，不等为0
            r = std::abs(r - labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

            if (i < trainNum)
                train_hr += r;  //累积正确数
            else
                test_hr += r;
        }

        test_hr /= samplesNum - trainNum;
        train_hr = trainNum > 0 ? train_hr / trainNum : 1.;

        printf("accuracy: train = %.1f%%, test = %.1f%%\n",
            train_hr * 100., test_hr * 100.);
        //保存训练结果
        model->save("./num_knn_pixel.yml");
    }
    //=============================== 预测部分==============================
    //预测分类
    Mat img = imread("E:/Program/OpenCV/vcworkspaces/knn_test/images/test/6.jpg");
    cvtColor(img, img, COLOR_BGR2GRAY);
    //threshold(src, src, 0, 255, CV_THRESH_OTSU);
    imshow("Image", img);
    resize(img, img, Size(20, 20));
    Mat test;
    test.push_back(img.reshape(0, 1));
    test.convertTo(test, CV_32F);
    int result = model->predict(test);

    cout << "识别数字：" << result << endl;
    while (char(waitKey(1)) != 'q') {}

    return 0;
}