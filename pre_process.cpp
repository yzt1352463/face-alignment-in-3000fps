#include "opencv2/opencv.hpp"
#include <algorithm>
#include "LBFRegressor.h"
#include "facedetect-dll.h"

using namespace std;
using namespace cv;
#pragma comment(lib,"libfacedetect.lib")

//std::string modelPath = "../img/";
//std::string cascadeName = "../haarcascade_frontalface_alt.xml";
//string dataPath = "./../../Datasets/";
//string cascadeName_used_cmp = "I:/dt/cascade.xml";

float get_definition(Mat img_in);
int wear_glasses(Mat img_in, Mat img_glasses, Mat &img_out);

int main_1()
{

	for (int i = 0; i < 258; i++)
	{
		Mat img_in;
		char name[100];
		sprintf(name, "F:/face/sobel_modify/CASIA-WebFacec/0000099/%03d.jpg", i + 1);
		img_in = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		//img_in = imread("F:/face/sobel_modify/CASIA-WebFacec/0000099/258.jpg",1);
		//printf("%d : %f\n", i + 1,get_definition(img_in));
		Mat glasses;
		glasses = imread("F:/face/sobel_modify/glass1.png", CV_LOAD_IMAGE_GRAYSCALE);
		Mat img_out;
		wear_glasses(img_in, glasses, img_out);
	}

	return 0;
}

bool cmp(short &a, short &b)
{
	if (a>b)
		return true;
	return false;
}

//输入灰度图像,返回清晰度，若返回-1，则表示图片读取失败
float get_definition(Mat img_in)
{

	Mat img_out;
	int pix_num = 50;//提取前 n 个边缘强度高的像素点
	float sum_pix = 0;
	vector<short> pix_value;
	if (img_in.data == NULL)
	{
		printf("image doesn't exist!!");
		return -1;

	}
	Sobel(img_in, img_out, CV_16S, 1, 1);
	for (int row = 0; row < img_out.rows; row++)
	{
		short* data = img_out.ptr<short>(row);
		for (int col = 0; col < img_out.cols; col++)
		{
			pix_value.push_back(abs(data[col]));
			/*pix_num++;
			sum_pix = sum_pix + abs(data[col]);
			partial_sort();*/
		}
	}
	partial_sort(pix_value.begin(), pix_value.begin() + pix_num, pix_value.end(), cmp);
	for (int i = 0; i < 50; i++)
	{
		sum_pix += pix_value[i];
	}
	float definition;
	definition = sum_pix / pix_num;
	return definition;
}
//输入灰度图片，输出为将图片带上眼镜
int wear_glasses(Mat img_in, Mat img_glasses, Mat &img_out)
{
	if (img_in.data == NULL)
	{
		printf("image doesn't exist!!");
		return -1;
	}

	// -- 0. Load LBF model
	LBFRegressor regressor;
	regressor.Load(modelPath + "LBF.model");
	// -- 1. Load the cascades
	CascadeClassifier cascade;
	extern string cascadeName;
	if (!cascade.load(cascadeName)){
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	//确定2个眼睛的定位点
	vector<cv::Point2f> eye;
	vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255) };

	int * pResults = NULL;
	pResults = facedetect_multiview((unsigned char*)(img_in.ptr(0)), img_in.cols, img_in.rows, img_in.step,
		1.2f, 4, 24);
	printf("%d faces detected.\n", (pResults ? *pResults : 0));

	//print the detection results
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 6 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
		faces.push_back(Rect(x, y, w, h));
	}


	// --Alignment

	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++)
	{
		Point center;
		Scalar color = colors[1 % 8];
		BoundingBox boundingbox;

		boundingbox.start_x = r->x;
		boundingbox.start_y = r->y;
		boundingbox.width = (r->width - 1);
		boundingbox.height = (r->height - 1);
		boundingbox.centroid_x = boundingbox.start_x + boundingbox.width / 2.0;
		boundingbox.centroid_y = boundingbox.start_y + boundingbox.height / 2.0;

		Mat_<double> current_shape = regressor.Predict(img_in, boundingbox, 1);

		eye.push_back(Point2f(current_shape(37, 0), current_shape(37, 1)));	//左眼坐标点
		eye.push_back(Point2f(current_shape(43, 0), current_shape(43, 1)));  //右眼坐标点


	}


	//将眼镜的图片缩放为对应2个眼睛点的图片

	if (eye.size())
	{
		float eye_dis; //两个眼睛的距离
		float glass_dis[5] = { 220, 2.0, 3.0 };
		eye_dis = sqrt(pow(eye[0].x - eye[1].x, 2) + pow(eye[0].y - eye[1].y, 2));
		float scale_glass;//眼镜的缩放比例，用来适应眼睛的距离
		scale_glass = eye_dis / glass_dis[0];
		Mat glass_re;
		resize(img_glasses, glass_re, Size2i(scale_glass * img_glasses.cols, scale_glass * img_glasses.rows));
		img_in.copyTo(img_out);
		//逐点赋值，将眼镜图片覆盖到目标上
		//for (int row = eye[0].y - 50 * scale_glass; row < glass_re.rows; row++)
		for (int i = 0; i < glass_re.rows; i++)
		{
			int row = eye[0].y - 50 * scale_glass + i;
			uchar* data = glass_re.ptr<uchar>(i);
			//for (int col = eye[0].x - 90 * scale_glass; col < glass_re.cols; col++)
			for (int j = 0; j < glass_re.cols; j++)
			{
				int col = eye[0].x - 90 * scale_glass + j;
				if (data[j] < 150)
					img_out.at<uchar>(row, col) = data[j];
			}
		}
		circle(img_out, eye[0], 1, Scalar(255, 255, 255), -1, 8, 0);
		circle(img_out, eye[1], 1, Scalar(255, 255, 255), -1, 8, 0);
		imshow("123", img_out);
		waitKey();
	}

	return 1;
}