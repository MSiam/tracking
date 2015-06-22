#define _USE_MATH_DEFINES

#ifndef INC_DSST_H
#define INC_DSST_H

#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include "Params.h"
//#include "HOG.h"
#include <windows.h>

using namespace std;
using namespace cv;

class DSSTTracker
{
	Params tParams;
	trackingSetup tSetup;
	HOGParams hParams;

public:
	cv::Mat inverseFourier(cv::Mat original, int flag=0);
	cv::Mat createFourier(cv::Mat original, int flag=0);
	Mat hann(int size);
	float *convert1DArray(Mat &patch);
	double *convertTo1DFloatArrayDouble(Mat &patch); //Normalize by 255
	Mat convert2DImage(float *arr, int w, int h);
	Point ComputeMaxDisplayfl(Mat &img,string winName="FloatImg");
	Mat *create_feature_map(Mat& patch, int full, int &nChns, Mat& Gray, bool scaling);
	Mat get_scale_sample(Mat img, trackingSetup tSetup, Params tParams, int &nDims,bool display = false);
	Mat *get_translation_sample(cv::Mat img, trackingSetup tSet, int &nDims);
	void train(bool first, cv::Mat img);
	Point updateCentroid(Point oldC, int w , int h , int imgw, int imgh);
	cv::Rect processFrame(cv::Mat img);
	void preprocess(int rows,int cols, cv::Mat img, cv::Rect bb);
	cv::Mat visualize(Rect rect, cv::Mat img,Scalar scalar = cvScalarAll(0));
	float *fhog(double *im, int* dims, int sbin);
	Point displayFloat(Mat img);
};

#endif