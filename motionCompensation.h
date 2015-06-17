#ifndef MComp_H
#define MComp_H

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Clusterer.h"
using namespace cv;
using namespace std;

class motionCompensation
{

public:
	int nrects;
	Clusterer clustrer;
	trackedRectangle *rects;

	void processFrame(cv::Mat frame, cv::Mat previousFrame);
	vector<Point2f> computeHomography(cv::Mat frame, cv::Mat previousFrame);
	vector<Point2f> detectCorners(cv::Mat frame);
	void clusterOutliers();
};

#endif