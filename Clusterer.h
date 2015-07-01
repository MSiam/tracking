/*
 * Clusterer.h
 *
 *  Created on: 22 May, 2015
 *      Author: Mennatullah
 */

#ifndef INC_Clustrer_H
#define INC_Clustrer_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include "dbscan.h"
#include "distance.h"
#include "KCFTracker.h"
#include "DSST.h"
using namespace std;
using namespace cv;

class trackedRectangle
{
public:
	cv::Rect bb;
	bool first;
	int trackID;
	bool neglected;
	CvPoint2D32f predictedCentroid, trackedCentroid;
	int ntracked, nNotDetected, nNearlyTracked;
	bool trackReady;
	int flag;
	
	//KalmanFilter kalman;
#ifdef KCF
	KCFTracker trObj;
#else
	DSSTTracker trObj;
#endif
};

class Clusterer
{
public:
	int nOutliers;
	vector<Point2f> outliers;
public:
	trackedRectangle *dbScanClustering(int &nrects, int w, int h);
	trackedRectangle *refineClusters(trackedRectangle *rects, int nrects);
	trackedRectangle *clearClusters(trackedRectangle *rects, int &nrects, int w, int h);
	trackedRectangle *mergeClusters(trackedRectangle *rects, int &nrects);
	trackedRectangle merge(trackedRectangle rect1, trackedRectangle rect2);

private:
	bool checkInsideRect(CvPoint2D32f p, CvRect rect);
	bool checkNearby(CvRect rect1, CvRect rect2);
	double getEuclidDistance(CvPoint2D32f p1, CvPoint2D32f p2);
	bool checkVariance(trackedRectangle rect);
};
#endif
