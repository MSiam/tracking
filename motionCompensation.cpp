#include "motionCompensation.h"

#define MAXCORNERS 100000

void motionCompensation::processFrame(cv::Mat frame, cv::Mat previousFrame)
{
	vector<Point2f> outliers= computeHomography(frame, previousFrame);
	
	nrects=0;
	clustrer.nOutliers= outliers.size();
	clustrer.outliers= outliers;
	rects= clustrer.dbScanClustering(nrects, frame.cols, frame.rows);
	for(int i=0; i<nrects; i++)
	{
		if(rects[i].bb.width+ rects[i].bb.x> frame.cols)
			rects[i].neglected= true;
		if(rects[i].bb.height+ rects[i].bb.y> frame.rows)
			rects[i].neglected= true;
	}

	for(int i=0; i<nrects; i++)
	{
		if(rects[i].neglected)
			continue;
		
		//rectangle(frame, Point2f(rects[i].bb.x, rects[i].bb.y), Point2f(rects[i].bb.x+rects[i].bb.width, rects[i].bb.y+rects[i].bb.height),Scalar(0,0,255) );
	}
	//imshow("Objects ", frame);
}

vector<Point2f> motionCompensation::detectCorners(cv::Mat frame)
{
	vector<Point2f> features;
	cv::Mat frameGray;
	cvtColor( frame, frameGray, CV_BGR2GRAY );
	cv::goodFeaturesToTrack(frameGray, features, MAXCORNERS, 0.0005,  3, noArray(), 3, true, 0.04);
	/*for( int i = 0; i < features.size(); i++ )
	{
		circle( frame, Point( features[i].x, features[i].y ), 1,  Scalar(0, 0, 255), 2, 8, 0 );
	}
	imshow("Corners ", frame);*/
	return features;
}

vector<Point2f> motionCompensation::computeHomography(cv::Mat frame, cv::Mat previousFrame)
{
	//1- Detect Harris Corners
	vector<Point2f> corners= detectCorners(frame);
	vector<Point2f> previousCorners= detectCorners(previousFrame);

	//2- Compute Lucas Kanade
	float thresh= 5;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::calcOpticalFlowPyrLK(previousFrame, frame, previousCorners, corners, status, err);
	/*for( int i = 0; i < corners.size(); i++ )
	{
		if(status[i]==1 && err[i]<thresh)
			line( frame, Point(previousCorners[i].x, previousCorners[i].y) ,Point( corners[i].x, corners[i].y ), Scalar(0, 0, 255));
	}
	imshow("OF ", frame);*/
	
	//3- Compute Homography 
	vector<uchar> mask;
	Mat H= findHomography(previousCorners, corners, CV_RANSAC, 0.7, mask);
	vector<Point2f> outliers;
	for(int i=0; i<corners.size(); i++)
	{
		if(mask[i]==0)
		{
			outliers.push_back(corners[i]);
			//circle( frame, Point( corners[i].x, corners[i].y ), 1,  Scalar(0, 0, 255), 2, 8, 0 );
		}
	}
	//imshow("Outliers", frame);
	return outliers;
}
