#include "tracker.h"

Point2f KalmanTracking::updateKalman(KalmanFilter kf, Point2f loc)
{
	Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
	Mat prediction = kf.predict();
	measurement(0) = loc.x;
	measurement(1) = loc.y;
	Mat estimated = kf.correct(measurement);
	Point2f statePt(estimated.at<float>(0),estimated.at<float>(1));
	return statePt;
}

Point2f KalmanTracking::predictKalman(KalmanFilter kf, Point2f loc)
{
	Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
	Mat prediction = kf.predict();
	Point2f predictPt(prediction.at<float>(0),prediction.at<float>(1));
	return predictPt;
}


KalmanFilter KalmanTracking::initKalman(Point2f loc, float dt)
{
	KalmanFilter KF(4, 2, 0);
	KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1);
	
	// init...
	KF.statePre.at<float>(0) = loc.x; //StatePost in my code :D!!
	KF.statePre.at<float>(1) = loc.y;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4)); //{0.5,0,0,0, 0,0.5,0,0, 0,0,0.5,0, 0,0,0,0.5};
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//{0.5, 0, 0, 0.5 };
	setIdentity(KF.errorCovPost, Scalar::all(.1));//{ 100,0,0,0, 0,100,0,0, 0,0,100,0, 0,0,0,100};

	return KF;
}

trackedRectangle *KalmanTracking::trackObjects(trackedRectangle *rects, Mat frame, int nrects, int dt, bool update)
{
	for(int i=0; i<nrects; i++)
	{
		if(rects[i].first && !update)
		{

			//1- Create Kalman Object
			//rects[i].kalman= initKalman(Point2f(rects[i].bb.x+rects[i].bb.width/2, rects[i].bb.y+rects[i].bb.height/2), dt);
			//cout<<"Init tracking "<<i<<endl;
			rects[i].dsst.preprocess(frame.rows, frame.cols, frame, rects[i].bb);
			rects[i].first= false;
		}
		else
			rects[i].bb= rects[i].dsst.processFrame(frame, true, true);
	}
	
	return rects;
}