#ifndef INC_Tracking_H
#define INC_Tracking_H

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "Clusterer.h"
//#include "DSST.h"

class KalmanTracking
{
public:
	Point2f updateKalman(KalmanFilter kf, Point2f loc);
	Point2f predictKalman(KalmanFilter kf, Point2f loc);
	trackedRectangle *trackObjects(trackedRectangle *rects, Mat frame, int nrects, int dt, bool update);
	KalmanFilter initKalman(Point2f loc, float dt);
};

#endif

