#ifndef INC_ASSOC_H
#define INC_ASSOC_H

#include "Clusterer.h"
#include "tracker.h"

class dataAssociate
{
public:
	int globalIDCounter;
	
	dataAssociate()
	{
		globalIDCounter=0;
	}
	int associate(trackedRectangle rect1, trackedRectangle rect2, double area);
	double checkIntersection(trackedRectangle rect1, trackedRectangle rect2);
	trackedRectangle *bindTrackingDetection(trackedRectangle *dets, int ndets, trackedRectangle *tracks, int &ntracks, Mat frame);
};

#endif