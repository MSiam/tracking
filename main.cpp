/*
 * main.h
 * Multiple Target Deetection and Tracking Test
 *  Created on: 22 May, 2015
 *      Author: Mennatullah
 */

#include <iostream>
#include <string>
#include "motionCompensation.h"
#include "dataAssociate.h"
//#include "tracker.h"
using namespace std;

string intToStr(int i, string path ,int sz , string post){
	string bla = "frame";
	for(int i = 0 ; i < sz ; i++)
		bla+="0";
	stringstream ss;
	ss<<i;
	string ret ="";
	ss>>ret;
	string name = bla.substr(0,bla.size()-ret.size());
	name = path+name+ret+post;
	return name;
}

int main(int argc, char *argv[])
{
	int currentDS= 0;
	string datasets[] = {"egtest01", "egtest02", "egtest03", "egtest04", "egtest05", "redTeam"};
	int frames[] = {1802, 1300, 2570, 1832, 1763, 1917};
	string path= "/home/mennatullah/Datasets/Eglin/"+datasets[currentDS]+"/";
	int frameNumber=0;
	
	motionCompensation mc;
	dataAssociate da;
	//KalmanTracking tr;
	cv::Mat frame, previousFrame;
	trackedRectangle *finalRects;
	int nfinalRects;
	while(frameNumber<frames[currentDS])
	{
		string fileName= intToStr(frameNumber, path, 5, ".jpg");
		cv::Mat frame= cv::imread(fileName.c_str());
		
		if(frameNumber>0)
		{
			if(frameNumber==1) //Initialize tracking
			{
				mc.processFrame(frame, previousFrame);
				finalRects= da.initTracking(mc.rects, frame, mc.nrects, 1);
				nfinalRects= mc.nrects;
				for(int i=0; i<nfinalRects; i++)
				{
					finalRects[i].trackID= da.globalIDCounter;
					da.globalIDCounter++;
				}
			}
			else
			{
				mc.processFrame(frame, previousFrame);
				finalRects= da.bindTrackingDetection(mc.rects, mc.nrects, finalRects, nfinalRects, frame);
			}
			/*Mat frame3= frame.clone();
			stringstream ss;
			ss<<frameNumber;
			putText(frame3, ss.str(), Point(0, 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(1), 2);
			for(int i=0; i<mc.nrects; i++)
			{
				if(mc.rects[i].neglected)
					continue;

				rectangle(frame3, Point2f(mc.rects[i].bb.x, mc.rects[i].bb.y), Point2f(mc.rects[i].bb.x+mc.rects[i].bb.width, mc.rects[i].bb.y+mc.rects[i].bb.height),Scalar(0,0,255) );
			}
			imshow("dets", frame3);
			*/
			//cout<<"here 4"<<endl;
			Mat frame2= frame.clone();
			stringstream ss2;
			ss2<<frameNumber;
			putText(frame2, ss2.str(), Point(0, 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(1), 2);
			for(int i=0; i<nfinalRects; i++)
			{
				if(finalRects[i].neglected || !finalRects[i].trackReady)
					continue;
		
				rectangle(frame2, Point2f(finalRects[i].bb.x, finalRects[i].bb.y), Point2f(finalRects[i].bb.x+finalRects[i].bb.width, finalRects[i].bb.y+finalRects[i].bb.height),Scalar(0,0,255) );
			}
			imshow("tracks", frame2);
		}
		
		waitKey(1);
		previousFrame= frame;
		frameNumber++;
	}
	
	return 0;
}
