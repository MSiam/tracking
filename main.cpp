#include <iostream>
#include <string>
#include "motionCompensation.h"
#include "dataAssociate.h"
#include "tracker.h"
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
	string datasets[] = {"egtest01", "egtest02"};

	//C:\Users\mincosy\Desktop\Aerial Tracking\datasets\egtest01
	string path= "C:\\Users\\mincosy\\Desktop\\Aerial Tracking\\datasets\\"+datasets[currentDS]+"\\";
	int frameNumber=0;
	
	motionCompensation mc;
	dataAssociate da;
	KalmanTracking tr;
	cv::Mat frame, previousFrame;
	trackedRectangle *finalRects;
	int nfinalRects;
	while(frameNumber<1802)
	{
		string fileName= intToStr(frameNumber, path, 5, ".jpg");
		cv::Mat frame= cv::imread(fileName.c_str());
		
		if(frameNumber>0)
		{
			if(frameNumber==1) //Initialize tracking
			{
				mc.processFrame(frame, previousFrame);
				finalRects= tr.trackObjects(mc.rects, frame, mc.nrects, 1, false);
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
				//finalRects= tr.trackObjects(finalRects, frame, nfinalRects, 1, false);
				//nfinalRects= nfinalRects;
			}

			for(int i=0; i<nfinalRects; i++)
			{
				if(finalRects[i].neglected)
					continue;
		
				rectangle(frame, Point2f(finalRects[i].bb.x, finalRects[i].bb.y), Point2f(finalRects[i].bb.x+finalRects[i].bb.width, finalRects[i].bb.y+finalRects[i].bb.height),Scalar(0,0,255) );
			}
		}
		imshow("testing", frame);
		waitKey(10);
		previousFrame= frame;
		frameNumber++;
	}
	
	return 0;
}
