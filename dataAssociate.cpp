#include "dataAssociate.h"

int dataAssociate::associate(trackedRectangle rect1, trackedRectangle rect2, double area)
{
	int flag;
	//double th= 0.55;
	double th= 0.6;
	double a1= rect1.bb.width*rect1.bb.height;
	double a2= rect2.bb.width*rect2.bb.height;
	
	double Rd= area/ a1;
	double Rt= area/ a2;
	if(Rd<th && Rt<th)//NA
		flag= 0;
	else if(Rd>=th && Rt>=th)//Obj
		flag=1;
	else if(Rd>=th && Rt<th) //split
		flag=2;
	else if(Rd<th && Rt>=th) //merge
		flag=3;
	return flag;
}


double dataAssociate::checkIntersection(trackedRectangle rect1, trackedRectangle rect2)
{
	//trackedRectangle *rect;
	if(rect1.bb.x+rect1.bb.width < rect2.bb.x)
		return 0;
	else if(rect1.bb.x > rect2.bb.x+ rect2.bb.width)
		return 0;
	else if(rect1.bb.y > rect2.bb.y+ rect2.bb.height)
		return 0;
	else if(rect1.bb.y + rect1.bb.height < rect2.bb.y)
		return 0;

	CvPoint2D32f topLeft, bottomRight;
	topLeft.x = rect1.bb.x > rect2.bb.x ? rect1.bb.x: rect2.bb.x;
	topLeft.y = rect1.bb.y > rect2.bb.y ? rect1.bb.y: rect2.bb.y;
	
	bottomRight.x = rect1.bb.x+rect1.bb.width > rect2.bb.x+rect2.bb.width ? rect2.bb.x+rect2.bb.width: rect1.bb.x+rect1.bb.width;
	bottomRight.y = rect1.bb.y+rect1.bb.height > rect2.bb.y+rect2.bb.height ? rect2.bb.y+rect2.bb.height: rect1.bb.y+rect1.bb.height;
		
	double area= (bottomRight.x- topLeft.x) * (bottomRight.y- topLeft.y);
	return area;
}

/*trackedRectangle *dataAssociate::bindTrackingDetection(trackedRectangle *dets, int ndets, trackedRectangle *tracks, int &ntracks, Mat frame)
{
	double threshold= 10;
	bool foundMatch=false;
	KalmanTracking tr;
	
	//1- Copy only the non neglected tracks
	trackedRectangle *finalTracks= new trackedRectangle[ndets+ntracks];
	int k=0;
	for(int i=0; i<ntracks; i++)
	{
		if(tracks[i].neglected)
			continue;
		finalTracks[k]=tracks[i];
		finalTracks[k].flag=-1;
		k++;
	}
	int nfinalTracks= k;
	
	//2- Initialize found of track to false till associated
	//int nrectsCounter=nrects3;
	for(int j=0; j<ndets; j++) //Iterate on rectangles from detection
	{
		foundMatch=false;
		if(dets[j].neglected)
			continue;

		for(int i=0; i<nfinalTracks; i++) //Iterate on rectangles from tracking
		{
			if(finalTracks[i].neglected )
				continue;

			//3- Associate detected target to a track
			trackedRectangle intersectRect;
			double area= checkIntersection(dets[j], finalTracks[i]);
			int flag= associate(dets[j], finalTracks[i], area); //flag= 0->NA, 1->Obj, 2->Split, 3->Merge
			if(finalTracks[i].flag<=0 || (finalTracks[i].flag!=1 && flag==1))
				finalTracks[i].flag= flag;

			if(flag==1 || flag==2 || flag==3)
			{
				foundMatch=true;
				//found[i]= true;
				
				if(flag==1)
				{
					Point2f pred= tr.updateKalman(finalTracks[i].kalman, Point2f(dets[j].bb.x + dets[j].bb.width/2, dets[j].bb.y + dets[j].bb.height/2));
					finalTracks[i].bb= finalTracks[i].dsst.processFrame(frame, false, true);
					/*finalTracks[i].bb.x= pred.x- dets[j].bb.width/2;
					finalTracks[i].bb.y= pred.y- dets[j].bb.height/2;
					finalTracks[i].bb.width= dets[j].bb.width;
					finalTracks[i].bb.height= dets[j].bb.height;
					 
					if(finalTracks[i].bb.x<0)
						finalTracks[i].bb.x=0;
					if(finalTracks[i].bb.y<0)
						finalTracks[i].bb.y=0;end of comment/////
					
					break;
				}
				else
				{
					//track.matchTemplate(mtDet.currentFrame, imgCol, &rects3[i]);
					finalTracks[i].bb= finalTracks[i].dsst.processFrame(frame, false, true);
					Point2f pred= tr.updateKalman(finalTracks[i].kalman, Point2f(finalTracks[i].bb.x+finalTracks[i].bb.width/2, finalTracks[i].bb.y+finalTracks[i].bb.height/2));
					/*rects3[i].centroid.x= pred.x;
					rects3[i].centroid.y= pred.y;
					rects3[i].topLeftCorner.x= rects3[i].centroid.x- rects3[i].width/2;
					rects3[i].topLeftCorner.y= rects3[i].centroid.y- rects3[i].height/2;
					if(rects3[i].topLeftCorner.x<0)
						rects3[i].topLeftCorner.x=0;
					if(rects3[i].topLeftCorner.y<0)
						rects3[i].topLeftCorner.y=0;///////end of comment/
				}
				
			}
		}

		if(!foundMatch) //New Object, should be added
		{
			finalTracks[nfinalTracks]= dets[j];
			track.updatePatch(mtDet.currentFrame, imgCol, &rects3[nrects3]);
			nrects3++;
		}
	}

	
	for(int i=0; i<nrectsCounter; i++)
	{
		bool boundry= false;
		int w= mtDet.currentFrame->width;
		int h= mtDet.currentFrame->height;
		if(rects3[i].topLeftCorner.x+ rects3[i].width>=(w-5))
			boundry= true;
		if(rects3[i].topLeftCorner.y+ rects3[i].height>=(h-5))
			boundry= true;
		if(rects3[i].topLeftCorner.x-5<=0)
			boundry= true;
		if(rects3[i].topLeftCorner.y-5<=0)
			boundry= true;

		if(rects3[i].flag==1)
		{
			rects3[i].ntracked++;
			rects3[i].nNotDetected=0;
			if(rects3[i].ntracked>2 && !rects3[i].trackReady)
			{
				rects3[i].trackID= globalIDCounter;
				globalIDCounter++;
				rects3[i].trackReady= true;
			}
		}
		else if(rects3[i].flag==2 || rects3[i].flag==3)
		{
			if(rects3[i].trackReady )//&& !boundry)
				rects3[i].nNotDetected=0;
			else
			{
				rects3[i].nNotDetected++;
				if(rects3[i].nNotDetected>3)
				{
					rects3[i].neglected= true;
					continue;
				}
				else if(rects[i].nNotDetected>1 && boundry)
					rects3[i].neglected= true;
			}
		}
		else 
		{
			if(nrects1!=0)
			{
				//4- Correct, this part should be only implemented to objects not newly detected
				//float *pred= updateKalman(rects3[i].kalman);//, coord(rects3[i].centroid.x, rects3[i].centroid.y));
				//if(pred[0]>0)
				{
					/*rects3[i].centroid.x= pred[0];
					rects3[i].centroid.y= pred[1];
					rects3[i].topLeftCorner.x= rects3[i].centroid.x- rects3[i].width/2;
					rects3[i].topLeftCorner.y= rects3[i].centroid.y- rects3[i].height/2;
					if(rects3[i].topLeftCorner.x<0)
						rects3[i].topLeftCorner.x=0;
					if(rects3[i].topLeftCorner.y<0)
						rects3[i].topLeftCorner.y=0;end of comment////
					track.matchTemplate(mtDet.currentFrame, imgCol, &rects3[i]);
					float *pred= updateKalman(rects3[i].kalman, coord(rects3[i].centroid.x, rects3[i].centroid.y));
				}
				
				rects3[i].nNotDetected++;
				if(rects3[i].nNotDetected>4 && rects3[i].ntracked<=10)
					rects3[i].neglected= true;
				else if(rects[i].nNotDetected>2 && boundry)
					rects3[i].neglected= true;
				else if(rects3[i].nNotDetected>15)
					rects3[i].neglected= true;
			}
			else
				double score= track.matchTemplate(mtDet.currentFrame, imgCol, &rects3[i]);


			rects3[i].topLeftCorner.x= rects3[i].centroid.x- rects3[i].width/2;
			rects3[i].topLeftCorner.y= rects3[i].centroid.y- rects3[i].height/2;
			if(rects3[i].topLeftCorner.x<0)
				rects3[i].topLeftCorner.x=0;
			if(rects3[i].topLeftCorner.y<0)
				rects3[i].topLeftCorner.y=0;
		}
	}
	

	//delete[] found;
	delete[] rects2;
	rects2=0;
	
	if(nrects1!=0)
		delete[] rects1;
	rects1=0;
	nrects2= nrects3;


	for(int i=0; i<nrects3; i++)
	{		
		if(rects3[i].neglected==true)
		{
			if(rects3[i].templateMatched!=0)
				cvReleaseImage(&rects3[i].templateMatched);
			rects3[i].templateMatched=0;
			if(rects3[i].candidateRegion!=0)
				cvReleaseImage(&rects3[i].candidateRegion);
			rects3[i].candidateRegion=0;
			if(!rects3[i].first)
				cvReleaseKalman(&rects3[i].kalman);
			rects3[i].first= true;
			if(rects3[i].templateMatchedFull!=0)
				cvReleaseImage(&rects3[i].templateMatchedFull);
			rects3[i].templateMatchedFull=0;
			if(rects3[i].kpts!=0)
				delete[] rects3[i].kpts;
			rects3[i].kpts=0;
			if(rects3[i].descs!=0)
				delete[] rects3[i].descs;
			rects3[i].descs=0;
			continue;
		}

		for(int j=0; j<nrects3; j++)
		{
			if(i==j)
				continue;
			if(rects3[i].neglected)
				continue;

			double area= help.checkIntersection(rects3[i], rects3[j]);
			if(area/ (rects3[i].width*rects3[i].height)>=0.5 || area/ (rects3[j].width*rects3[j].height)>=0.5)
			{
				int removedIndex;
				if(rects3[j].ntracked> rects3[i].ntracked)
					removedIndex= i;
				else if(rects3[j].ntracked== rects3[i].ntracked)
				{
					if(rects3[j].nNotDetected< rects3[i].nNotDetected)
						removedIndex= i;
					else
						removedIndex=j;
				}
				else
					removedIndex=j;
				
				if(rects3[removedIndex].ntracked<20)
				{
					rects3[removedIndex].neglected= true;
					if(rects3[removedIndex].templateMatched!=0)
						cvReleaseImage(&rects3[removedIndex].templateMatched);
					rects3[removedIndex].templateMatched=0;
					if(rects3[removedIndex].candidateRegion!=0)
						cvReleaseImage(&rects3[removedIndex].candidateRegion);
					rects3[removedIndex].candidateRegion=0;
					if(!rects3[removedIndex].first)
						cvReleaseKalman(&rects3[removedIndex].kalman);
					rects3[i].first= true;
					if(rects3[removedIndex].templateMatchedFull!=0)
						cvReleaseImage(&rects3[removedIndex].templateMatchedFull);
					rects3[removedIndex].templateMatchedFull=0;
					if(rects3[removedIndex].kpts!=0)
						delete[] rects3[removedIndex].kpts;
					rects3[removedIndex].kpts=0;
					if(rects3[removedIndex].descs!=0)
						delete[] rects3[removedIndex].descs;
					rects3[removedIndex].descs=0;
				}
			}
		}
	}
	return rects3;
}*/