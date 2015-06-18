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

trackedRectangle *dataAssociate::bindTrackingDetection(trackedRectangle *dets, int ndets, trackedRectangle *tracks, int &ntracks, Mat frame)
{
	double threshold= 10;
	bool foundMatch=false;
	
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
	int nrectsCounter= nfinalTracks;

	//2- Initialize found of track to false till associated
	for(int j=0; j<ndets; j++) //Iterate on rectangles from detection
	{
		foundMatch=false;
		if(dets[j].neglected)
			continue;

		for(int i=0; i<nrectsCounter; i++) //Iterate on rectangles from tracking
		{
			if(finalTracks[i].neglected )//|| rects3[i].flag==1 || rects3[i].flag==2 || rects3[i].flag==3)//associated before
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
				
				if(flag==1)// Obj Flag
				{
					//float *pred= updateKalman(rects3[i].kalman, coord(rects1[j].centroid.x, rects1[j].centroid.y));
					//finalTracks[i].centroid.x= pred[0];
					//finalTracks[i].centroid.y= pred[1];
					//rects3[i].width= rects1[j].width;
					//rects3[i].height= rects1[j].height;

					//rects3[i].topLeftCorner.x= rects3[i].centroid.x- rects3[i].width/2;
					//rects3[i].topLeftCorner.y= rects3[i].centroid.y- rects3[i].height/2;
					//if(rects3[i].topLeftCorner.x<0)
						//rects3[i].topLeftCorner.x=0;
					//if(rects3[i].topLeftCorner.y<0)
						//rects3[i].topLeftCorner.y=0;
					
					//track.updatePatch(mtDet.currentFrame, imgCol, &rects3[i]);
					finalTracks[i].bb= finalTracks[i].dsst.processFrame(frame, true, true);
					break;
				}
				else
				{
					finalTracks[i].bb= finalTracks[i].bb= finalTracks[i].dsst.processFrame(frame, true, false);
					/*track.matchTemplate(mtDet.currentFrame, imgCol, &rects3[i]);
					float *pred= updateKalman(rects3[i].kalman, coord(rects3[i].centroid.x, rects3[i].centroid.y));
					rects3[i].centroid.x= pred[0];
					rects3[i].centroid.y= pred[1];
					rects3[i].topLeftCorner.x= rects3[i].centroid.x- rects3[i].width/2;
					rects3[i].topLeftCorner.y= rects3[i].centroid.y- rects3[i].height/2;
					if(rects3[i].topLeftCorner.x<0)
						rects3[i].topLeftCorner.x=0;
					if(rects3[i].topLeftCorner.y<0)
						rects3[i].topLeftCorner.y=0;*/
				}
				
			}
		}

		if(!foundMatch) //New Object, should be added
		{
			finalTracks[nfinalTracks]= dets[j];
			finalTracks[nfinalTracks].dsst.preprocess(frame.rows, frame.cols, frame, finalTracks[nfinalTracks].bb);
			finalTracks[nfinalTracks].first= false;
			//track.updatePatch(mtDet.currentFrame, imgCol, &rects3[nrects3]);
			nfinalTracks++;
		}
	}

	
	for(int i=0; i<nrectsCounter; i++)
	{
		bool boundry= false;
		int w= frame.cols;
		int h= frame.rows;
		if(finalTracks[i].bb.x+ finalTracks[i].bb.width>=(w-5))
			boundry= true;
		if(finalTracks[i].bb.y+ finalTracks[i].bb.height>=(h-5))
			boundry= true;
		if(finalTracks[i].bb.x-5<=0)
			boundry= true;
		if(finalTracks[i].bb.y-5<=0)
			boundry= true;

		if(finalTracks[i].flag==1)
		{
			finalTracks[i].ntracked++;
			finalTracks[i].nNotDetected=0;
			if(finalTracks[i].ntracked>2 && !finalTracks[i].trackReady)
			{
				finalTracks[i].trackID= globalIDCounter;
				globalIDCounter++;
				finalTracks[i].trackReady= true;
			}
		}
		else if(finalTracks[i].flag==2 || finalTracks[i].flag==3)
		{
			if(finalTracks[i].trackReady )//&& !boundry)
				finalTracks[i].nNotDetected=0;
			else
			{
				finalTracks[i].nNotDetected++;
				if(finalTracks[i].nNotDetected>3)
				{
					finalTracks[i].neglected= true;
					continue;
				}
				else if(finalTracks[i].nNotDetected>1 && boundry)///// it was here rects not rects3
					finalTracks[i].neglected= true;
			}
		}
		else 
		{
			if(ndets!=0)
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
						rects3[i].topLeftCorner.y=0;*/
					//track.matchTemplate(mtDet.currentFrame, imgCol, &rects3[i]);
					//float *pred= updateKalman(rects3[i].kalman, coord(rects3[i].centroid.x, rects3[i].centroid.y));
					finalTracks[i].bb= finalTracks[i].dsst.processFrame(frame, true, false);
				}
				
				finalTracks[i].nNotDetected++;
				if(finalTracks[i].nNotDetected>4 && finalTracks[i].ntracked<=10)
					finalTracks[i].neglected= true;
				else if(finalTracks[i].nNotDetected>2 && boundry)///////it was rects here as well not rects3
					finalTracks[i].neglected= true;
				else if(finalTracks[i].nNotDetected>15)
					finalTracks[i].neglected= true;
			}
			else
				finalTracks[i].bb= finalTracks[i].dsst.processFrame(frame, true, false);
				//double score= track.matchTemplate(mtDet.currentFrame, imgCol, &rects3[i]);
		}
	}
	

	//delete[] found;
	delete[] tracks;
	tracks=0;
	
	if(ndets!=0)
		delete[] dets;
	dets=0;
	ntracks= nfinalTracks;


	for(int i=0; i<nfinalTracks; i++)
	{		
		if(finalTracks[i].neglected==true)
		{
			finalTracks[i].first= true;
			continue;
		}

		for(int j=0; j<nfinalTracks; j++)
		{
			if(i==j)
				continue;
			if(finalTracks[i].neglected)
				continue;

			double area= checkIntersection(finalTracks[i], finalTracks[j]);
			if(area/ (finalTracks[i].bb.width*finalTracks[i].bb.height)>=0.5 || area/ (finalTracks[j].bb.width*finalTracks[j].bb.height)>=0.5)
			{
				int removedIndex;
				if(finalTracks[j].ntracked> finalTracks[i].ntracked)
					removedIndex= i;
				else if(finalTracks[j].ntracked== finalTracks[i].ntracked)
				{
					if(finalTracks[j].nNotDetected< finalTracks[i].nNotDetected)
						removedIndex= i;
					else
						removedIndex=j;
				}
				else
					removedIndex=j;
				
				if(finalTracks[removedIndex].ntracked<20)
				{
					finalTracks[removedIndex].neglected= true;
				}
			}
		}
	}
	return finalTracks;
}
