/*
 * Clusterer.cpp
 *
 *  Created on: 22 May, 2015
 *      Author: Mennatullah
 */

#include "Clusterer.h"

trackedRectangle *Clusterer::dbScanClustering(int &nrects, int w, int h)
{
	using namespace Metrics;

	Clustering::Points ps;

	for(int i=0; i<nOutliers; i++)
	{
		Clustering::Point p(2);
		p[0]= outliers[i].x;
		p[1]= outliers[i].y;
		ps.push_back(p);
	}
	double rad=30;
	Clustering::DBSCAN clusters(ps, rad, 2); 
	
	Distance<Euclidean<Clustering::Point> > d;
	clusters.computeSimilarity(d);     

	clusters.run_cluster();

	trackedRectangle *rectangles= new trackedRectangle[clusters._clusters.size()];
	nrects=0;
	BOOST_FOREACH(Clustering::Cluster c, clusters._clusters)
	{	
		rectangles[nrects].first=true;
		rectangles[nrects].trackReady=false;
		rectangles[nrects].ntracked=0;
		rectangles[nrects].nNotDetected=0;
		rectangles[nrects].neglected= false;
		rectangles[nrects].trackID= -1;
		rectangles[nrects].flag=-1;
		rectangles[nrects].nNearlyTracked=0;
		
		double avgX, avgY, maxX, minX, maxY, minY;
		int k=0;
		avgY=0; avgX=0;
		minX=1000; minY=1000;
		maxX=-1; maxY=-1;
		BOOST_FOREACH(Clustering::PointId pid, c)
		{
			CvPoint2D32f cor= cvPoint2D32f(clusters._ps[pid][0], clusters._ps[pid][1]);
			if(minX> cor.x)
				minX=cor.x;
			if(minY> cor.y)
				minY=cor.y;
			if(maxX< cor.x)
				maxX=cor.x;
			if(maxY< cor.y)
				maxY=cor.y;

			avgX+= cor.x;
			avgY+= cor.y;

			k++;
		}
		rectangles[nrects].bb= Rect(minX, minY, maxX-minX, maxY-minY);
		rectangles[nrects].bb.height= rectangles[nrects].bb.height<15? rectangles[nrects].bb.height+5: rectangles[nrects].bb.height;
		
		if(rectangles[nrects].bb.x<0)
			rectangles[nrects].bb.x=0;
		if(rectangles[nrects].bb.y<0)
			rectangles[nrects].bb.y=0;
		
		nrects++;
	}

	return rectangles;
}

trackedRectangle *Clusterer::clearClusters(trackedRectangle *rects, int &nrects, int w, int h)
{
	//postprocessing on the clusters
	double indent= 5;
	for(int i=0; i<nrects; i++)
	{
		if(rects[i].bb.width+ rects[i].bb.x>= w)
			rects[i].neglected= true;
		if(rects[i].bb.height+ rects[i].bb.y>= h)
			rects[i].neglected= true;
		if(rects[i].bb.width<=10)
			rects[i].bb.width=15;
		if(rects[i].bb.height<=10)
			rects[i].bb.height=15;

		Point2f centroid= Point2f(rects[i].bb.x+rects[i].bb.width/2, rects[i].bb.y+ rects[i].bb.height/2);
		if((centroid.x-indent)<=0 || (centroid.x+indent)>=w)
			rects[i].neglected= true;
		if((centroid.y-indent)<=0 || (centroid.y+indent)>=h)
			rects[i].neglected= true;

		indent=2;
		if((centroid.x-rects[i].bb.width/2-indent)<=0 || (centroid.x+rects[i].bb.width/2+indent)>=w)
			rects[i].neglected= true;
		if((centroid.y-rects[i].bb.height/2-indent)<=0 || (centroid.y+rects[i].bb.height/2+indent)>=h)
			rects[i].neglected= true;
		if(rects[i].bb.width>150 || rects[i].bb.height>150)
			rects[i].neglected= true;
	}

	trackedRectangle *rects2;
	int counter=0;
	for(int i=0; i<nrects; i++)
	{
		if(rects[i].neglected)
			continue;
		counter++;
	}
	rects2= new trackedRectangle[counter];

	int k=0;
	indent=5;
	for(int i=0; i<nrects; i++)
	{
		if(rects[i].neglected)
			continue;

		rects2[k]= rects[i];

		rects2[k].bb.x-=indent;
		rects2[k].bb.x = rects2[k].bb.x<0?0:rects2[k].bb.x;
		rects2[k].bb.y-=indent;
		rects2[k].bb.y = rects2[k].bb.y<0?0:rects2[k].bb.y;
		
		rects2[k].bb.width+= 2*indent;
		double tlx= rects2[k].bb.x;
		rects2[k].bb.width= tlx+ rects2[k].bb.width>= w? w- tlx: rects2[k].bb.width;
		
		rects2[k].bb.height+= 2*indent;
		double tly= rects2[k].bb.y;
		rects2[k].bb.height= tly+ rects2[k].bb.height>= h? h- tly: rects2[k].bb.height;
		k++;
	}
	
	delete[] rects;
	nrects= counter;
	return rects2;
}

/*bool Clusterer::checkInsideRect(CvPoint2D32f p, trackedRectangle rect)
{
	bool Xinside= (p.x>=rect.topLeftCorner.x && p.x<=rect.topLeftCorner.x + rect.width);
	bool Yinside= (p.y>=rect.topLeftCorner.y && p.y<=rect.topLeftCorner.y + rect.height);
	if( Xinside && Yinside)
		return true;
	else
		return false;
}

double Clusterer::getEuclidDistance(CvPoint2D32f p1, CvPoint2D32f p2)
{
	double d= sqrtf(powf(p1.x-p2.x, 2) + powf(p1.y-p2.y, 2));
	return d;
}*/