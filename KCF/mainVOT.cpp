/*
 * mainVOT.cpp
 *
 *  Created on: May 28, 2015
 *      Author: sara
 */

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include "KCFTracker.h"
#include "defines.h"
#include "vot.hpp"

using namespace cv;
using namespace std;

struct GT
{
	double x1;
	double y1;
	double x2;
	double y2;
	double x3;
	double y3;
	double x4;
	double y4;
};
struct target
{
	CvRect init;
	int firstFrame;

	target(int x, int y, int w, int h, int firstF)
	{
		init.x = x;
		init.y = y;
		init.width = w;
		init.height = h;
		firstFrame = firstF;
	}
};

GT *readGroundtruth(string path)
{
	GT *groundtruth = new GT[2000];
	ifstream fin((path + "groundtruth.txt").c_str());
	double y;
	char x;
	int frameNumber = 0;
	while (!fin.eof())
	{
		GT currentGT;
		fin >> currentGT.x1 >> x >> currentGT.y1 >> x >> currentGT.x2 >> x >> currentGT.y2 >> x >> currentGT.x3 >> x >> currentGT.y3 >> x >> currentGT.x4 >> x >> currentGT.y4;
		groundtruth[frameNumber] = currentGT;
		frameNumber++;
	}
	return groundtruth;
}

void convert8To4(VOTPolygon p, Point& centroid, int& width, int& height)
{
	int top = round(MIN(p.y3, MIN(p.y4, MIN(p.y1, p.y2))));
	int left = round(MIN(p.x3, MIN(p.x4, MIN(p.x1, p.x2))));
	int bottom = round(MAX(p.y3, MAX(p.y4, MAX(p.y1, p.y2))));
	int right = round(MAX(p.x3, MAX(p.x4, MAX(p.x1, p.x2))));

	double x = round((p.x3 + p.x4 + p.x1 + p.x2) / 4) - 1;
	double y = round((p.y3 + p.y4 + p.y1 + p.y2) / 4.0) - 1;
	double A1 = sqrt(pow(p.x1 - p.x2, 2) + pow(p.y1 - p.y2, 2)) * sqrt(pow(p.x2 - p.x3, 2) + pow(p.y2 - p.y3, 2));
	double A2 = (right - left) * (bottom - top);
	double s = sqrt(A1 / A2);
	width = round(s * (right - left) + 1);
	height = round(s * (bottom - top) + 1);

	centroid = Point(x, y);
}

string intToStr(int i, string path, int sz, string post)
{
	string bla = "";
	for (int i = 0; i < sz; i++)
		bla += "0";
	stringstream ss;
	ss << i;
	string ret = "";
	ss >> ret;
	string name = bla.substr(0, bla.size() - ret.size());
	name = path + name + ret + post;
	return name;
}
void visualize(Mat& img, VOTPolygon p)
{
	int h, w;
	Point cen;
	convert8To4(p, cen, w, h);
	rectangle(img, Rect(cen.x - w / 2, cen.y - h / 2, w, h), Scalar(0, 255, 0), 2);
}

void generateFiles(string dataset, int begin, int end, int count)
{
	string outDir = "";	//the path to the generated images.txt, region.txt (beside the binary tracker_vot)
	ifstream fin2((outDir + "output.txt").c_str());
	string fileName = intToStr(1, dataset, count, ".jpg");
	char x;
	ofstream images((outDir + "images.txt").c_str());

	for (int i = begin; i < end; i++)	//20000
	{
		string fileName = intToStr(i, dataset, count, ".jpg");
		Mat img = imread(fileName);
		if (!img.data)
			break;
		images << fileName << endl;
	}
	//177.68,183.11,175.57,127.67,226.96,125.55,229.07,180.99
	//196.34,158.07,198.21,116.09,243.61,117.94,241.73,159.93

	/*	ofstream region((outDir + "region.txt").c_str());
	ifstream fgin((dataset + "groundtruth.txt").c_str());
	GT currentGT;
	string line;
	getline(fgin, line);
	fgin >> currentGT.blx_ >> x >> currentGT.bly_ >> x >> currentGT.tlx_ >> x >> currentGT.tly_ >> x >> currentGT.trx_ >> x >> currentGT.try_ >> x >> currentGT.brx_ >> x >> currentGT.bry_;

	region << line << endl;
	region.close();*/
	images.close();
}


int main(int argc, char **argv)
{
	cv::Mat initFrame;

	string working_directory = "D://Scene_DS//VIVID//";
	string datasetNames[] = { "egtest01//", "egtest02//", "egtest03//", "egtest04//", "egtest05//","redteam//" };
	int nFrames[] = { 1820, 1300, 2570, 1832, 1763, 1917 };
	int currentDS = 1;
	string initFile = working_directory + datasetNames[currentDS] + "InitMulti.txt";

	std::vector<target> targets;
	ifstream fin(initFile.c_str());
	int ntargets, firstFrame;
	int x, y, w, h;
	while (!fin.eof())
	{
		fin >> firstFrame >> ntargets;
		for (int j = 0; j < ntargets; j++)
		{
			fin >> x >> y >> w >> h;
			target t(x, y, w, h, firstFrame);
			targets.push_back(t);
		}
	}
	fin.close();
	
	int currentTarget = 0;
	ofstream fout("region.txt");
	fout << targets[currentTarget].init.x << "," << targets[currentTarget].init.y << "," << targets[currentTarget].init.x + targets[currentTarget].init.width << "," << targets[currentTarget].init.y << ",";
	fout << targets[currentTarget].init.x + targets[currentTarget].init.width << "," << targets[currentTarget].init.y + targets[currentTarget].init.height << "," << targets[currentTarget].init.x << "," << targets[currentTarget].init.y + targets[currentTarget].init.height << endl;
	fout.close();

	//load region, images and prepare for output
	generateFiles(working_directory + datasetNames[currentDS] + "frame", targets[currentTarget].firstFrame, nFrames[currentDS], 5);

	//generateFiles("/media/New Volume/Scene_DS/VOT/fernando/");
	VOT vot_io("region.txt", "images.txt", "output.txt");
	VOTPolygon p = vot_io.getInitPolygon();

	Point centroid;
	w, h;
	convert8To4(p, centroid, w, h);
	initFrame;
	bool resize_image = 0;//sqrt((w * h)) >= 100;
	double resizeFactor = 1;//resize_image ? 2 : 1;
	//cout<<resize_image<<endl;
	vot_io.getNextImage(initFrame);
	if (resize_image)
	{
		centroid.x /= 2;
		centroid.y /= 2;
		w /= 2;
		h /= 2;
		resize(initFrame, initFrame, Size(), 0.5, 0.5);
	}
	KCFTracker KCF;
	//cout << "Start " << h << "  " << w << endl;

	KCF.preprocess(initFrame, centroid, w, h);
	vot_io.outputPolygon(p);
	//cout << "Done preprocess" << endl;
	int frameCnt = 1;
	while (true)
	{
		Mat frame;
		int nextFrame = vot_io.getNextImage(frame);

		if (nextFrame != 1 || !frame.data)
			break;
		Mat displayImg = frame.clone();
		if (resize_image)
		{
			resize(frame, frame, Size(), 0.5, 0.5);
		}
		frameCnt++;

		Rect rect;
		rect = KCF.processFrame(frame);

		rect.x *= resizeFactor;
		rect.y *= resizeFactor;
		rect.width *= resizeFactor;
		rect.height *= resizeFactor;
		//cout << "Done Frame " << frameCnt << endl;
		//visualize(displayImg, gt[i]);
		putText(displayImg, intToStr(frameCnt, "", 0, ""), Point(10, 10), 1, 1, Scalar(0, 255, 0), 3);
		rectangle(displayImg, rect, Scalar(255, 0, 0), 2);
		imshow("", displayImg);
		waitKey(1);

		VOTPolygon result;

		result.x1 = rect.x;
		result.y1 = rect.y;
		result.x2 = rect.x + rect.width;
		result.y2 = rect.y;
		result.x3 = rect.x + rect.width;
		result.y3 = rect.y + rect.height;
		result.x4 = rect.x;
		result.y4 = rect.y + rect.height;

		vot_io.outputPolygon(result);

	}

}

