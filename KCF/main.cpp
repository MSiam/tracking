/*
 * main.cpp
 *
 *  Created on: May 6, 2015
 *      Author: Sara
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

void convert8To4(GT p, Point& centroid, int& width, int& height)
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
void visualize(Mat& img, GT p)
{
	int h, w;
	Point cen;
	convert8To4(p, cen, w, h);
	rectangle(img, Rect(cen.x - w / 2, cen.y - h / 2, w, h), Scalar(0, 255, 0), 2);
}
int main(int argc, char **argv)
{
	freopen("profiling.txt", "wt", stderr);
	int ds = 23;
	string datasest[] =
	{ "ball", "basketball", "bicycle", "bolt", "car", "david", "diving", "drunk", "fernando", "fish1", "fish2", "gymnastics", "hand1", "hand2", "jogging", "motocross", "polarbear", "skating", "sphere", "sunshade", "surfing", "torus", "trellis", "tunnel", "woman" };
	int frameCnt = 1;
	string base_dir = "D:/Scene_DS/VOT/" + datasest[ds] + "/";
	string fileName = intToStr(frameCnt, base_dir, 8, ".jpg");
	Mat initFrame = imread(fileName, 1);
	if (!initFrame.data)
		cout << "NULL" << endl;
	/*imshow("", initFrame);
	 waitKey();*/
	GT * gt = readGroundtruth(base_dir);
	Point centroid;
	int w, h;
	convert8To4(gt[0], centroid, w, h);
	KCFTracker KCF;
	cout << "Start " << h << "  " << w << endl;
	double preprocessTime;

	timeOfBlock(KCF.preprocess(initFrame, centroid, w, h);, preprocessTime);
	//cout << "Done preprocess" << endl;
	ofstream log("log.txt");
	log.close();
	double totTime = 0;
	for (int i = 0; i < 2000; i++)
	{
		frameCnt++;
		fileName = intToStr(frameCnt, base_dir, 8, ".jpg");
		Mat frame = imread(fileName, 1);
		//cout<<frameCnt<<endl;
		if (!frame.data)
			break;
		Mat displayImg = frame.clone();
		double processTime;
		Rect rect;
		timeOfBlock(rect = KCF.processFrame(frame);, processTime);
		totTime += processTime;
		//cout << "Done Frame " << frameCnt << endl;
		visualize(displayImg, gt[i]);
		rectangle(displayImg, rect, Scalar(255, 0, 0), 2);
		imshow("", displayImg);
		waitKey(1);
	}
	cout << frameCnt / (totTime / 1000) << endl;
	waitKey();

}

