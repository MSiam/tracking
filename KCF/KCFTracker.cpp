/*
 * KCFTracker.cpp
 *
 *  Created on: May 6, 2015
 *      Author: Sara
 */
#define _USE_MATH_DEFINES
#include <math.h>
#include "KCFTracker.h"
#include <iostream>
#include "stdio.h"
#include "HOG.h"
#include <fstream>
#include <iomanip>
#include "defines.h"
using namespace cv;
using namespace std;

// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
double uu[9] =
{ 1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397 };
double vv[9] =
{ 0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420 };

static inline float min(float x, float y)
{
	return (x <= y ? x : y);
}
static inline float max(float x, float y)
{
	return (x <= y ? y : x);
}

static inline int min(int x, int y)
{
	return (x <= y ? x : y);
}
static inline int max(int x, int y)
{
	return (x <= y ? y : x);
}

// main function:
// takes a double color image and a bin size
// returns HOG features
float *fhog(double *im, int* dims, int sbin)
{

	// memory for caching orientation histograms & their norms
	int blocks[2];
	blocks[0] = (int) round((double) dims[0] / (double) sbin);
	blocks[1] = (int) round((double) dims[1] / (double) sbin);
	float *hist = (float *) calloc(blocks[0] * blocks[1] * 18, sizeof(float));
	float *norm = (float *) calloc(blocks[0] * blocks[1], sizeof(float));

	// memory for HOG features
	int out[3];
	out[0] = max(blocks[0] - 2, 0);
	out[1] = max(blocks[1] - 2, 0);
	out[2] = 27 + 4 + 1;
	float *feat = (float*) malloc(out[0] * out[1] * out[2] * sizeof(float));

	int visible[2];
	visible[0] = blocks[0] * sbin;
	visible[1] = blocks[1] * sbin;
	//cout << "Blaaa " << dims[0] << "  " << blocks[0] << "  " << out[0] << "   " << dims[1] << "  " << blocks[1] << "  " << out[1] << endl;
	//cout<<"pixels "<<visible[0] - 1<<"  "<<visible[1] - 1<<endl;
	for (int x = 1; x < visible[1] - 1; x++)
	{
		for (int y = 1; y < visible[0] - 1; y++)
		{
			// first color channel
			double *s = im + min(x, dims[1] - 2) * dims[0] + min(y, dims[0] - 2);
			double dy = *(s + 1) - *(s - 1);
			double dx = *(s + dims[0]) - *(s - dims[0]);
			double v = dx * dx + dy * dy;

			// snap to one of 18 orientations
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; o++)
			{
				double dot = uu[o] * dx + vv[o] * dy;
				if (dot > best_dot)
				{
					best_dot = dot;
					best_o = o;
				}
				else if (-dot > best_dot)
				{
					best_dot = -dot;
					best_o = o + 9;
				}
			}

			// add to 4 histograms around pixel using linear interpolation
			double xp = ((double) x + 0.5) / (double) sbin - 0.5;
			double yp = ((double) y + 0.5) / (double) sbin - 0.5;
			int ixp = (int) floor(xp);
			int iyp = (int) floor(yp);
			double vx0 = xp - ixp;
			double vy0 = yp - iyp;
			double vx1 = 1.0 - vx0;
			double vy1 = 1.0 - vy0;
			v = sqrt(v);

			if (ixp >= 0 && iyp >= 0)
			{
				*(hist + ixp * blocks[0] + iyp + best_o * blocks[0] * blocks[1]) += vx1 * vy1 * v;
			}

			if (ixp + 1 < blocks[1] && iyp >= 0)
			{
				*(hist + (ixp + 1) * blocks[0] + iyp + best_o * blocks[0] * blocks[1]) += vx0 * vy1 * v;
			}

			if (ixp >= 0 && iyp + 1 < blocks[0])
			{
				*(hist + ixp * blocks[0] + (iyp + 1) + best_o * blocks[0] * blocks[1]) += vx1 * vy0 * v;
			}

			if (ixp + 1 < blocks[1] && iyp + 1 < blocks[0])
			{
				*(hist + (ixp + 1) * blocks[0] + (iyp + 1) + best_o * blocks[0] * blocks[1]) += vx0 * vy0 * v;
			}
		}
	}

	// compute energy in each block by summing over orientations
	for (int o = 0; o < 9; o++)
	{
		float *src1 = hist + o * blocks[0] * blocks[1];
		float *src2 = hist + (o + 9) * blocks[0] * blocks[1];
		float *dst = norm;
		float *end = norm + blocks[1] * blocks[0];
		while (dst < end)
		{
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

	// compute features
	for (int x = 0; x < out[1]; x++)
	{
		for (int y = 0; y < out[0]; y++)
		{
			float *dst = feat + x * out[0] + y;
			float *src, *p, n1, n2, n3, n4;

			p = norm + (x + 1) * blocks[0] + y + 1;
			n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + (x + 1) * blocks[0] + y;
			n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x * blocks[0] + y + 1;
			n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x * blocks[0] + y;
			n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);

			float t1 = 0;
			float t2 = 0;
			float t3 = 0;
			float t4 = 0;

			// contrast-sensitive features
			src = hist + (x + 1) * blocks[0] + (y + 1);
			for (int o = 0; o < 18; o++)
			{
				float h1 = min(*src * n1, 0.2);
				float h2 = min(*src * n2, 0.2);
				float h3 = min(*src * n3, 0.2);
				float h4 = min(*src * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst += out[0] * out[1];
				src += blocks[0] * blocks[1];
			}

			// contrast-insensitive features
			src = hist + (x + 1) * blocks[0] + (y + 1);
			for (int o = 0; o < 9; o++)
			{
				float sum = *src + *(src + 9 * blocks[0] * blocks[1]);
				float h1 = min(sum * n1, 0.2);
				float h2 = min(sum * n2, 0.2);
				float h3 = min(sum * n3, 0.2);
				float h4 = min(sum * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				dst += out[0] * out[1];
				src += blocks[0] * blocks[1];
			}

			// texture features
			*dst = 0.2357 * t1;
			dst += out[0] * out[1];
			*dst = 0.2357 * t2;
			dst += out[0] * out[1];
			*dst = 0.2357 * t3;
			dst += out[0] * out[1];
			*dst = 0.2357 * t4;

			// truncation feature
			dst += out[0] * out[1];
			*dst = 0;
		}
	}

	free(hist);
	free(norm);
	return feat;
}

KCFTracker::KCFTracker()
{

}

KCFTracker::~KCFTracker()
{

}

Point ComputeMaxfl(Mat img)
{
	Mat imgU = img.clone();
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);

	/*imgU -= minVal;
	 imgU.convertTo(imgU, CV_8U, 255.0 / (maxVal - minVal));
	 imshow("FloatImg", imgU);
	 waitKey();*/
	return maxLoc;
}
Point updateCentroid(Point oldC, int w, int h, int imgw, int imgh)
{
	bool outBorder = false;
	int left = oldC.x - w / 2;
	if (left <= 0)
	{
		left = 1;
		outBorder = true;
	}
	int top = oldC.y - h / 2;
	if (top <= 0)
	{
		top = 1;
		outBorder = true;
	}

	if ((left + w) >= imgw)
	{
		left = imgw - w - 1;
		outBorder = true;
	}

	if ((top + h) >= imgh)
	{
		top = imgh - h - 1;
		outBorder = true;
	}
	Point newPt;
	if (outBorder)
	{
		newPt.x = left + w / 2;
		newPt.y = top + h / 2;
	}
	else
		newPt = oldC;
	return newPt;
}

Point KCFTracker::displayFloat(Mat img)
{
	Mat imgU = img.clone();
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
	imgU -= minVal;
	imgU.convertTo(imgU, CV_8U, 255.0 / (maxVal - minVal));
	imshow("FloatImg", imgU);

	return maxLoc;
}

void KCFTracker::createFourier(cv::Mat original, cv::Mat& complexI, int flag)
{
	Mat planes[] =
	{ Mat_<double>(original), Mat::zeros(original.size(), CV_64F) };

	cv::merge(planes, 2, complexI);
	cv::dft(complexI, complexI, flag);  // Applying DFT without padding
	//cout << "Fourier" << endl;
}
void KCFTracker::gaussian_shaped_labels(double sigma, int sz_w, int sz_h, Mat& shiftedFilter)
{
	cv::Mat transFilter(sz_h, sz_w, CV_64FC1);
	shiftedFilter = Mat(sz_h, sz_w, CV_64FC1);
	//TODO slow access
	for (int r = -sz_h / 2; r < ceil((double) sz_h / 2); r++)
		for (int c = -sz_w / 2; c < ceil((double) sz_w / 2); c++)
			transFilter.at<double>(r + sz_h / 2, c + sz_w / 2) = exp(-0.5 * ((double) ((r + 1) * (r + 1) + (c + 1) * (c + 1)) / (sigma * sigma)));

	//labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
	int shiftX = -sz_w / 2.0 + 1;
	int shiftY = -sz_h / 2.0 + 1;

	for (int i = 0; i < transFilter.rows; i++)
		for (int j = 0; j < transFilter.cols; j++)
		{
			shiftedFilter.at<double>(i, j) = transFilter.at<double>((i - shiftY) % transFilter.rows, (j - shiftX) % transFilter.cols);
		}
	if(shiftedFilter.at<double>(0,0) != 1)
		cout<<"SHIFT ERROR"<<endl;
}
void KCFTracker::hann(int size, Mat& arr)
{
	arr = Mat(size, 1, CV_64FC1);
	float multiplier;
	for (int i = 0; i < size; i++)
	{
		multiplier = 0.5 * (1 - cos(2 * M_PI * i / (size - 1)));
		*((double *) (arr.data + i * arr.step[0])) = multiplier;
	}
}
float *KCFTracker::convertTo1DFloatArray(Mat &patch)
{

	float *img = (float*) calloc(patch.rows * patch.cols, sizeof(float));

	int k = 0;
	for (int i = 0; i < patch.cols; i++)
		for (int j = 0; j < patch.rows; j++)
			img[k++] = (float) patch.at<unsigned char>(j, i) / 255.0;

	/*
	 imshow("", patch);
	 waitKey();
	 for (int i = 0; i < patch.rows * patch.cols; i++)
	 cout << img[i] << endl;
	 cout << "here" << endl;*/
	return img;
}
double *convertTo1DFloatArrayDouble(Mat &patch)
{

	double *img = (double*) calloc(patch.rows * patch.cols, sizeof(double));

	int k = 0;
	for (int i = 0; i < patch.cols; i++)
		for (int j = 0; j < patch.rows; j++)
			img[k++] = (double) patch.at<unsigned char>(j, i) / 255.0;

	/*
	 imshow("", patch);
	 waitKey();
	 for (int i = 0; i < patch.rows * patch.cols; i++)
	 cout << img[i] << endl;
	 cout << "here" << endl;*/
	return img;
}

Mat *KCFTracker::createFeatureMap2(Mat& patch, int &nChns, bool isScaling)
{

	int h = patch.rows, w = patch.cols;

	int binSize = isScaling ? hParams.scaleBinSize : hParams.binSize;
	int hb = h / binSize;
	int wb = w / binSize;

	Mat padPatch = patch;
/*
       int totalHPad = ceil((patch.rows/binSize + 1.5) * binSize) - patch.rows;
       int bottom = totalHPad/2;
       int top = totalHPad - bottom;
       
       int totalWPad =ceil((patch.cols/binSize + 1.5) * binSize) - patch.cols;
       int right = totalWPad / 2;
       int left = totalWPad - right;
	   */
	int totalHPad = binSize + 2 - (patch.rows % binSize);
	int top = totalHPad / 2;
	int bottom = totalHPad - top;

	int totalWPad = binSize + 2 - (patch.cols % binSize);
	int left = totalWPad / 2;
	int right = totalWPad - left;
	//cout << "TopBottom " << top << "  " << bottom << "  LeftRight "<<left<<"  "<<right<<endl;
	copyMakeBorder(patch, padPatch, top, bottom, left, right, BORDER_REPLICATE);
	//cout << "new Size" << padPatch.size() << endl;
	double* imgD = convertTo1DFloatArrayDouble(padPatch);
	int dims[] =
	{ padPatch.rows, padPatch.cols };
	hb = (int) round((double) dims[0] / (double) binSize) - 2;
	wb = (int) round((double) dims[1] / (double) binSize) - 2;
	//cout << "henaaa " << wb << "   " << hb << endl;
	//cout << tSetup.trans_cos_win.size() << endl;
	float* H = fhog(imgD, dims, 4);
	/*imshow("patch", patch);
	 imshow("patchPad", padPatch);
	 waitKey();*/
	Mat *featureMap;
	nChns = 31;
	featureMap = new Mat[nChns];

	for (int i = 0; i < nChns; i++)
		featureMap[i] = cv::Mat(hb, wb, CV_64FC1);

	for (int j = 0; j < wb; j++)
		for (int i = 0; i < hb; i++)
			for (int k = 0; k < nChns; k++)
				featureMap[k].at<double>(i, j) = H[k * (hb * wb) + j * hb + i];

	if (!isScaling)
		for (int i = 0; i < nChns; i++)
			featureMap[i] = featureMap[i].mul(tSetup.trans_cos_win);


	free(imgD);
	free(H);

	return featureMap;

}
Mat *KCFTracker::createFeatureMap(Mat& patch, int &nChns, bool isScaling)
{

	int h = patch.rows, w = patch.cols;
	float* M = (float*) calloc(h * w, sizeof(float));
	float* O = (float*) calloc(h * w, sizeof(float));
	
	float *img = convertTo1DFloatArray(patch);

	gradMag(img, M, O, h, w, 1, 1);

	int binSize = isScaling ? hParams.scaleBinSize : hParams.binSize;
	int hb = h / binSize;
	int wb = w / binSize;

	nChns = hParams.nOrients * 3 + 5;
	float *H = (float*) calloc(hb * wb * nChns, sizeof(float));
	//cout << hb << "  " << wb << endl;
	//cerr << "b4 fhogSSE" << endl;
	fhogSSE(M, O, H, h, w, binSize, hParams.nOrients, hParams.softBin, hParams.clipHog);
	//cerr << "after fhogSSE" << endl;
	Mat *featureMap;
	
	nChns = 28;
	featureMap = new Mat[nChns];
	for (int i = 0; i<nChns; i++)
		featureMap[i] = cv::Mat(hb, wb, CV_64FC1);

	patch.convertTo(featureMap[0], CV_64FC1);
	for (int j = 0; j<wb; j++)
		for (int i = 0; i<hb; i++)
			for (int k = 0; k<nChns - 1; k++)
				featureMap[k + 1].at<double>(i, j) = H[k*(hb*wb) + j*hb + i];
	/*
	nChns = 31;
	featureMap = new Mat[nChns];
	for (int i = 0; i < nChns; i++)
		featureMap[i] = cv::Mat(hb, wb, CV_64FC1);

	for (int j = 0; j < wb; j++)
		for (int i = 0; i < hb; i++)
			for (int k = 0; k < nChns; k++)
				featureMap[k].at<double>(i, j) = H[k * (hb * wb) + j * hb + i];
*/
	if (!isScaling)
		for (int i = 0; i < nChns; i++)
			featureMap[i] = featureMap[i].mul(tSetup.trans_cos_win);

	/*
	 freopen("log.txt", "wt", stdout);

	 for (int k = 0; k < nChns; k++)
	 {
	 for (int j = 0; j < wb; j++)
	 {
	 for (int i = 0; i < hb; i++)
	 cout <<  fixed << setprecision(4)<<featureMap[k].at<double>(i, j) << endl;
	 }
	 }

	 */

	free(img);
	free(H);
	free(M);
	free(O);

	return featureMap;
}
void KCFTracker::inverseFourier(cv::Mat original, cv::Mat& output, int flag)
{
	cv::idft(original, output, DFT_REAL_OUTPUT | DFT_SCALE);
}
void KCFTracker::gaussian_correlation(Mat* xf, Mat* yf, int nChns, double sigma, Mat & corrF)
{
	//ofstream fout("log.txt");
	int w = xf[0].cols;
	int h = xf[0].rows;
	double xx = 0;
	double yy = 0;
	for (int i = 0; i < nChns; i++)
		for (int j = 0; j < h; j++)
			for (int k = 0; k < w; k++)
			{
				Vec2d bla = xf[i].at<Vec2d>(j, k);
				xx += bla[0] * bla[0] + bla[1] * bla[1];
				bla = yf[i].at<Vec2d>(j, k);
				yy += bla[0] * bla[0] + bla[1] * bla[1];
			}
	xx /= (w * h);
	yy /= (w * h);
	//cout << xx << "  " << yy << endl;
	Mat *xyf = new Mat[nChns];
	Mat corr = cv::Mat::zeros(h, w, CV_64FC1);
	for (int ch = 0; ch < nChns; ch++)
	{
		mulSpectrums(xf[ch], yf[ch], xyf[ch], 0, true);
		inverseFourier(xyf[ch], xyf[ch]);
	}
	/*
	 for (int i = 0; i < nChns; i++)
	 for (int k = 0; k < w; k++)
	 for (int j = 0; j < h; j++)
	 fout << fixed << setprecision(4) << xyf[i].at<double>(j, k) << endl;*/

	for (int i = 0; i < nChns; i++)
		corr += xyf[i];
	/*	for (int k = 0; k < w; k++)
	 for (int j = 0; j < h; j++)
	 fout << fixed << setprecision(4) << corr.at<double>(j, k) << endl;*/

	corr *= -2;
	corr += xx + yy;
	corr /= (w * h * nChns);
	max(corr, 0);
	corr *= (-1 / (sigma * sigma));
	exp(corr, corr);

	/*for (int k = 0; k < w; k++)
	 for (int j = 0; j < h; j++)
	 fout << fixed << setprecision(4) << corr.at<double>(j, k) << endl;*/
	createFourier(corr, corrF);
	/*
	 for (int j = 0; j < corrF.cols; j++)
	 {
	 for (int i = 0; i < corrF.rows; i++)
	 {
	 string complexN = (corrF.at<Vec2d>(i, j)[1] <= 0 ? "" : "+");
	 if (abs(corrF.at<Vec2d>(i, j)[1]) != 0)
	 fout << fixed << setprecision(4) << corrF.at<Vec2d>(i, j)[0] << complexN << fixed << setprecision(4) << corrF.at<
	 Vec2d>(i, j)[1] << "i" << endl;
	 else
	 fout << fixed << setprecision(4) << corrF.at<Vec2d>(i, j)[0] << endl;
	 }
	 }
	 fout.close();*/
	delete[] xyf;
}
void KCFTracker::train(Mat img, bool first)
{
	double trainTime = 0;
//Create Patch
	float pw = tSetup.padded.width * tSetup.current_scale_factor;
	float ph = tSetup.padded.height * tSetup.current_scale_factor;
	int centerX = tSetup.centroid.x + 1;
	int centerY = tSetup.centroid.y + 1;

	int tlX1, tlY1, w1, w2, h1, h2;
	tlX1 = max(0.0, centerX - floor(pw / 2.0));
	int padToX = (int) centerX - pw / 2 < 0 ? (int) ceil(centerX - pw / 2) : 0;
	w1 = (padToX + pw);
	w2 = (tlX1 + w1) >= img.cols ? img.cols - tlX1 : w1;

	tlY1 = max(0, (int) ceil(centerY - ph / 2));
	int padToY =
			(int) ceil(centerY - ph / 2) < 0 ? (int) ceil(centerY - ph / 2) : 0;
	h1 = (padToY + ph);
	h2 = (tlY1 + h1) >= img.rows ? img.rows - tlY1 : h1;

	Rect rect(tlX1, tlY1, w2, h2);
	Mat patch = img(rect);

	Mat roi;
	double subwindow;
	timeOfBlock( copyMakeBorder(patch, roi, abs(padToY), h1 - h2, abs(padToX), w1 - w2, BORDER_REPLICATE);, subwindow);
	trainTime += subwindow;
	int interpolation;
	if (tSetup.padded.width > roi.cols)
		interpolation = INTER_LINEAR;
	else
		interpolation = INTER_AREA;
	resize(roi, roi, cv::Size(tSetup.padded.width, tSetup.padded.height), 0, 0, interpolation);

	int nChns;
	Mat* feature_map;
	double transFeatureMap;
	timeOfBlock( feature_map = createFeatureMap(roi, nChns);, transFeatureMap);
	trainTime += transFeatureMap;

	Mat *feature_map_fourier;
	feature_map_fourier = new Mat[nChns];
//cout<<"Feature Map Fourier Translation"<<endl;
	double trainFourier;
	timeOfBlock( for (int i = 0; i < nChns; i++) createFourier(feature_map[i],feature_map_fourier[i] );, trainFourier);
	trainTime += trainFourier;

	/*freopen("log.txt", "wt", stdout);

	 for (int k = 0; k < nChns; k++)
	 {
	 for (int j = 0; j < feature_map_fourier[k].cols; j++)
	 {
	 for (int i = 0; i < feature_map_fourier[k].rows; i++)
	 {
	 string complexN = (feature_map_fourier[k].at<Vec2d>(i, j)[1] <= 0 ? "" : "+");
	 if (abs(feature_map_fourier[k].at<Vec2d>(i, j)[1]) != 0)
	 cout << fixed << setprecision(4) << feature_map_fourier[k].at<Vec2d>(i, j)[0] << complexN << fixed << setprecision(4) << feature_map_fourier[k].at<
	 Vec2d>(i, j)[1] << "i" << endl;
	 else
	 cout << fixed << setprecision(4) << feature_map_fourier[k].at<Vec2d>(i, j)[0] << endl;
	 }
	 }
	 }*/
	Mat corr;
	double transCorr;
	timeOfBlock( gaussian_correlation(feature_map_fourier, feature_map_fourier, nChns, tParams.kernel_sigma, corr);, transCorr);
	trainTime += transCorr;

	Mat temp, temp2;
	Mat corrLambda = corr + tParams.lambda;
	Mat* alpha = new Mat(corrLambda.size(), corrLambda.type());
	mulSpectrums(tSetup.transFourier, corrLambda, temp, true);
	mulSpectrums(corrLambda, corrLambda, temp2, true);

	for (int i = 0; i < corr.rows; i++)
		for (int j = 0; j < corr.cols; j++)
			alpha->at<Vec2d>(i, j) = temp.at<Vec2d>(i, j) / (
					abs(temp2.at<Vec2d>(i, j)[0]) == 0.0 ? 1 : temp2.at<Vec2d>(i, j)[0]);

	/*	freopen("log.txt", "wt", stdout);
	 for (int j = 0; j < alpha->cols; j++)
	 {
	 for (int i = 0; i < alpha->rows; i++)
	 {
	 string complexN = (alpha->at < Vec2d > (i, j)[1] <= 0 ? "" : "+");
	 if (abs(alpha->at < Vec2d > (i, j)[1]) != 0)
	 cout << fixed << setprecision(4) << alpha->at < Vec2d > (i, j)[0] << complexN << fixed << setprecision(4) << alpha->at < Vec2d > (i, j)[1] << "i" << endl;
	 else
	 cout << fixed << setprecision(4) << alpha->at < Vec2d > (i, j)[0] << endl;
	 }

	 }*/
	Mat *num_scale = 0;
	int nDimsScale;
	Mat den_scale;
	int nPixel;
	if (tParams.enableScaling)
	{
		Mat feature_map_scale_fourier;
		double scaleFeatures;
		timeOfBlock( feature_map_scale_fourier = get_scale_sample(img, nDimsScale);, scaleFeatures);
		trainTime += scaleFeatures;

		nPixel = feature_map_scale_fourier.rows;
		num_scale = new Mat[nPixel];
		den_scale = cv::Mat::zeros(1, nDimsScale, CV_64FC2);

		double scaleNumDen;
		timeOfBlock(
		for (int i = 0; i < feature_map_scale_fourier.rows; i++)
		{
			Mat temp(1, nDimsScale, CV_64FC2);
			for (int j = 0; j < nDimsScale; j++)
				temp.at<Vec2d>(0, j) = feature_map_scale_fourier.at<Vec2d>(i, j);

			mulSpectrums(tSetup.scaleFourier, temp, num_scale[i], 0, true);

		}

		Mat temp3;
		mulSpectrums(feature_map_scale_fourier, feature_map_scale_fourier, temp3, 0, true);

		for (int i = 0; i < temp3.cols; i++)
		{
			for (int j = 0; j < temp3.rows; j++)
			{
				den_scale.at<Vec2d>(0, i)[0] += temp3.at<Vec2d>(j, i)[0];
				den_scale.at<Vec2d>(0, i)[1] += temp3.at<Vec2d>(j, i)[1];
			}
		}
		,scaleNumDen);
		trainTime+=scaleNumDen;
	}
	double updateParam;
	timeOfBlock(
	if (first)
	{
		tSetup.model_alphaf = alpha;
		tSetup.model_xf = feature_map_fourier;

		if (tParams.enableScaling)
		{
			tSetup.num_scale = num_scale;
			tSetup.nNumScale = nDimsScale;
			tSetup.den_scale = den_scale;
		}
	}
	else
	{
		//model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
		//model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;

		*tSetup.model_alphaf = (1 - tParams.interp_factor) * (*tSetup.model_alphaf) + tParams.interp_factor * (*alpha);

		/*	freopen("alpha.txt", "wt", stdout);
		 for (int j = 0; j < tSetup.model_alphaf->cols; j++)
		 for (int i = 0; i < tSetup.model_alphaf->rows; i++)
		 {
		 string complexN = (tSetup.model_alphaf->at<Vec2d>(i, j)[1] <= 0 ? "" : "+");
		 if (abs(tSetup.model_alphaf->at<Vec2d>(i, j)[1]) != 0)
		 cout << fixed << setprecision(4) << tSetup.model_alphaf->at<Vec2d>(i, j)[0] << complexN << fixed << setprecision(4) << tSetup.model_alphaf->at<
		 Vec2d>(i, j)[1] << "i" << endl;
		 else
		 cout << fixed << setprecision(4) << tSetup.model_alphaf->at<Vec2d>(i, j)[0] << endl;
		 }
		 */
		for (int i = 0; i < nChns; i++)
			tSetup.model_xf[i] = (1 - tParams.interp_factor) * tSetup.model_xf[i] + tParams.interp_factor * feature_map_fourier[i];
		/*

		 freopen("xf.txt", "wt", stdout);
		 for (int k = 0; k < nChns; k++)
		 for (int j = 0; j < tSetup.model_xf[k].cols; j++)
		 {
		 for (int i = 0; i < tSetup.model_xf[k].rows; i++)
		 {
		 string complexN = (tSetup.model_xf[k].at<Vec2d>(i, j)[1] <= 0 ? "" : "+");
		 if (abs(tSetup.model_xf[k].at<Vec2d>(i, j)[1]) != 0)
		 cout << fixed << setprecision(4) << tSetup.model_xf[k].at<Vec2d>(i, j)[0] << complexN << fixed << setprecision(4) << tSetup.model_xf[k].at<
		 Vec2d>(i, j)[1] << "i" << endl;
		 else
		 cout << fixed << setprecision(4) << tSetup.model_xf[k].at<Vec2d>(i, j)[0] << endl;
		 }
		 }
		 */
		if (tParams.enableScaling)
		{
			for (int i = 0; i < nPixel; i++)
				tSetup.num_scale[i] = tSetup.num_scale[i].mul(1 - tParams.scale_learning_rate) + num_scale[i].mul(tParams.scale_learning_rate);
			tSetup.den_scale = tSetup.den_scale.mul(1 - tParams.scale_learning_rate) + den_scale.mul(tParams.scale_learning_rate);

			delete[] num_scale;
		}
		delete[] feature_map_fourier;
		delete alpha;
	},updateParam);
	trainTime += updateParam;
	//cerr << "			Train Time " << trainTime << endl;
	delete[] feature_map;
}
Mat KCFTracker::get_scale_sample(Mat img, int &nDims, bool display)
{
	Mat featureMapScale;
	CvRect patchSize;

	Mat roiGray;
	Mat roiResized;
	Mat feature_map_scale_fourier;
	for (int i = 0; i < tParams.number_scales; i++)
	{
		//Create Patch
		float pw = tSetup.scaleFactors[i] * tSetup.current_scale_factor * tSetup.original.width;
		float ph = tSetup.scaleFactors[i] * tSetup.current_scale_factor * tSetup.original.height;

		int tlX1, tlY1, w1, w2, h1, h2;
		tlX1 = max(0, (int) ceil(tSetup.centroid.x + 1 - pw / 2));
		int padToX =
				(int) ceil(tSetup.centroid.x + 1 - pw / 2) < 0 ? (int) ceil(tSetup.centroid.x + 1 - pw / 2) : 0;
		w1 = (padToX + pw);
		w2 = (tlX1 + w1) >= img.cols ? img.cols - tlX1 : w1;

		tlY1 = max(0, (int) ceil(tSetup.centroid.y + 1 - ph / 2));
		int padToY =
				(int) ceil(tSetup.centroid.y + 1 - ph / 2) < 0 ? (int) ceil(tSetup.centroid.y + 1 - ph / 2) : 0;
		h1 = (padToY + ph);
		h2 = (tlY1 + h1) >= img.rows ? img.rows - tlY1 : h1;
		Rect rect(tlX1, tlY1, w2, h2);
		Mat patch = img(rect);
		Mat roi;

		copyMakeBorder(patch, roi, abs(padToY), h1 - h2, abs(padToX), w1 - w2, BORDER_REPLICATE);

		Rect patchSize = Rect(tlX1, tlY1, roi.cols, roi.rows);

		int interpolation;
		if (tSetup.scale_model_sz.width > patchSize.width)
			interpolation = INTER_LINEAR;
		else
			interpolation = INTER_AREA;
		resize(roi, roiResized, cv::Size(tSetup.scale_model_sz.width, tSetup.scale_model_sz.height), 0, 0, interpolation);
		if (display)
		{
			imshow("roi", roi);
			waitKey();
		}
		//Extract Features
		int nChns;
		Mat *featureMap = createFeatureMap(roiResized, nChns, true);

		float s = tSetup.scale_cos_win.at<double>(i, 0);

		//Multiply by scale window + Save it as 1D array in the big array
		if (featureMapScale.data == NULL) //i==0)
		{
			featureMapScale = Mat(featureMap[0].rows * featureMap[0].cols * nChns, tParams.number_scales, CV_64FC1);
			feature_map_scale_fourier = Mat(featureMap[0].rows * featureMap[0].cols * nChns, tParams.number_scales, CV_64FC1);
		}

		int k = 0;

		for (int j = 0; j < nChns; j++)
		{
			for (int m = 0; m < featureMap[j].cols; m++)
				for (int l = 0; l < featureMap[j].rows; l++)
				{
					featureMapScale.at<double>(k, i) = featureMap[j].at<double>(l, m) * s;
					k++;
				}
		}
		delete[] featureMap;

	}

	Mat feature_map_scale_fourier_temp;
	createFourier(featureMapScale, feature_map_scale_fourier_temp, DFT_ROWS);

	nDims = tParams.number_scales;

	return feature_map_scale_fourier_temp;
}

void KCFTracker::preprocess(Mat imgOrig, Point centroid, int w, int h)
{
	double convertGray;
	Mat img;
	timeOfBlock( cv::cvtColor(imgOrig, img, CV_BGR2GRAY);, convertGray);
	int rows = img.rows;
	int cols = img.cols;
	tSetup.centroid.x = centroid.x;
	tSetup.centroid.y = centroid.y;
	tSetup.original = Size(w, h);

	tSetup.padded.width = floor(tSetup.original.width * (1.0 + tParams.padding));
	tSetup.padded.height = floor(tSetup.original.height * (1.0 + tParams.padding));

	////////////////Localization Parameters/////////////////

	//output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	double output_sigma = sqrt(tSetup.original.width * tSetup.original.height) * tParams.output_sigma_factor / hParams.binSize;

	int sz_w = tSetup.padded.width / hParams.binSize;
	int sz_h = tSetup.padded.height / hParams.binSize;
	//cout<<sz_w<<"  "<<sz_h<<endl;
	//yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
	//tSetup.transFourier = cv::Mat::zeros(tSetup.transFourier.size(), tSetup.transFourier.type());
	double filterTrans;
	Mat gauss;
	timeOfBlock( gaussian_shaped_labels(output_sigma, sz_w, sz_h,gauss); createFourier(gauss,tSetup.transFourier);

	//cout << "hereafter" << endl;
	//TODO remove
	/*	ofstream fout("log.txt");
	 for (int j = 0; j < tSetup.transFourier.cols; j++)
	 {
	 for (int i = 0; i < tSetup.transFourier.rows; i++)
	 {
	 string complexN = (tSetup.transFourier.at < Vec2d > (i, j)[1] <= 0 ? "" : "+");
	 if (abs(tSetup.transFourier.at < Vec2d > (i, j)[1]) != 0)
	 fout << fixed << setprecision(4) << tSetup.transFourier.at < Vec2d > (i, j)[0] << complexN << fixed << setprecision(4) << tSetup.transFourier.at < Vec2d > (i, j)[1] << "i" << endl;
	 else
	 fout << fixed << setprecision(4) << tSetup.transFourier.at < Vec2d > (i, j)[0] << endl;
	 }
	 }
	 fout.close();*/

	//cos_window = hann(size(yf,1)) * hann(size(yf,2))';
	cv::Mat trans_cosine_win(sz_h, sz_w, CV_64FC1);Mat cos1; hann(sz_h,cos1); cv::Mat cos2; hann(sz_w,cos2); tSetup.trans_cos_win = cos1 * cos2.t();, filterTrans);
	double scalingParam;
	////////////////Scaling Parameters/////////////////
	timeOfBlock( if (tParams.enableScaling) {
	//B- Create Scale Gaussian Filters

	double scaleSigma = tParams.number_scales / sqrt(tParams.number_scales) * tParams.scale_sigma_factor; cv::Mat scaleFilter(1, tParams.number_scales, CV_64FC1); for (int r = -tParams.number_scales / 2; r < ceil((double) tParams.number_scales / 2); r++) scaleFilter.at<double>(0, r + tParams.number_scales / 2) = exp(-0.5 * ((double) (r * r) / (scaleSigma * scaleSigma))); createFourier(scaleFilter, tSetup.scaleFourier);

	cv::Mat scale_cosine_win(tParams.number_scales, 1, CV_64FC1); hann(tParams.number_scales, tSetup.scale_cos_win);

	double *scaleFactors = new double[tParams.number_scales]; for (int i = 1; i <= tParams.number_scales; i++) scaleFactors[i - 1] = pow(tParams.scale_step, (ceil((double) tParams.number_scales / 2) - i));

	tSetup.scaleFactors = scaleFactors;

	//compute the resize dimensions used for feature extraction in the scale estimation
	float scale_model_factor = 1; int area = tSetup.original.width * tSetup.original.height; if (area > tParams.scale_model_max_area) scale_model_factor = sqrt((double) tParams.scale_model_max_area / area);

	tSetup.scale_model_sz = Size(floor(tSetup.original.width * scale_model_factor), floor(tSetup.original.height * scale_model_factor));
	ofstream fout("log.txt");
	fout << "heeere" << endl;
	fout.close();
	// find maximum and minimum scales
	tSetup.min_scale_factor = pow(tParams.scale_step, ceil(log(max(5.0 / tSetup.padded.width, 5.0 / tSetup.padded.height)) / log(tParams.scale_step))); tSetup.max_scale_factor = pow(tParams.scale_step, floor(log(min((float) rows / tSetup.original.height, (float) cols / tSetup.original.width)) / log(tParams.scale_step))); }, scalingParam);
	tSetup.current_scale_factor = 1;
	train(img, true);
}
Rect KCFTracker::processFrame(cv::Mat imgOrig)
{

	double totTime = 0;
	double convertGray;
	Mat img;
	timeOfBlock(cv::cvtColor(imgOrig, img, CV_BGR2GRAY) ;, convertGray);
	totTime += convertGray;

	//Create Patch
	float pw = tSetup.padded.width * tSetup.current_scale_factor;
	float ph = tSetup.padded.height * tSetup.current_scale_factor;
	int centerX = tSetup.centroid.x + 1;
	int centerY = tSetup.centroid.y + 1;
	//cout << "Centre " << centerX << "  " << centerY << endl;
	int tlX1, tlY1, w1, w2, h1, h2;
	tlX1 = max(0.0, centerX - floor(pw / 2.0));
	int padToX = (int) centerX - pw / 2 < 0 ? (int) ceil(centerX - pw / 2) : 0;
	w1 = (padToX + pw);
	w2 = (tlX1 + w1) >= img.cols ? img.cols - tlX1 : w1;

	tlY1 = max(0, (int) ceil(centerY - ph / 2));
	int padToY =
			(int) ceil(centerY - ph / 2) < 0 ? (int) ceil(centerY - ph / 2) : 0;
	h1 = (padToY + ph);
	h2 = (tlY1 + h1) >= img.rows ? img.rows - tlY1 : h1;

	Rect rect(tlX1, tlY1, w2, h2);
	Mat patch = img(rect);

	Mat roi;
	copyMakeBorder(patch, roi, abs(padToY), h1 - h2, abs(padToX), w1 - w2, BORDER_REPLICATE);
	int interpolation;
	if (tSetup.padded.width > roi.cols)
		interpolation = INTER_LINEAR;
	else
		interpolation = INTER_AREA;
	//imshow("roi",roi);
	resize(roi, roi, cv::Size(tSetup.padded.width, tSetup.padded.height), 0, 0, interpolation);
	//imshow("roiResi",roi);
	//zf = fft2(get_features(patch, features, cell_size, cos_window));

	int nChns;
	Mat* feature_map;
	double getFeatureMap;
	timeOfBlock(feature_map = createFeatureMap(roi, nChns) ;, getFeatureMap);
	totTime += getFeatureMap;

	Mat *feature_map_fourier = new Mat[nChns];
	double featuresFourier;
	//cout<<"Feature Map Fourier Translation"<<endl;
	timeOfBlock(for (int i = 0; i < nChns; i++) createFourier(feature_map[i], feature_map_fourier[i]) ;, featuresFourier);
	totTime += featuresFourier;

	double corrTime;
	timeOfBlock( Mat corr; gaussian_correlation(feature_map_fourier, tSetup.model_xf, nChns, tParams.kernel_sigma, corr); Mat temp; mulSpectrums(corr, *tSetup.model_alphaf, temp, false);

	inverseFourier(temp, temp); double mxVal = -1; Point mxLoc;

	//[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
	Point delta = ComputeMaxfl(temp); int w = temp.cols; int h = temp.rows;
	//ofstream fout("log.txt", ofstream::app);
	//fout<<delta.x<<","<<delta.y<<","<<fixed<<setprecision(4)<< temp.at<double>(delta)<<endl;
	if (delta.x > w / 2 - 1) delta.x -= w; if (delta.y > h / 2 - 1) delta.y -= h;
	//pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
	tSetup.centroid = tSetup.centroid + (hParams.binSize * Point(delta.x, delta.y));, corrTime);
	tSetup.centroid = updateCentroid(tSetup.centroid, tSetup.original.width * tSetup.current_scale_factor, tSetup.original.height * tSetup.current_scale_factor, img.cols, img.rows);
	totTime += corrTime;

	//Scaling estimation

	if (tParams.enableScaling)
	{
		int nDimsScale;

		Mat feature_map_scale_fourier;
		double scaleFeatures;
		timeOfBlock(feature_map_scale_fourier = get_scale_sample(img, nDimsScale);, scaleFeatures);
		totTime += scaleFeatures;

		Mat* tempScale = new Mat[feature_map_scale_fourier.rows];
		double scaleEst;
		timeOfBlock( for (int i = 0; i < feature_map_scale_fourier.rows; i++) { Mat temp1(1, feature_map_scale_fourier.cols, CV_64FC2); for (int j = 0; j < feature_map_scale_fourier.cols; j++) temp1.at<Vec2d>(0, j) = feature_map_scale_fourier.at<Vec2d>(i, j);

		mulSpectrums(tSetup.num_scale[i], temp1, tempScale[i], 0, false); }

		Mat sumDenScale(1, nDimsScale, CV_64F); for (int k = 0; k < nDimsScale; k++) sumDenScale.at<double>(0, k) = tSetup.den_scale.at<Vec2d>(0, k)[0] + tParams.lambda;

		Mat sumTempScale(1, nDimsScale, CV_64FC2); sumTempScale = cv::Mat::zeros(sumTempScale.size(), CV_64FC2); for (int k = 0; k < nDimsScale; k++) { for (int i = 0; i < feature_map_scale_fourier.rows; i++) sumTempScale.at<Vec2d>(0, k) += tempScale[i].at<Vec2d>(0, k);

		sumTempScale.at<Vec2d>(0, k) /= sumDenScale.at<double>(0, k); }

		Mat scale_response = cv::Mat::zeros(1, nDimsScale, CV_64FC1); inverseFourier(sumTempScale, scale_response); Point maxLocScale = ComputeMaxfl(scale_response);

		tSetup.current_scale_factor = tSetup.current_scale_factor * tSetup.scaleFactors[maxLocScale.x];, scaleEst);
		totTime += scaleEst;

		if (tSetup.current_scale_factor < tSetup.min_scale_factor)
			tSetup.current_scale_factor = tSetup.min_scale_factor;
		if (tSetup.current_scale_factor > tSetup.max_scale_factor)
			tSetup.current_scale_factor = tSetup.max_scale_factor;

	}

	//fout << delta.x<<","<<delta.y<<","<<tSetup.centroid.x << "," << tSetup.centroid.y << endl;
	double trainTime;
	timeOfBlock( train(img);, trainTime);
	totTime += trainTime;
	Point centroid = updateCentroid(tSetup.centroid, tSetup.original.width * tSetup.current_scale_factor, tSetup.original.height * tSetup.current_scale_factor, img.cols, img.rows); //to make sure not out of boundary
	int left = centroid.x - (tSetup.original.width / 2 * tSetup.current_scale_factor);
	int top = centroid.y - (tSetup.original.height / 2 * tSetup.current_scale_factor);
	rect = Rect(left, top, tSetup.original.width * tSetup.current_scale_factor, tSetup.original.height * tSetup.current_scale_factor);
	//fout.close();
	//cerr << "			Calc Process Time: " << totTime << endl;
	tSetup.centroid = centroid;
	delete[] feature_map;
	delete[] feature_map_fourier;
	return rect;
}
