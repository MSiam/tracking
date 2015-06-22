/*
 * KCFTracker.h
 *
 *  Created on: May 6, 2015
 *      Author: Sara
 */

#ifndef KCFTRACKER_H_
#define KCFTRACKER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
class KCFTracker
{
private:

	struct HOGParams
	{
		int binSize;
		int scaleBinSize;
		int nOrients;
		int softBin;
		float clipHog;
		HOGParams()
		{
			binSize = 1;
			scaleBinSize = 4;
			nOrients = 9;
			clipHog = 0.2;
			softBin = -1;
		}
	};
	struct trackingSetup
	{
		cv::Mat trans_cos_win;
		cv::Mat scale_cos_win;

		cv::Mat transFourier;
		cv::Mat scaleFourier;
		cv::Mat* model_alphaf; //one 2D Mat w*h
		cv::Mat* model_xf; //nChns 2D mat w*h

		int nNumTrans;
		cv::Mat *num_trans;
		cv::Mat den_trans;
		int nNumScale;
		cv::Mat *num_scale; // w*h 1D Mat 1*numScales
		cv::Mat den_scale; // one 1D Mat 1*numScales

		double *scaleFactors;
		cv::Size scale_model_sz;

		float min_scale_factor;
		float max_scale_factor;

		float current_scale_factor;

		cv::Point centroid;
		cv::Size original;
		cv::Size padded;

	};

	struct Params
	{
		double padding; //extra area surrounding the target
		double lambda; //regularization

		double output_sigma_factor; //spatial bandwidth (proportional to target)
		double interp_factor; //linear interpolation factor for adaptation

		double kernel_sigma; //gaussian kernel bandwidth

		//for scaling
		int number_scales;
		double scale_step;

		double scale_model_max_area;
		double scale_sigma_factor;
		double scale_learning_rate;

		bool enableScaling;

		Params()
		{
			padding = 1;
			lambda = 1e-4;
			output_sigma_factor = 0.1;
			interp_factor = 0.02;
			kernel_sigma = 0.5;

			number_scales = 33;
			scale_step = 1.02;
			scale_model_max_area = 512;
			scale_sigma_factor = 1.0 / 4;
			scale_learning_rate = 0.025;

			enableScaling = false;

		}

	};
	Params tParams;
	trackingSetup tSetup;
	HOGParams hParams;
	void train(cv::Mat img, bool first = false);
	void gaussian_correlation(cv::Mat* xf, cv::Mat* yf, int nChns, double sigma, cv::Mat & corrF);
	cv::Point displayFloat(cv::Mat img);
	void createFourier(cv::Mat original, cv::Mat& complexI, int flag = 0);
	void gaussian_shaped_labels(double sigma, int sz_w, int sz_h, cv::Mat& shiftedFilter);
	void hann(int size, cv::Mat& arr);
	float *convertTo1DFloatArray(cv::Mat &patch);
	void inverseFourier(cv::Mat original, cv::Mat& output, int flag = 0);
	cv::Mat get_scale_sample(cv::Mat img, int &nDims, bool display = false);
public:
	KCFTracker();
	cv::Mat *createFeatureMap2(cv::Mat& patch, int &nChns, bool isScaling = false);
	cv::Mat *createFeatureMap(cv::Mat& patch, int &nChns, bool isScaling = false);
	virtual ~KCFTracker();
	void preprocess(cv::Mat img, cv::Point centroid, int w, int h);
	cv::Rect processFrame(cv::Mat img);
};

#endif /* KCFTRACKER_H_ */
