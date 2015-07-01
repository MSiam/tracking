/*
 * DSST.cpp
 *
 *  Created on: 22 May, 2015
 *      Author: Sara & Mennatullah
 */

#include "DSST.h"

#define eps 0.0001

// unit vectors used to compute gradient orientation
static double uu[9] = { 1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397 }; 
static double vv[9] = { 0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420 };

/*static inline double round(double num)
{
     return (num > 0.0) ? floor(num + 0.5) : ceil(num - 0.5);
}*/
static inline float min(float x, float y)
{
	return (x <= y ? x : y);
}
static inline float max(float x, float y)
{
	return (x <= y ? y : x);
}

/*static inline double round(double num)
{
     return (num > 0.0) ? floor(num + 0.5) : ceil(num - 0.5);
}*/
static inline int min(int x, int y)
{
	return (x <= y ? x : y);
}
static inline int max(int x, int y)
{
	return (x <= y ? y : x);
}

cv::Mat DSSTTracker::inverseFourier(cv::Mat original, int flag)
{
	Mat output;
	cv::idft(original, output, DFT_REAL_OUTPUT|DFT_SCALE);  // Applying DFT without padding
	return output;
}

cv::Mat DSSTTracker::createFourier(cv::Mat original, int flag)
{
	Mat planes[] = {Mat_<double>(original), Mat::zeros(original.size(), CV_64F)};
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);
	cv::dft(complexI, complexI, flag);  // Applying DFT without padding
	return complexI;
}

Mat DSSTTracker::hann(int size)
{
	cv::Mat arr(size, 1, CV_32FC1);
	float multiplier;
	for (int i = 0; i < size; i++) 
	{
		multiplier = 0.5 * (1 - cos(2*M_PI*i/(size-1)));
		* ((float * )(arr.data +i*arr.step[0])) = multiplier;
	}
	return arr;
}

float *DSSTTracker::convert1DArray(Mat &patch)
{
	float *img=  (float*)calloc(patch.rows*patch.cols*3,sizeof(float));

	int k=0;
	for(int i=0; i<patch.cols; i++)
		for(int j=0; j<patch.rows; j++)
		{
			cv::Vec3b vc= patch.at<cv::Vec3b>(j, i);
			img[k]= (float)vc[2];
			img[k+patch.cols*patch.rows]= (float)vc[1];
			img[k+patch.cols*patch.rows*2]= (float)vc[0];

			k++;
		}
	return img;
}

Mat DSSTTracker::convert2DImage(float *arr, int w, int h)
{
	int k=0;
	Mat img(h, w, CV_32F);
	
	for(int i=0; i<img.cols; i++)
		for(int j=0; j<img.rows; j++)
		{
			img.at<float>(j, i)= arr[k];
			k++;
		}
		
	Mat imgU;
	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
	img -= minVal;
	img.convertTo(imgU,CV_8U,255.0/(maxVal-minVal));
	
	return imgU;
}
Point DSSTTracker::ComputeMaxDisplayfl(Mat &img,string winName)
{
	Mat imgU;
	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
		
	return maxLoc;
}

Mat *DSSTTracker::create_feature_map(Mat& patch, int full, int &nChns, Mat& Gray, bool scaling)
{
	int h = patch.rows, w = patch.cols;
	float* M = (float*) calloc(h*w,sizeof(float));
	float* O = (float*) calloc(h*w,sizeof(float));
		
	float *img= convert1DArray(patch);
	gradMag(img, M ,O, h, w, 3, full );
	if(!scaling)
	{
		hParams.binSize = 1;
	}
	else
	{
		hParams.binSize= 4;
	}
	int hb = h/hParams.binSize; int wb = w/hParams.binSize;
	nChns = hParams.nOrients*3+5;
	float *H= (float*) calloc(hb*wb*nChns,sizeof(float));
	fhogSSE( M, O, H, h, w, hParams.binSize, hParams.nOrients, hParams.softBin, hParams.clipHog );
	int l=0;
	Mat *featureMap;
	if(!scaling)
	{
		nChns=28;
		featureMap= new Mat[nChns];
		for(int i=0; i<nChns; i++)
			featureMap[i]= cv::Mat(hb, wb, CV_32FC1);

		Gray.convertTo(featureMap[0],CV_32FC1);
		for(int j=0; j<wb; j++)
			for(int i=0; i<hb; i++)
				for(int k=0; k<nChns-1; k++)
					featureMap[k+1].at<float>(i, j)= H[k*(hb*wb)+j*hb+i];
	}
	else
	{
		nChns=31;
		featureMap= new Mat[nChns];
		for(int i=0; i<nChns; i++)
			featureMap[i]= cv::Mat(hb, wb, CV_32FC1);

		for(int j=0; j<wb; j++)
			for(int i=0; i<hb; i++)
				for(int k=0; k<nChns; k++)
					featureMap[k].at<float>(i, j)= H[k*(hb*wb)+j*hb+i];

			
	}
	
	free(img);
	free(H);
	free(M);
	free(O);
		
	return featureMap;
}

double *DSSTTracker::convertTo1DFloatArrayDouble(Mat &patch)
{
	
	double *img = (double*) calloc(patch.rows * patch.cols*3, sizeof(double));

	int k=0;
	for(int i=0; i<patch.cols; i++)
		for(int j=0; j<patch.rows; j++)
		{
			cv::Vec3b vc= patch.at<cv::Vec3b>(j, i);
			img[k]= (double)vc[2];
			img[k+patch.cols*patch.rows]= (double)vc[1];
			img[k+patch.cols*patch.rows*2]= (double)vc[0];

			k++;
		}

	/*
	 imshow("", patch);
	 waitKey();
	 for (int i = 0; i < patch.rows * patch.cols; i++)
	 cout << img[i] << endl;
	 cout << "here" << endl;*/
	return img;
}

float *fhog2(double *im, int* dims, int sbin)
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
	//cout<<"Feat "<<out[0]<<" "<<out[1]<<endl;

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

Mat *DSSTTracker::create_feature_map2(Mat& patch, int full, int &nChns, Mat& Gray, bool scaling)
{
	int h = patch.rows, w = patch.cols;
	
	int binSize = hParams.binSize;
	int hb = h / binSize;
	int wb = w / binSize;

	Mat padPatch = patch;

	int totalHPad = ceil(patch.rows/binSize + 1.5) * binSize - patch.rows;
	int top = totalHPad/2;
	int bottom = totalHPad - top;
	
	int totalWPad = ceil(patch.cols/binSize + 1.5) * binSize - patch.cols;
	int left = totalWPad / 2;
	int right = totalWPad - left;
	
	copyMakeBorder(patch, padPatch, top, bottom, left, right, BORDER_REPLICATE);
	double* imgD = convertTo1DFloatArrayDouble(padPatch);
	int dims[] ={ padPatch.rows, padPatch.cols };
	
	//cout << "henaaa " << wb << "   " << hb << endl;
	//cout << "Cosine window "<<tSetup.trans_cos_win.size() << endl;
	
	float* H = fhog2(imgD, dims, binSize);
	
	int l=0;
	Mat *featureMap;
	if(!scaling)
	{
		nChns=28;
		featureMap= new Mat[nChns];
		for(int i=0; i<nChns; i++)
			featureMap[i]= cv::Mat(hb, wb, CV_32FC1);

		Gray.convertTo(featureMap[0],CV_32FC1);
		for(int j=0; j<wb; j++)
			for(int i=0; i<hb; i++)
				for(int k=0; k<nChns-1; k++)
					featureMap[k+1].at<float>(i, j)= H[k*(hb*wb)+j*hb+i];
	}
	else
	{
		nChns=31;
		featureMap= new Mat[nChns];
		for(int i=0; i<nChns; i++)
			featureMap[i]= cv::Mat(hb, wb, CV_32FC1);

		for(int j=0; j<wb; j++)
			for(int i=0; i<hb; i++)
				for(int k=0; k<nChns; k++)
					featureMap[k].at<float>(i, j)= H[k*(hb*wb)+j*hb+i];

			
	}
	
	free(imgD);
	free(H);
	return featureMap;
}

Mat DSSTTracker::get_scale_sample(Mat img, trackingSetup tSetup, Params tParams, int &nDims,bool display)
{
	Mat featureMapScale;
	CvRect patchSize;

	Mat roiGray;
	Mat roiResized;
	Mat feature_map_scale_fourier;
	for(int i=0; i<tParams.number_scales; i++)
	{
		float pw= tSetup.scaleFactors[i]*tSetup.current_scale_factor*tSetup.original.width;
		float ph= tSetup.scaleFactors[i]*tSetup.current_scale_factor*tSetup.original.height;
		
		int tlX1, tlY1, w1, w2, h1, h2;
		tlX1= max(0,(int)ceil(tSetup.centroid.x+1-pw/2));
		int padToX = (int)ceil(tSetup.centroid.x+1-pw/2) < 0 ? (int)ceil(tSetup.centroid.x+1-pw/2) : 0;
		w1 = (padToX + pw);
		w2 = (tlX1 + w1) >= img.cols ? img.cols - tlX1 : w1;
	
		tlY1= max(0,(int)ceil(tSetup.centroid.y+1-ph/2));
		int padToY = (int)ceil(tSetup.centroid.y+1-ph/2) < 0 ? (int)ceil(tSetup.centroid.y+1-ph/2) : 0;
		h1 = (padToY + ph);
		h2 = (tlY1 + h1) >= img.rows ? img.rows - tlY1 : h1;
		
		Rect rect(tlX1,tlY1,w2,h2);
		Mat patch = img(rect);
		Mat roi;
			
		copyMakeBorder(patch, roi,abs(padToY),h1 - h2, abs(padToX), w1-w2, BORDER_REPLICATE);
		Rect patchSize= Rect(tlX1, tlY1, roi.cols, roi.rows);
		
		int interpolation;
		if(tSetup.scale_model_sz.width > patchSize.width)
			interpolation = INTER_LINEAR;
		else
			interpolation = INTER_AREA;
		resize(roi,roiResized,cv::Size(tSetup.scale_model_sz.width, tSetup.scale_model_sz.height),0,0,interpolation);
		
		int nChns;
		#ifdef SSE	
			Mat m= Mat();
			Mat *featureMap= create_feature_map(roiResized,1, nChns, m, true);
		#else
			Mat m= Mat();
			Mat *featureMap= create_feature_map2(roiResized,1, nChns, m, true);
		#endif
		
		float s= tSetup.scale_cos_win.at<float>(i,0);
		
		if(featureMapScale.data== NULL)//i==0)
		{
			featureMapScale= Mat(featureMap[0].rows*featureMap[0].cols*nChns, tParams.number_scales, CV_32FC1);
			feature_map_scale_fourier= Mat(featureMap[0].rows*featureMap[0].cols*nChns, tParams.number_scales, CV_32FC1);
		}

		int k=0;
			
		for(int j=0; j<nChns; j++)
		{
			for(int m=0; m<featureMap[j].cols ; m++)
				for(int l=0; l<featureMap[j].rows ; l++)
				{
					featureMapScale.at<float>(k, i)= featureMap[j].at<float>(l, m) * s;
					k++;	
				}
		}
		delete []featureMap;
		if(display){
		imshow("roi",roi);
		waitKey();}
	}
	Mat featureMapTempDouble;
	featureMapScale.convertTo(featureMapTempDouble, CV_64FC1);
		
	Mat feature_map_scale_fourier_temp= createFourier(featureMapTempDouble, DFT_ROWS);
		
	nDims= tParams.number_scales;

	return feature_map_scale_fourier_temp;
}

Mat *DSSTTracker::get_translation_sample(cv::Mat img, trackingSetup tSet, int &nDims)
{
	float pw= tSetup.padded.width*tSet.current_scale_factor;
	float ph= tSetup.padded.height*tSet.current_scale_factor;
	int centerX= tSetup.centroid.x+1;
	int centerY= tSetup.centroid.y+1;
	
	int tlX1, tlY1, w1, w2, h1, h2;
	tlX1= max(0,(int)ceil(centerX-pw/2));
	int padToX = (int)ceil(centerX-pw/2) < 0 ? (int)ceil(centerX-pw/2) : 0;
	w1 = (padToX + pw);
	w2 = (tlX1 + w1) >= img.cols ? img.cols - tlX1 : w1;
	
	tlY1= max(0,(int)ceil(centerY-ph/2));
	int padToY = (int)ceil(centerY-ph/2) < 0 ? (int)ceil(centerY-ph/2) : 0;
	h1 = (padToY + ph);
	h2 = (tlY1 + h1) >= img.rows ? img.rows - tlY1 : h1;

	Rect rect(tlX1,tlY1,w2,h2);
	Mat patch = img(rect);
	Mat roi;
	copyMakeBorder(patch, roi,abs(padToY),h1 - h2, abs(padToX), w1-w2, BORDER_REPLICATE);
	
	Rect patchSize= Rect(tlX1, tlY1, roi.cols, roi.rows);

	int interpolation;
	if(tSetup.padded.width > patchSize.width)
		interpolation = INTER_LINEAR;
	else
		interpolation = INTER_AREA;
		
	resize(roi,roi,cv::Size(tSetup.padded.width, tSetup.padded.height),0,0,interpolation);
	cv::Mat roiGray;
	cv::cvtColor(roi, roiGray, CV_BGR2GRAY);

	Mat roiGrayFlot(roiGray.rows, roiGray.cols, CV_32FC1);
	roiGray.convertTo(roiGrayFlot,CV_32FC1);
	roiGrayFlot= roiGrayFlot.mul((float)1/255);
	roiGrayFlot= roiGrayFlot-0.5;

	int hb, wb, nChns;
	#ifdef SSE	
		Mat *featureMap= create_feature_map(roi,1, nChns, roiGrayFlot, false);
	#else
		Mat *featureMap= create_feature_map2(roi,1, nChns, roiGrayFlot, false);
	#endif

	nDims= nChns;
	
	for(int i=0; i<nChns; i++)
		featureMap[i] = featureMap[i].mul(tSetup.trans_cos_win);

	return featureMap;
}

void DSSTTracker::train(bool first, cv::Mat img)
{
	//Model update:
	//1- Extract samples ftrans and fscale from It at pt and st .
	//A- Extract translation sample
	int nDims=0;
	Mat *feature_map= get_translation_sample(img, tSetup, nDims);
		
	//B- Compute Denominator Translation, Numerator Translation
	Mat *feature_map_fourier= new Mat[nDims];
	Mat *num= new Mat[nDims];
	
	Mat den(feature_map[0].rows, feature_map[0].cols, CV_64FC2);
	den = cv::Mat::zeros(feature_map[0].rows, feature_map[0].cols, CV_64FC2);

	for (int i=0; i<nDims; i++)
	{
		Mat feature_map_double(feature_map[i].rows, feature_map[i].cols, CV_64FC1);
		feature_map[i].convertTo(feature_map_double, CV_64FC1);
		feature_map_fourier[i]= createFourier(feature_map_double);
		mulSpectrums(tSetup.transFourier, feature_map_fourier[i], num[i], 0, true);

		Mat temp;
		mulSpectrums(feature_map_fourier[i], feature_map_fourier[i], temp, 0, true);
		den= den+temp;
	}
	if(tSetup.enableScaling)
	{
		int nDimsScale;
		Mat feature_map_scale_fourier= get_scale_sample(img, tSetup, tParams, nDimsScale);//I have to convert featuremap to double first
		
		Mat *num_scale= new Mat[feature_map_scale_fourier.rows];
		Mat den_scale(1, nDimsScale, CV_64FC2);
		den_scale= cv::Mat::zeros(1, nDimsScale, CV_64FC2);
	
		for(int i=0; i<feature_map_scale_fourier.rows; i++)
		{
			Mat temp(1, nDimsScale, CV_64FC2);
			for(int j=0; j<nDimsScale; j++)
				temp.at<Vec2d>(0, j)= feature_map_scale_fourier.at<Vec2d>(i, j);
		
			mulSpectrums(tSetup.scaleFourier, temp, num_scale[i], 0, true);
		
		}


		Mat temp;
		mulSpectrums(feature_map_scale_fourier, feature_map_scale_fourier, temp, 0, true);
		
		for(int i=0; i<temp.cols; i++)
		{
			for(int j=0; j<temp.rows; j++)
			{
				den_scale.at<Vec2d>(0,i)[0] += temp.at<Vec2d>(j,i)[0];
				den_scale.at<Vec2d>(0,i)[1] += temp.at<Vec2d>(j,i)[1];
			}
		}
	
		//Update Our Model
		if(first)
		{
			tSetup.num_trans= num;
			tSetup.nNumTrans= nDims;
			tSetup.den_trans= den;

			tSetup.num_scale= num_scale;
			tSetup.nNumScale= nDimsScale;
			tSetup.den_scale= den_scale;
		}
		else
		{
			for(int i=0; i<tSetup.nNumTrans; i++)
				tSetup.num_trans[i]= tSetup.num_trans[i].mul(1- tParams.learning_rate) + num[i].mul(tParams.learning_rate);
			tSetup.den_trans= tSetup.den_trans.mul(1- tParams.learning_rate) + den.mul(tParams.learning_rate);

			for(int i=0; i<feature_map_scale_fourier.rows; i++)
				tSetup.num_scale[i]= tSetup.num_scale[i].mul(1- tParams.learning_rate) + num_scale[i].mul(tParams.learning_rate);
			tSetup.den_scale= tSetup.den_scale.mul(1- tParams.learning_rate) + den_scale.mul(tParams.learning_rate);

			delete[] num;
			delete[] num_scale;
		
		}
	}
	else
	{
		if(first)
		{
			tSetup.num_trans= num;
			tSetup.nNumTrans= nDims;
			tSetup.den_trans= den;

		}
		else
		{
			for(int i=0; i<tSetup.nNumTrans; i++)
				tSetup.num_trans[i]= tSetup.num_trans[i].mul(1- tParams.learning_rate) + num[i].mul(tParams.learning_rate);
			tSetup.den_trans= tSetup.den_trans.mul(1- tParams.learning_rate) + den.mul(tParams.learning_rate);

			delete[] num;
		
		}
	}
	delete[] feature_map;
	delete[] feature_map_fourier;
}

Point DSSTTracker::updateCentroid(Point oldC, int w , int h , int imgw, int imgh)
{	
	bool outBorder= false;
	int left = oldC.x - w/2 ;
	if(left<= 0)
	{
		left= 1;
		outBorder= true;
	}
	int top = oldC.y - h/2;
	if(top<=0)
	{
		top =1;
		outBorder= true;
	}

	if((left + w)>= imgw)
	{
		left = imgw- w-1;
		outBorder = true;
	}
		
	if((top + h) >= imgh)
	{
		top= imgh- h-1; 
		outBorder = true;
	}
	Point newPt;
	if(outBorder)
	{
		newPt.x= left + w/2;
		newPt.y= top + h/2;
	}
	else
		newPt = oldC;
	return newPt;
}

cv::Rect DSSTTracker::processFrame(cv::Mat img)
{
	int nDims = 0;
	Mat *feature_map= get_translation_sample(img, tSetup, nDims);	
	Mat *feature_map_fourier= new Mat[nDims];
	
	for (int i=0; i<nDims; i++)
	{
		Mat feature_map_double(feature_map[i].rows, feature_map[i].cols, CV_64FC1);
		feature_map[i].convertTo(feature_map_double, CV_64FC1);
		feature_map_fourier[i]= createFourier(feature_map_double);
	}
	
	Mat* temp = new Mat[nDims];
	for(int i = 0 ; i< nDims ; i++)
		mulSpectrums(tSetup.num_trans[i], feature_map_fourier[i], temp[i], 0, false);

	int w = tSetup.num_trans[0].cols, h = tSetup.num_trans[0].rows;

	Mat sumDen(h,w,CV_64F);
	
	for(int j = 0 ; j < h ; j++)
		for(int k = 0 ; k < w ; k++)
			sumDen.at<double>(j,k) = tSetup.den_trans.at<Vec2d>(j,k)[0] + tParams.lambda;

	Mat sumTemp(h,w,CV_64FC2);
	sumTemp= cv::Mat::zeros(sumTemp.size(), CV_64FC2);
	for(int j = 0 ; j < h ; j++)
		for(int k = 0 ; k < w ; k++)
		{
			for(int i = 0 ; i < nDims ; i++)
				sumTemp.at<Vec2d>(j,k) +=  temp[i].at<Vec2d>(j,k);

			sumTemp.at<Vec2d>(j,k) /= sumDen.at<double>(j,k);
		}

	Mat trans_response = cv::Mat::zeros(sumTemp.rows, sumTemp.cols, CV_64FC1);
	trans_response = inverseFourier(sumTemp);
		
		
	Point maxLoc = ComputeMaxDisplayfl(trans_response);
	tSetup.centroid.x +=  cvRound((maxLoc.x-tSetup.padded.width/2+1)*tSetup.current_scale_factor);
	tSetup.centroid.y +=  cvRound((maxLoc.y-tSetup.padded.height/2+1)*tSetup.current_scale_factor);

	tSetup.centroid = updateCentroid(tSetup.centroid,tSetup.original.width*tSetup.current_scale_factor,tSetup.original.height*tSetup.current_scale_factor,img.cols,img.rows);
	
	if(tSetup.enableScaling)
	{
		int nDimsScale;
	
		Mat feature_map_scale_fourier = get_scale_sample(img, tSetup, tParams, nDimsScale);//I have to convert featuremap to double first
		
		Mat* tempScale = new Mat[feature_map_scale_fourier.rows];
		
		for(int i=0; i<feature_map_scale_fourier.rows; i++)
		{
			Mat temp1(1, feature_map_scale_fourier.cols, CV_64FC2);
			for(int j=0; j<feature_map_scale_fourier.cols; j++)
				temp1.at<Vec2d>(0, j)= feature_map_scale_fourier.at<Vec2d>(i, j);

			mulSpectrums(tSetup.num_scale[i], temp1, tempScale[i],0, false);
		}
		w = nDimsScale;
		Mat sumDenScale(1,w,CV_64F);
		for(int k = 0 ; k < w ; k++)
			sumDenScale.at<double>(0,k) = tSetup.den_scale.at<Vec2d>(0,k)[0] + tParams.lambda;
		Mat sumTempScale(1,w,CV_64FC2);
		sumTempScale= cv::Mat::zeros(sumTempScale.size(), CV_64FC2);
		for(int k = 0 ; k < w ; k++)
		{
		
			for(int i = 0 ; i < feature_map_scale_fourier.rows ; i++)
				sumTempScale.at<Vec2d>(0,k) +=  tempScale[i].at<Vec2d>(0,k);

			sumTempScale.at<Vec2d>(0,k) /= sumDenScale.at<double>(0,k);
		}
		
		Mat scale_response = cv::Mat::zeros(1, nDimsScale, CV_64FC1);
		scale_response = inverseFourier(sumTempScale);
	
		Point maxLocScale = ComputeMaxDisplayfl(scale_response);

		tSetup.current_scale_factor= tSetup.current_scale_factor* tSetup.scaleFactors[maxLocScale.x];
		if(tSetup.current_scale_factor< tSetup.min_scale_factor)
			tSetup.current_scale_factor= tSetup.min_scale_factor;
		if(tSetup.current_scale_factor> tSetup.max_scale_factor)
			tSetup.current_scale_factor= tSetup.max_scale_factor;
	
		delete[] tempScale;
	
	}
	
	train(false, img);

	tSetup.centroid = updateCentroid(tSetup.centroid,tSetup.original.width*tSetup.current_scale_factor,tSetup.original.height*tSetup.current_scale_factor,img.cols,img.rows);
	int left = tSetup.centroid.x - (tSetup.original.width/2*tSetup.current_scale_factor) ;
	int top = tSetup.centroid.y - (tSetup.original.height/2*tSetup.current_scale_factor);
	cv::Rect rect(left, top, tSetup.original.width*tSetup.current_scale_factor, tSetup.original.height*tSetup.current_scale_factor);
		
	delete[] feature_map;
	delete[] feature_map_fourier;
	delete[] temp;
		
	return rect;
}

/*DSSTTracker::~DSSTTracker()
{
	if(tSetup.num_trans!=0)
	{
		cout<<"here "<<tSetup.num_trans<<endl;
		delete[] tSetup.num_trans;
	}
	if (tSetup.enableScaling)
	{
		delete[] tSetup.num_scale;
		delete[] tSetup.scaleFactors;
	}
}*/

DSSTTracker::DSSTTracker()
{
	tSetup.num_trans=0;
	tSetup.enableScaling= false;
	tSetup.num_scale=0;
}
void DSSTTracker::preprocess(int rows,int cols, cv::Mat img, cv::Rect bb)
{
	tSetup.enableScaling= false;
	tSetup.centroid.x = bb.x+bb.width/2;
	tSetup.centroid.y = bb.y+bb.height/2;
	tSetup.original.width = bb.width+1;
	tSetup.original.height = bb.height+1;
		
	//0- Preprocessing
	//A- Create Translation Gaussian Filters
	tSetup.padded.width= floor(tSetup.original.width * (1 + tParams.padding));
	tSetup.padded.height = floor(tSetup.original.height * (1 + tParams.padding));
	int szPadding_w = tSetup.padded.width;
	int szPadding_h = tSetup.padded.height;
		
	//don't change to float cause it causes huge truncation errors :) .. asdna el transFilter
	float transSigma= sqrt(float(tSetup.original.width*tSetup.original.height))*tParams.output_sigma_factor;
	cv::Mat transFilter(szPadding_h,szPadding_w,CV_64FC1);
	for(int r = -szPadding_h/2 ; r < ceil((double)szPadding_h/2) ; r++)
		for(int c = -szPadding_w/2 ; c < ceil((double)szPadding_w/2) ; c++)
			transFilter.at<double>(r+szPadding_h/2,c+szPadding_w/2) = exp(-0.5 * ((double)((r+1)*(r+1) + (c+1)*(c+1)) / (transSigma*transSigma)));
			
	tSetup.transFourier= createFourier(transFilter);
	
	//B- Create Scale Gaussian Filters
	//We don't know why this equation?!
	double scaleSigma= tParams.number_scales/ sqrtf(tParams.number_scales) * tParams.scale_sigma_factor;
	cv::Mat scaleFilter(1, tParams.number_scales, CV_64FC1);
	for(int r = -tParams.number_scales/2 ; r < ceil((double)tParams.number_scales/2) ; r++)
		scaleFilter.at<double>(0,r+tParams.number_scales/2)= exp(-0.5 * ((double)(r*r) / (scaleSigma*scaleSigma)));
	tSetup.scaleFourier = createFourier(scaleFilter);

	//Create Cosine Windows to give less weight to boundarie
	cv::Mat trans_cosine_win(szPadding_h, szPadding_w, CV_32FC1);
	cv::Mat cos1= hann(szPadding_h);
	cv::Mat cos2= hann(szPadding_w);
	tSetup.trans_cos_win= cos1*cos2.t();
	cv::Mat scale_cosine_win(tParams.number_scales, 1, CV_32FC1);
	tSetup.scale_cos_win= hann(tParams.number_scales);

	//Create Scale Factors
	double *scaleFactors= new double[tParams.number_scales];
	for(int i = 1 ; i <= tParams.number_scales ; i++)
		scaleFactors[i-1] = pow(tParams.scale_step , (ceil((double)tParams.number_scales/2) - i));
	tSetup.scaleFactors= scaleFactors;

	//compute the resize dimensions used for feature extraction in the scale estimation
	float scale_model_factor = 1;
	int area = tSetup.original.width*tSetup.original.height;
	if ( area > tParams.scale_model_max_area)
		scale_model_factor = sqrt((double)tParams.scale_model_max_area/area);
	
	
	tSetup.scale_model_sz = Size(floor(tSetup.original.width * scale_model_factor), floor(tSetup.original.height*scale_model_factor));
		
	// find maximum and minimum scales
	//why 
	tSetup.min_scale_factor = pow(tParams.scale_step , ceil(log(max(5.0/szPadding_h,5.0/szPadding_w)) / log(tParams.scale_step)));
	tSetup.max_scale_factor = pow(tParams.scale_step , floor(log(min((float)rows/tSetup.original.height,(float)cols/tSetup.original.width)) / log(tParams.scale_step)));
	tSetup.current_scale_factor= 1;

	hParams.binSize=1;
	hParams.nOrients= 9;
	hParams.clipHog= 0.2;
	hParams.softBin= -1;

	train(true, img);
}

cv::Mat DSSTTracker::visualize(Rect rect, cv::Mat img,Scalar scalar)
{
	Mat retImg = img.clone();
	cv::rectangle(retImg, cvPoint(rect.x, rect.y), cvPoint(rect.x+rect.width, rect.y+rect.height), scalar, 2);
	return retImg;
}
