// DSST.cpp : Defines the entry point for the console application.
//
#define _USE_MATH_DEFINES

#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include "Params.h"
#include "HOG.h"
#include "vot.hpp"
#include <windows.h>

using namespace std;
using namespace cv;

class DSSTTracker
{
	Params tParams;
	trackingSetup tSetup;
	HOGParams hParams;

public:
	/*~DSSTTracker()
	{
		delete[] tSetup.num_trans;
		delete[] tSetup.num_scale;
		delete[] tSetup.scaleFactors;
	}*/
	cv::Mat inverseFourier(cv::Mat original, int flag=0)
	{
		/*cout<<" INverse Fourier"<<endl;
		for(int i = 0 ; i < original.rows ; i++)
		{
			for(int j = 0 ; j < original.cols ; j++)
				cout<<i << "  "<<original.at<Vec2d>(i,j)<<"  ";
		cout<<endl;
		}*/
		Mat output;
		cv::idft(original, output, DFT_REAL_OUTPUT|DFT_SCALE);  // Applying DFT without padding
		//normalize(output, output, 0, 1, CV_MINMAX);

		/*for(int i = 0 ; i < output.rows ; i++)
		{
			for(int j = 0 ; j < output.cols ; j++)
				cout<<i << "  "<<output.at<double>(i,j)<<"  ";
		cout<<endl;
		}*/
		return output;//planes[0];
	}

	cv::Mat createFourier(cv::Mat original, int flag=0)
	{
		Mat planes[] = {Mat_<double>(original), Mat::zeros(original.size(), CV_64F)};
		cv::Mat complexI;
		cv::merge(planes, 2, complexI);
		cv::dft(complexI, complexI, flag);  // Applying DFT without padding
		//split(complexI,planes);
		/*for(int i = 0 ; i < complexI.rows ; i++){
			cout<<endl;
			for(int j = 0 ; j < complexI.cols ; j++)
				cout<<complexI.at<Vec2d>(i,j)<<"  ";
		}
		cout<<complexI.size()<<endl;
		*/
		//Mat complexIsmall;

		return complexI;
	}

	Mat hann(int size)
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

	string intToStr(int i, string path ,int sz , string post){
		string bla = "";
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

	float *convert1DArray(Mat &patch)
	{
		//float *img= new float[patch.rows*patch.cols*3];
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
		//for(int i=0; i<10; i++)
		//	cout<<img[i]<<endl;

		return img;
	}

	Mat convert2DImage(float *arr, int w, int h)
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
	Point ComputeMaxDisplayfl(Mat &img,string winName="FloatImg")
	{
		Mat imgU;
		double minVal; 
		double maxVal; 
		Point minLoc; 
		Point maxLoc;
		minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
		
		//img -= minVal;
		//img.convertTo(imgU,CV_8U,255.0/(maxVal-minVal));
		/*cout<<"Norm "<<endl;
		for(int i = 0 ; i < img.rows ; i++)
		{
			for(int j = 0 ; j < img.cols ; j++)
				cout<<i << "  "<<(int)img.at<uchar>(i,j)<<"  ";
		cout<<endl;
		}*/
		//imshow(winName,imgU);
		return maxLoc;
	}

	Mat *create_feature_map(Mat& patch, int full, int &nChns, Mat& Gray, bool scaling)
	{
		int h = patch.rows, w = patch.cols;
		//freopen("map.txt","wt",stdout);
		//float* M = new float[h*w] ,*O = new float[h*w];
		float* M = (float*) calloc(h*w,sizeof(float));
		float* O = (float*) calloc(h*w,sizeof(float));
		//imshow("Feature Map", patch);

		float *img= convert1DArray(patch);
		//fout<<"before manitude"<<endl;
		gradMag(img, M ,O, h, w, 3, full );
		//fout<<"after manitude"<<endl;
		/*imshow("featureMap",patch);
	
		waitKey();*/
		/*Mat magImg= convert2DImage(M, w, h);
		Mat orientImg= convert2DImage(O, w, h);
		ComputeMaxDisplayfl(magImg,"Mag");
		ComputeMaxDisplayfl(orientImg,"Ori");*/
		//imshow("Magnitude", magImg);
		//imshow("orientation", orientImg);

		if(!scaling)
		{
			hParams.binSize = 1;
		}
		else
		{
			hParams.binSize= 4;
		}
		int hb = h/hParams.binSize; int wb = w/hParams.binSize;
		//int hb = ceil((float)h/hParams.binSize); int wb = ceil((float)w/hParams.binSize);
		//const int hb= ceil((double)h/binSize), wb= ceil((double)w/binSize), nb=hb*wb, nbo=nb*nOrients;
		//int hb = cvRound((double)h/hParams.binSize); int wb = cvRound((double)w/hParams.binSize);
		nChns = hParams.nOrients*3+5;
		float *H= (float*) calloc(hb*wb*nChns,sizeof(float));
		//fout<<"before hog"<<hb<<" "<<wb<<" "<<hParams.binSize<<endl;
		//fout.close();
		fhogSSE( M, O, H, h, w, hParams.binSize, hParams.nOrients, hParams.softBin, hParams.clipHog );
		//fout<<"after hog"<<endl;
		//fout.close();
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
	
		/*delete[] img;
		delete[] H;
		delete[] M;
		delete[] O;*/
		free(img);
		free(H);
		free(M);
		free(O);
		
		return featureMap;
	}

	Mat get_scale_sample(Mat img, trackingSetup tSetup, Params tParams, int &nDims,bool display = false)
	{
		Mat featureMapScale;
		CvRect patchSize;

		Mat roiGray;
		Mat roiResized;
		Mat feature_map_scale_fourier;
		for(int i=0; i<tParams.number_scales; i++)
		{
			//Create Patch
			float pw= tSetup.scaleFactors[i]*tSetup.current_scale_factor*tSetup.original.width;
			float ph= tSetup.scaleFactors[i]*tSetup.current_scale_factor*tSetup.original.height;
			//cout<<"Size: "<<tSetup.original<<endl;
			//cout<<"Centroid: "<<tSetup.centroid<<endl;
		
			int tlX1, tlY1, w1, w2, h1, h2;
			tlX1= max(0,(int)ceil(tSetup.centroid.x+1-pw/2));
			int padToX = (int)ceil(tSetup.centroid.x+1-pw/2) < 0 ? (int)ceil(tSetup.centroid.x+1-pw/2) : 0;
			w1 = (padToX + pw);
			w2 = (tlX1 + w1) >= img.cols ? img.cols - tlX1 : w1;
	
			tlY1= max(0,(int)ceil(tSetup.centroid.y+1-ph/2));
			int padToY = (int)ceil(tSetup.centroid.y+1-ph/2) < 0 ? (int)ceil(tSetup.centroid.y+1-ph/2) : 0;
			h1 = (padToY + ph);
			h2 = (tlY1 + h1) >= img.rows ? img.rows - tlY1 : h1;
			//ofstream fout("C://Users//mincosy//Desktop//Tracking//Final_DSST+C++2//DSST//logzebala.txt",  std::ofstream::out | std::ofstream::app);
			
			//fout<<"Tleft , rows " << tlY1 <<"  "<<img.rows <<endl;
			//fout<<"Centroid in scale scample"<<tSetup.centroid.x<<"  "<< tSetup.centroid.y<<endl;
			//fout<<pw<<"   "<<ph<<"   "<<tlX1<<"  "<<tlY1<<"  "<< img.cols<<"  "<< img.rows<<endl;
			Rect rect(tlX1,tlY1,w2,h2);
			Mat patch = img(rect);
			Mat roi;
			//fout<<"Rect "<<rect.width<<"  "<<rect.height<<endl;
			copyMakeBorder(patch, roi,abs(padToY),h1 - h2, abs(padToX), w1-w2, BORDER_REPLICATE);
		
			/*patchSize.x= max(0,(int)ceil(tSetup.centroid.x+1 - pw/2));
			patchSize.y= max(0,(int)ceil(tSetup.centroid.y+1 - ph/2));		
			patchSize.width = min((int)pw,img.cols-patchSize.x);
			patchSize.height = min((int)ph,img.rows-patchSize.y);*/
		
			Rect patchSize= Rect(tlX1, tlY1, roi.cols, roi.rows);
		
			//cout<<"Current Scal Factor: "<<tSetup.current_scale_factor<<endl;
			//cout<<"Patch Size: "<<patchSize.x<<" "<<patchSize.y<<" "<<patchSize.width<<" "<<patchSize.height<<endl;

			int interpolation;
			if(tSetup.scale_model_sz.width > patchSize.width)
				interpolation = INTER_LINEAR;
			else
				interpolation = INTER_AREA;
			//fout<<"Resize Scale "<<roi.cols << "  "<<roi.rows<<endl;
			//fout.close();
			resize(roi,roiResized,cv::Size(tSetup.scale_model_sz.width, tSetup.scale_model_sz.height),0,0,interpolation);
		
			//Extract Features		
			int nChns;
			Mat *featureMap= create_feature_map(roiResized,1, nChns, Mat(), true);
		
			float s= tSetup.scale_cos_win.at<float>(i,0);
		
			//Multiply by scale window + Save it as 1D array in the big array
			if(featureMapScale.data== NULL)//i==0)
			{
				featureMapScale= Mat(featureMap[0].rows*featureMap[0].cols*nChns, tParams.number_scales, CV_32FC1);
				feature_map_scale_fourier= Mat(featureMap[0].rows*featureMap[0].cols*nChns, tParams.number_scales, CV_32FC1);
			}

			int k=0;
			//cout<<"Feature Map"<<endl;
			for(int j=0; j<nChns; j++)
			{
				for(int m=0; m<featureMap[j].cols ; m++)
					for(int l=0; l<featureMap[j].rows ; l++)
					{
						featureMapScale.at<float>(k, i)= featureMap[j].at<float>(l, m) * s;
						//cout<<featureMapScale.at<float>(k, i)<<" ";//featureMap[j].at<float>(l, m)<<" ";
						k++;	
					}
			}
			//cout<<endl;
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

	Mat *get_translation_sample(cv::Mat img, trackingSetup tSet, int &nDims)
	{
	
		//freopen("trans.txt","wt",stdout);
		//Create Patch
		float pw= tSetup.padded.width*tSet.current_scale_factor;//tSet.initialPadded.width*tSet.current_scale_factor;
		float ph= tSetup.padded.height*tSet.current_scale_factor;//tSet.initialPadded.height*tSet.current_scale_factor;
		int centerX= tSetup.centroid.x+1;//1 to compare with matlab //tSet.initialPadded.x+tSet.initialPadded.width/2;
		int centerY= tSetup.centroid.y+1;//tSet.initialPadded.y+tSet.initialPadded.height/2;
		//cout<<centerX<<"  "<<centerY<<endl;
		//CvRect patchSize= cvRect(max(0,(int)ceil(centerX-pw/2)), max(0,(int)ceil(centerY-ph/2)), pw, ph);

		//Mat roi = cv::Mat::zeros(ph, pw, img.type());
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
	
		//patchSize.width = min((int)pw,img.cols-patchSize.x);
		//patchSize.height = min((int)ph,img.rows-patchSize.y);
		//cout<<"Rect: "<<patchSize.x<<" "<<patchSize.y<<" "<<patchSize.width<<" "<<patchSize.height<<"   "<<pw<<"  "<<ph<<endl;
	
		//imshow("final", roi);
		//waitKey();
		Rect patchSize= Rect(tlX1, tlY1, roi.cols, roi.rows);

		int interpolation;
		if(tSetup.padded.width > patchSize.width)
			interpolation = INTER_LINEAR;
		else
			interpolation = INTER_AREA;
		//ofstream fout("C://Users//mincosy//Desktop//Tracking//Final_DSST+C++2//DSST//logzebala.txt",  std::ofstream::out | std::ofstream::app);
		//cout<<"Resize Trans "<<roi.cols << "  "<<roi.rows<<endl;
		//fout.close();
		resize(roi,roi,cv::Size(tSetup.padded.width, tSetup.padded.height),0,0,interpolation);
	
		//Create Feature Map
		cv::Mat roiGray;

		cv::cvtColor(roi, roiGray, CV_BGR2GRAY);

		Mat roiGrayFlot(roiGray.rows, roiGray.cols, CV_32FC1);
		roiGray.convertTo(roiGrayFlot,CV_32FC1);
		roiGrayFlot= roiGrayFlot.mul((float)1/255);
		roiGrayFlot= roiGrayFlot-0.5;

		int hb, wb, nChns;
		
		Mat *featureMap= create_feature_map(roi,1, nChns, roiGrayFlot, false);
		nDims= nChns;
	
		//waitKey();
		for(int i=0; i<nChns; i++)
			featureMap[i] = featureMap[i].mul(tSetup.trans_cos_win);

		//imshow("", img);
		//waitKey();
		return featureMap;
	}

	void train(bool first, cv::Mat img)
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

		//cout<<"Feature Map Fourier Translation"<<endl;
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
		int nDimsScale;
		//cout<<"Current Scale tany: "<<tSetup.current_scale_factor<<endl;
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
		//cout<<"Temp "<<temp.rows<<" "<<temp.cols<<endl;
		/*cout<<"Feature scale Fourier"<<endl;
		for(int i = 0 ; i < feature_map_scale_fourier.rows ; i++){
			for(int j = 0 ; j < feature_map_scale_fourier.cols ; j++)
				cout<<feature_map_scale_fourier.at<Vec2d>(i,j)<<"  ";
			cout<<endl;
		}*/
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
		delete[] feature_map;
		delete[] feature_map_fourier;
	}

	Point updateCentroid(Point oldC, int w , int h , int imgw, int imgh)
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

	double computePSRValue(Point maxLoc, Mat trans_response)
	{
		double psr=0;

		//Compute Mean and Standard deviation
		double mean= 0;
		double stdDev= 0;
		int count=0;
		for(int i=0; i<trans_response.cols; i++)
			for(int j=0; j<trans_response.rows; j++)
			{
				if((i>(maxLoc.x-11) && i<maxLoc.x+11) && (j>(maxLoc.y-11) && j<maxLoc.y+11))
					continue;
				mean+= trans_response.at<double>(i, j);
				count++;
			}
		mean = mean / (double)count;
		for(int i=0; i<trans_response.cols; i++)
			for(int j=0; j<trans_response.rows; j++)
			{
				if((i>(maxLoc.x-11) && i<maxLoc.x+11) && (j>(maxLoc.y-11) && j<maxLoc.y+11))
					continue;
				stdDev+= (trans_response.at<double>(i, j) - mean)*(trans_response.at<double>(i, j) - mean);
			}
		stdDev/= (double)count;
		stdDev = sqrt(stdDev);
		//Compute psr.
		/*ofstream fout("C://Users//mincosy//Desktop//Tracking//Final_DSST_C++//DSST//logzebala.txt",  std::ofstream::out | std::ofstream::app);
		psr= (trans_response.at<double>(maxLoc.x, maxLoc.y) - mean) / stdDev;
		fout<<"Mean, std, peak "<<mean<<"  "<<stdDev<<"  "<<trans_response.at<double>(maxLoc.x, maxLoc.y)<<endl;
		fout.close();*/
		return psr;
	}

	cv::Rect processFrame(cv::Mat img, bool enableScaling)
	{
		//ofstream fout("C://Users//mincosy//Desktop//Tracking//Final_DSST_C++//DSST//logzebala.txt",  std::ofstream::out | std::ofstream::app);
		//fout<<"entered process frame"<<endl;
		int nDims = 0;
		Mat *feature_map= get_translation_sample(img, tSetup, nDims);	
		Mat *feature_map_fourier= new Mat[nDims];
	
		for (int i=0; i<nDims; i++)
		{
			Mat feature_map_double(feature_map[i].rows, feature_map[i].cols, CV_64FC1);
			feature_map[i].convertTo(feature_map_double, CV_64FC1);
			feature_map_fourier[i]= createFourier(feature_map_double);
		}
	
		//response = real(ifft2(sum(hf_num .* xtf, 3) ./ (hf_den + lambda)));	
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
		//imshow("trans_response", trans_response);
		//waitKey();
		//Compute the PSR here right?thinking
		double psr= computePSRValue(maxLoc, trans_response);
		double Thresh = 0;//20;
		//ofstream fout("C://Users//mincosy//Desktop//Tracking//Final_DSST_C++//DSST//logzebala.txt",  std::ofstream::out | std::ofstream::app);
		//fout<<"PSR "<<psr<<endl;
		//fout.close();
		tSetup.centroid.x +=  cvRound((maxLoc.x-tSetup.padded.width/2+1)*tSetup.current_scale_factor);
		tSetup.centroid.y +=  cvRound((maxLoc.y-tSetup.padded.height/2+1)*tSetup.current_scale_factor);

		tSetup.centroid = updateCentroid(tSetup.centroid,tSetup.original.width*tSetup.current_scale_factor,tSetup.original.height*tSetup.current_scale_factor,img.cols,img.rows);
	
	    //fout<<"before scaling"<<endl;
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
		if(enableScaling)
		{
			Point maxLocScale = ComputeMaxDisplayfl(scale_response);

			tSetup.current_scale_factor= tSetup.current_scale_factor* tSetup.scaleFactors[maxLocScale.x];
			if(tSetup.current_scale_factor< tSetup.min_scale_factor)
				tSetup.current_scale_factor= tSetup.min_scale_factor;
			if(tSetup.current_scale_factor> tSetup.max_scale_factor)
				tSetup.current_scale_factor= tSetup.max_scale_factor;
	
		}

		
		
		//fout<<"before train"<<endl;
		//if(psr>Thresh)
		//	tParams.learning_rate= 0.15;
		//else
		//	tParams.learning_rate= 0.05;
		train(false, img);

		/*ofstream fout("C://Users//mincosy//Desktop//Tracking//Final_DSST_C++//DSST//logzebala.txt",  std::ofstream::out | std::ofstream::app);
		fout<<"Centroid "<<tSetup.centroid.x<<" "<<tSetup.centroid.y<<endl;
		fout<<"Size "<<tSetup.original.width<<" "<<tSetup.original.height<<endl;
		*/
		tSetup.centroid = updateCentroid(tSetup.centroid,tSetup.original.width*tSetup.current_scale_factor,tSetup.original.height*tSetup.current_scale_factor,img.cols,img.rows);
		int left = tSetup.centroid.x - (tSetup.original.width/2*tSetup.current_scale_factor) ;
		int top = tSetup.centroid.y - (tSetup.original.height/2*tSetup.current_scale_factor);
		cv::Rect rect(left, top, tSetup.original.width*tSetup.current_scale_factor, tSetup.original.height*tSetup.current_scale_factor);
		//fout<<"Updated Centroid "<<tSetup.centroid.x<<" "<<tSetup.centroid.y<<" "<<rect.width<<"  "<<rect.height<<endl;
		//fout.close();

		delete[] feature_map;
		delete[] feature_map_fourier;
		delete[] temp;
		delete[] tempScale;
		//fout<<"Finished deleting :D?hzhahahaha "<<tSetup.centroid.x<<" "<<tSetup.centroid.y<<endl;
		//fout.close();

		return rect;
	}

	void preprocess(int rows,int cols, cv::Mat img, int left, int top, int right, int bottom, VOTPolygon p)
	{
		//tSetup.centroid.x = left + (right-left)/2 - 1;
		//tSetup.centroid.y = top + (bottom - top)/2 - 1;
		//tSetup.original.width = right - left;
		//tSetup.original.height = bottom - top;

		tSetup.centroid.x = cvRound((p.x1+ p.x2+ p.x3+ p.x4) /4);
		tSetup.centroid.y = cvRound((p.y1+ p.y2+ p.y3+ p.y4) /4);
		double A1 = sqrt(pow(p.x1-p.x2,2) + pow(p.y1-p.y2,2)) * sqrt(pow(p.x2-p.x3,2) + pow(p.y2- p.y3,2));
		double A2 = (right - left) * (bottom - top);
		double s = sqrt(A1/A2);
		tSetup.original.width = s * (right - left) + 1;
		tSetup.original.height = s * (bottom - top) + 1;
		
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
	
		/*ofstream fout("H://logZebala.txt");
		fout<<transFilter.rows<<" "<<transFilter.cols<<endl;
		for(int i=0; i<transFilter.rows; i++)
		{
			for(int j=0; j<transFilter.cols; j++)
				fout<<transFilter.at<double>(i, j)<<" ";
			fout<<endl;
		}
		fout.close();*/
		//B- Create Scale Gaussian Filters
		//We don't know why this equation?!
		double scaleSigma= tParams.number_scales/ sqrtf(tParams.number_scales) * tParams.scale_sigma_factor;
		cv::Mat scaleFilter(1, tParams.number_scales, CV_64FC1);
		for(int r = -tParams.number_scales/2 ; r < ceil((double)tParams.number_scales/2) ; r++)
			scaleFilter.at<double>(0,r+tParams.number_scales/2)= exp(-0.5 * ((double)(r*r) / (scaleSigma*scaleSigma)));
		//	* ((double * )(scaleFilter.data + scaleFilter.step[1]+ scaleFilter.step[0]*(r+tParams.number_scales/2)))= exp(-0.5 * ((double)(r*r) / (scaleSigma*scaleSigma)));
		tSetup.scaleFourier = createFourier(scaleFilter);

		//cout<<"Preprocess"<<endl;
		//for(int i = 0 ; i < tSetup.scaleFourier.cols ; i++)
		//	cout<<tSetup.scaleFourier.at<Vec2d>(0,i)<<endl;

		//for(int i=0; i<10; i++)
		//	cout<<"Scale Filter: "<<tSetup.scaleFourier.at<double>(i, 0)<<endl;
	
		//cv::imshow("transFilter",transFilter);
		//waitKey();
		//cv::imshow("scaleFilter",scaleFilter);			
	
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

		//cout<<"train"<<endl;
		train(true, img);
	}

	GT *readGroundtruth(string path)
	{
		GT *groundtruth= new GT[1000];
		ifstream fin((path+"groundtruth.txt").c_str());
		double y;
		char x;
		int frameNumber=0;
		while(!fin.eof())
		{
			GT currentGT;
			fin>>currentGT.blx_>>x>>currentGT.bly_>>x>>currentGT.tlx_>>x>>currentGT.tly_>>x>>currentGT.trx_>>x>>currentGT.try_>>x>>currentGT.brx_>>x>>currentGT.bry_;
			groundtruth[frameNumber]= currentGT;
			frameNumber++;
		}
		return groundtruth;
	}

	CvRect convert8To4(GT groundtruth)
	{
		double x1= (groundtruth.blx_+groundtruth.tlx_) /2;
		double x2= (groundtruth.brx_+ groundtruth.trx_) /2;
		double y1= (groundtruth.tly_ + groundtruth.try_) /2;
		double y2= (groundtruth.bry_+groundtruth.bly_) /2;

		CvRect rect= cvRect(x1, y1, x2-x1, y2-y1);

		return rect;
	}

	cv::Mat visualize(Rect rect, cv::Mat img,Scalar scalar = cvScalarAll(0))
	{
		Mat retImg = img.clone();
		//CvRect rect= convert8To4(groundtruth);	
		cv::rectangle(retImg, cvPoint(rect.x, rect.y), cvPoint(rect.x+rect.width, rect.y+rect.height), scalar, 2);
		//cv::imshow("DSST Tracking", img);
		return retImg;
	}

	cv::Mat visualize(GT groundtruth, cv::Mat img)
	{
		CvRect rect= convert8To4(groundtruth);	
		cv::rectangle(img, cvPoint(rect.x, rect.y), cvPoint(rect.x+rect.width, rect.y+rect.height), cvScalarAll(0), 2);
		//cv::imshow("DSST Tracking", img);
		return img;
	}
	Mat readFromFile(string fileName)
	{
		//freopen(fileName.c_str(),"r",stdin);
		int  h = 240 , w = 320;
		int x;
		Mat frame(h, w, CV_8UC3);
		for(int j = 0 ; j < w ; j++)
			for(int i = 0  ; i < h ;i++)
			{
				cin>>x;
				frame.at<Vec3b>(i,j)[2] = x;
			}
		for(int j = 0 ; j < w ; j++)
			for(int i = 0  ; i < h ;i++)
			{
				cin>>x;
				frame.at<Vec3b>(i,j)[1] = x;
			}
		for(int j = 0 ; j < w ; j++)
			for(int i = 0  ; i < h ;i++)
			{
				cin>>x;
				frame.at<Vec3b>(i,j)[0] = x;
			}
		return frame;
	}

	void init()
	{
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


};

int main()
{
	cv::Mat initFrame;

	string working_directory="D://Scene_DS//VIVID//";
	string datasetNames[]= {"egtest01//", "egtest02//", "egtest03//","egtest04//","egtest05//","redteam//"};
	int nFrames[] = { 1820, 1300, 2570, 1832, 1763, 1917 };
	int currentDS = 2;
	string initFile = working_directory + datasetNames[currentDS] + "InitMulti.txt";
	
	std::vector<target> targets;
	ifstream fin(initFile.c_str());
	int ntargets, firstFrame;
	int x, y, w, h;
	while(!fin.eof())
	{
		fin>>firstFrame>>ntargets;
		for(int j=0; j<ntargets; j++)
		{
			fin>>x>>y>>w>>h;
			target t(x, y, w, h,firstFrame);
			targets.push_back(t);
		}
	}
	fin.close();
	
	int currentTarget= 3;
	ofstream fout("region.txt");
	fout<<targets[currentTarget].init.x<<","<<targets[currentTarget].init.y<<","<<targets[currentTarget].init.x+targets[currentTarget].init.width<<","<<targets[currentTarget].init.y<<",";
	fout<<targets[currentTarget].init.x+targets[currentTarget].init.width<<","<<targets[currentTarget].init.y+targets[currentTarget].init.height<<","<<targets[currentTarget].init.x<<","<<targets[currentTarget].init.y+targets[currentTarget].init.height<<endl;
    fout.close();

	//load region, images and prepare for output
	DSSTTracker dsst;	
	dsst.generateFiles(working_directory + datasetNames[currentDS] + "frame", targets[currentTarget].firstFrame, nFrames[currentDS], 5);
    
	VOT vot_io("region.txt", "images.txt", "output.txt");   
    VOTPolygon p = vot_io.getInitPolygon();
	
   
    int top = cvRound(MIN(p.y1, MIN(p.y2, MIN(p.y3, p.y4))));
    int left = cvRound(MIN(p.x1, MIN(p.x2, MIN(p.x3, p.x4))));
    int bottom = cvRound(MAX(p.y1, MAX(p.y2, MAX(p.y3, p.y4))));
    int right = cvRound(MAX(p.x1, MAX(p.x2, MAX(p.x3, p.x4))));
    
	vot_io.getNextImage(initFrame);//, true);
	
	
	dsst.preprocess(initFrame.rows,initFrame.cols, initFrame, left, top, right, bottom, p);
	vot_io.outputPolygon(p);

	int frameNumber=1; 
    while (true)
	{
		cout << frameNumber << endl;
		Mat currentFrame;
		int nextFrame = vot_io.getNextImage(currentFrame);//, false);
		
		if(nextFrame!=1 || !currentFrame.data)
			break;
		
		cv::Rect rect= dsst.processFrame(currentFrame, false);
		
		cv::Mat currentFrame2= currentFrame.clone();
		currentFrame2= dsst.visualize(rect, currentFrame2);
		putText(currentFrame2, dsst.intToStr(frameNumber,"",0,""), Point(10,10),1,1,Scalar(0,255,0),3);
		imshow("TestingTeet", currentFrame2);
		frameNumber++;
		/*if (frameNumber >= 1101)
			waitKey(1000);
		else*/
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
		//currentFrame.release();
    }	
	return 0;
}