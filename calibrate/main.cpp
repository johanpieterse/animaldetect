// Main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/features2d.hpp>

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace cv;
using namespace std;

Mat image1;
Mat image2;
Mat image2_filtered;
Mat image1_filtered;
Mat diffImage;
Mat diffImageMask;
Mat final;


int filter_kernel = 4;
int max_filter_kernel = 25;
void filter_kernel_callback(int, void*);

int diff_threshold = 50;
int max_diff_threshold = 255;
void diff_threshold_callback(int, void*);


int area_threshold = 25;
int max_area_threshold = 1000;
void area_threshold_callback(int, void*);


std::map<std::string, int> options; // global?

void readConfig()
{

	std::ifstream cfgfile("config.txt");
	std::string line;
	if (cfgfile.is_open())
	{
		while (getline(cfgfile, line))

			//for (std::string line; std::getline(cfgfile, line); )
		{
			std::istringstream iss(line);
			std::string id, eq, val;

			bool error = false;

			if (!(iss >> id))
			{
				error = true;
			}
			else if (id[0] == '#')
			{
				continue;
			}
			else if (!(iss >> eq >> val >> std::ws) || eq != "=" || iss.get() != EOF)
			{
				error = true;
			}

			if (error)
			{
				options["filter_kernel"] = filter_kernel;
				options["diff_threshold"] = diff_threshold;
				options["area_threshold"] = area_threshold;
				break;
			}
			else
			{
				options[id] = atoi(val.c_str());
			}
		}

		cfgfile.close();

		std::map<std::string, int>::iterator it;
		it = options.find("filter_kernel");
		if (it != options.end())
		{
			filter_kernel = options["filter_kernel"];
		}

		it = options.find("diff_threshold");
		if (it != options.end())
		{
			diff_threshold = options["diff_threshold"];
		}

		it = options.find("area_threshold");
		if (it != options.end())
		{
			area_threshold = options["area_threshold"];
		}
	}
}

void writeConfig()
{
	std::ofstream  cfgfile("config.txt");
	if (cfgfile.is_open())
	{
		options["filter_kernel"] = filter_kernel;
		options["diff_threshold"] = diff_threshold;
		options["area_threshold"] = area_threshold;

		for (std::map<std::string, int>::iterator it = options.begin(); it != options.end(); ++it)
		{
			cfgfile << it->first << " = " << it->second << "\n";
		}

		cfgfile.close();
	}
}

void getFiltered()
{
	Mat dst, kernel;
	int kernel_size = (filter_kernel * 2) + 1;
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size *kernel_size);
	cv::filter2D(image1, image1_filtered, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	cv::filter2D(image2, image2_filtered, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
}


void getDiff(const Mat backgroundImage, const Mat currentImage, int threshold)
{
	
	cv::absdiff(backgroundImage, currentImage, diffImage);

	diffImageMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);

	float f_threshold = (float)threshold;
	float dist;

	for (int j = 0; j<diffImage.rows; ++j)
		for (int i = 0; i<diffImage.cols; ++i)
		{
			cv::Vec3b pix = diffImage.at<cv::Vec3b>(j, i);

			dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
			dist = sqrt(dist);

			if (dist>f_threshold)
			{
				diffImageMask.at<unsigned char>(j, i) = 255;
			}
		}
}


void getContours()
{
	//Contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	/// Find contours
	findContours(diffImageMask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	image2.copyTo(final);

	for (int i = 0; i < contours.size(); i++)
	{
		if (boundRect[i].area() > area_threshold * 10) //10000)
		{
			rectangle(final, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 255), 2, 8, 0);
		}
	}
}

int main(int argc, char** argv)
{
	if (argc != 3 )
	{
		cout << " Usage: ad_calibrate Image1 Image2" << endl;
		return -1;
	}


	image1 = imread(argv[1], IMREAD_COLOR); // Read the file
	if (!image1.data) // Check for invalid input
	{
		cout << "Could not open or find the image1" << std::endl;
		return -1;
	}

	image2 = imread(argv[2], IMREAD_COLOR); // Read the file
	if (!image2.data) // Check for invalid input
	{
		cout << "Could not open or find the image2" << std::endl;
		return -1;
	}

	/*if (image1.size().width > 2000)
	{
		cv::resize(image1, image1, cvSize(0, 0), 0.5, 0.5);
		cv::resize(image2, image2, image1.size(), 0, 0);
	}*/

	readConfig();
	
	getFiltered();
	cv::namedWindow("filtered", WINDOW_NORMAL); // Create a window for display.
	cv::imshow("filtered", image2_filtered);
	createTrackbar("Kernel:", "filtered", &filter_kernel, max_filter_kernel, filter_kernel_callback);

	getDiff(image1_filtered, image2_filtered, diff_threshold);
	cv::namedWindow("diffImageMask", WINDOW_NORMAL); // Create a window for display.
	cv::imshow("diffImageMask", diffImageMask);
	createTrackbar("Threshold:", "diffImageMask", &diff_threshold, max_diff_threshold, diff_threshold_callback);

	getContours();
	cv::namedWindow("Final", WINDOW_NORMAL);
	cv::imshow("Final", final);
	createTrackbar("Area:", "Final", &area_threshold, max_area_threshold, area_threshold_callback);

	waitKey(0); // Wait for a keystroke in the window

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);

	imwrite("test.jpg", final, compression_params);
	writeConfig();

	return 0;
}


void filter_kernel_callback(int, void*)
{
	getFiltered();
	
	cv::namedWindow("filtered", WINDOW_NORMAL); // Create a window for display.
	cv::imshow("filtered", image2_filtered);
	
	diff_threshold_callback(0, 0);
}

void diff_threshold_callback(int, void*)
{
	getDiff(image1_filtered, image2_filtered, diff_threshold);

	cv::namedWindow("diffImageMask", WINDOW_NORMAL); // Create a window for display.
	cv::imshow("diffImageMask", diffImageMask);

	area_threshold_callback(0, 0);
}

void area_threshold_callback(int, void*)
{
	getContours();

	cv::namedWindow("Final", WINDOW_NORMAL);
	cv::imshow("Final", final);
}
