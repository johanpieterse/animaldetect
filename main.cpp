// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


//int main()
//{
//   
//	char * test = "123";
//	printf(test);
//
//	return 0;
//
//}
//


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>


using namespace cv;
using namespace std;

cv::Mat result; // Result correlation will be placed here
cv::Mat imageResult;
cv::Mat diffImage;
cv::Mat foregroundMask;

// Compare two images by getting the L2 error (square-root of sum of squared error).
double getSimilarity(const Mat A, const Mat B) {
	if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
		// Calculate the L2 relative error between images.
		double errorL2 = norm(A, B, CV_L2);
		// Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
		double similarity = errorL2 / (double)(A.rows * A.cols);
		return similarity;
	}
	else {
		//Images have a different size
		return 100000000.0;  // Return a bad value
	}
}

double getMatchTemplate(const Mat A, const Mat B, const Mat mask) {
	cv::Mat image;  // Your input image
	cv::Mat templ;  // Your template image of the screw 
	

	image = A;
	templ = B;
					// Do template matching across whole image
	cv::matchTemplate(image, templ, result, CV_TM_CCORR_NORMED); // , mask);

	// Find a best match:
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;

	//Mat mask2 = cv::zeros(mask.size(), CV_8U);

	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	// Regards to documentation the best match is in maxima location
	// (http://opencv.willowgarage.com/documentation/cpp/object_detection.html)

	// Move center of detected screw to the correct position:  
	cv::Point screwCenter = maxLoc + cv::Point(templ.cols / 2, templ.rows / 2);

	image.copyTo(imageResult);


	//cv::rectangle(result, Rect rec, const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0)
	//Scalar color(rand() & 255, rand() & 255, rand() & 255);
	Scalar color(255, 0, 0);
	cv::rectangle(imageResult, Rect(maxLoc.x, maxLoc.y, templ.cols, templ.rows), color, 5, LINE_8, 0);

	return maxVal;
}


void getDiff(const Mat backgroundImage, const Mat currentImage)
{
	
	cv::absdiff(backgroundImage, currentImage, diffImage);

	foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);

	float threshold = 50.0f;
	float dist;

	for (int j = 0; j<diffImage.rows; ++j)
		for (int i = 0; i<diffImage.cols; ++i)
		{
			cv::Vec3b pix = diffImage.at<cv::Vec3b>(j, i);

			dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
			dist = sqrt(dist);

			if (dist>threshold)
			{
				foregroundMask.at<unsigned char>(j, i) = 255;
			}
		}
}

int main(int argc, char** argv)
{
	if (argc != 4 )
	{
		cout << " Usage: display_image ImageToLoadAndDisplay1 ImageToLoadAndDisplay2" << endl;
		return -1;
	}

	Mat image1;
	image1 = imread(argv[1], IMREAD_COLOR); // Read the file

	Mat image1_filtered;
	//cv::bilateralFilter(image1, image1_filtered, 9, 75, 75);
	//cv::GaussianBlur(image1, image1_filtered, cv::Size(25, 25),25);
	Mat dst, kernel;
	int kernel_size = 9;
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size *kernel_size);
	cv::filter2D(image1, image1_filtered, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);

	Mat image2;
	image2 = imread(argv[2], IMREAD_COLOR); // Read the file
	
	Mat image2_filtered;
	//cv::bilateralFilter(image2, image2_filtered, 9, 75, 75);
	//cv::GaussianBlur(image2, image2_filtered, cv::Size(25, 25), 25);
	cv::filter2D(image2, image2_filtered, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);

	Mat mask;
	mask = imread(argv[3], IMREAD_COLOR); // Read the sfile
	
	int frame = 0;
	//Mat croppedImage2 = image2(Rect(0 + frame, 0 + frame, image2.cols - (frame*2), image2.rows - (frame*2)));

	

	if (!image1.data) // Check for invalid input
	{
		cout << "Could not open or find the image1" << std::endl;
		return -1;
	}

	if (!image2.data) // Check for invalid input
	{
		cout << "Could not open or find the image2" << std::endl;
		return -1;
	}

	//namedWindow("Image 1", WINDOW_NORMAL); // Create a window for display.
	//imshow("Image 1", image1); // Show our image inside it.

	//namedWindow("Image 2", WINDOW_NORMAL); // Create a window for display.
	//imshow("Image 2", croppedImage2); // Show our image inside it.

	//double similarity = getSimilarity(image1, image2);
	//cout << "Similarity:" << similarity << std::endl;


	//double matchtemplate = getMatchTemplate(image1, croppedImage2, mask);
	//cout << "Matchtemplate:" << matchtemplate << std::endl;

	getDiff(image1_filtered, image2_filtered);
	//getDiff(image1, image2);

	//namedWindow("Result", WINDOW_NORMAL); // Create a window for display.
	//imshow("Result", result);

	//namedWindow("Image Result", WINDOW_NORMAL); // Create a window for display.
	//imshow("Image Result", imageResult); // Show our image inside it.

	cv::namedWindow("filtered", WINDOW_NORMAL); // Create a window for display.
	cv::imshow("filtered", image2_filtered);

	cv::namedWindow("diffImage", WINDOW_NORMAL); // Create a window for display.
	cv::imshow("diffImage", diffImage);

	//Dilation
	int dilation_type = MORPH_RECT;
	int dilation_size = 25;

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	Mat dilation_dst;
	dilate(foregroundMask, dilation_dst, element);

	cv::namedWindow("dilation_dst", WINDOW_NORMAL); // Create a window for display.
	cv::imshow("dilation_dst", dilation_dst);


#if true
	// Set up the detector with default parameters.
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 200;
	params.maxThreshold = 255;

	params.filterByColor = true;
	params.blobColor = 255;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 10000;
	params.maxArea = 10000000;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = false;
	params.minInertiaRatio = 0.01;


	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	
	std::vector<KeyPoint> keypoints;
	// Detect blobs.
	detector->detect(dilation_dst, keypoints);
	

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat im_with_keypoints;
	cv::drawKeypoints(foregroundMask, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	cv::namedWindow("im_with_keypoints", WINDOW_NORMAL);
	cv::imshow("im_with_keypoints", im_with_keypoints);

	//std::vector<Point2f> thepoints;
	////convert(const std::vector<KeyPoint>& keypoints, CV_OUT std::vector<Point2f>& points2f);
	//KeyPoint::convert(keypoints, thepoints);
	//Rect boundRect;
	//boundRect = boundingRect(thepoints);
	//rectangle(keypoints, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 2, 8, 0);

	for (int i = 0; i < keypoints.size(); i++)
	{
		Rect bRect = Rect(keypoints[i].pt.x - (keypoints[i].size / 2.0), keypoints[i].pt.y - (keypoints[i].size / 2.0), keypoints[i].size, keypoints[i].size);
		rectangle(image2, bRect.tl(), bRect.br(), Scalar(0, 255, 0), 2, 8, 0);
	}

	// Show finale image
	cv::namedWindow("image2", WINDOW_NORMAL);
	cv::imshow("image2", image2);

#endif

	
#if true
	//Contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	/// Find contours
	findContours(foregroundMask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	/// Draw polygonal contour + bonding rects + circles
	RNG rng(12345);
	Mat drawing = Mat::zeros(foregroundMask.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{

		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours_poly, i, Scalar(0, 0, 255), 1, 8, vector<Vec4i>(), 0, Point());
		//drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		if (boundRect[i].area() > 2500 ) //10000)
		{
			rectangle(image2, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2, 8, 0);
		}
	}

	cv::namedWindow("Contours", WINDOW_NORMAL);
	cv::imshow("Contours", drawing);

	cv::namedWindow("Final", WINDOW_NORMAL);
	cv::imshow("Final", image2);
#endif

	//namedWindow("foregroundMask", WINDOW_NORMAL); // Create a window for display.
	//imshow("foregroundMask", foregroundMask);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

