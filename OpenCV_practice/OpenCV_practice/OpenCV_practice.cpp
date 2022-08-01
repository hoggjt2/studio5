
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
/*
int main()
{
	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Display Window", image);
	waitKey(0);
	return 0;
}
*/
/*
int main() {

	Mat image;

	namedWindow("cam 0");
	namedWindow("cam 1");

	VideoCapture cap(0);
	VideoCapture cap1(1);

	if (!cap.isOpened()) {

		cout << "cannot open camera";

	}
	if (!cap1.isOpened()) {

		cout << "cannot open camera";

	}

	while (true) {

		cap >> image;

		imshow("cam 0", image);

		cap1 >> image;

		imshow("cam 1", image);

		waitKey(25);

	}

	return 0;

}
*/
int main(int argc, char** argv) {
	VideoCapture video_load(0);//capturing video from default camera//
	VideoCapture video_load1(1);//capturing video from secondary camera//
	namedWindow("Adjust");//declaring window to show the image//
	int Hue_Lower_Value = 0;//initial hue value(lower)//
	int Hue_Lower_Upper_Value = 22;//initial hue value(upper)//
	int Saturation_Lower_Value = 0;//initial saturation(lower)//
	int Saturation_Upper_Value = 255;//initial saturation(upper)//
	int Value_Lower = 0;//initial value (lower)//
	int Value_Upper = 255;//initial saturation(upper)//
	createTrackbar("Hue_Lower", "Adjust", &Hue_Lower_Value, 179);//track-bar for lower hue//
	createTrackbar("Hue_Upper", "Adjust", &Hue_Lower_Upper_Value, 179);//track-bar for lower-upper hue//
	createTrackbar("Sat_Lower", "Adjust", &Saturation_Lower_Value, 255);//track-bar for lower saturation//
	createTrackbar("Sat_Upper", "Adjust", &Saturation_Upper_Value, 255);//track-bar for higher saturation//
	createTrackbar("Val_Lower", "Adjust", &Value_Lower, 255);//track-bar for lower value//
	createTrackbar("Val_Upper", "Adjust", &Value_Upper, 255);//track-bar for upper value//
	while (1) {
		Mat actual_Image;//matrix to load actual image//
		bool temp = video_load.read(actual_Image);//loading actual image to matrix from video stream//
		Mat convert_to_HSV;//declaring a matrix to store converted image//
		cvtColor(actual_Image, convert_to_HSV, COLOR_BGR2HSV);//converting BGR image to HSV and storing it in convert_to_HSV matrix//
		Mat detection_screen;//declaring matrix for window where object will be detected//
		inRange(convert_to_HSV, Scalar(Hue_Lower_Value, Saturation_Lower_Value, Value_Lower), Scalar(Hue_Lower_Upper_Value, Saturation_Upper_Value, Value_Upper), detection_screen);//applying track-bar modified value of track-bar//
		erode(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological opening for removing small objects from foreground//
		dilate(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological opening for removing small object from foreground//
		dilate(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological closing for filling up small holes in foreground//
		erode(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological closing for filling up small holes in foreground//
		imshow("Colour Detection 0", detection_screen);//showing detected object//
		imshow("Cam 0", actual_Image);//showing actual image//
		/*
		//Secondary Camera
		temp = video_load1.read(actual_Image);//loading actual image to matrix from video stream//
		//Mat convert_to_HSV;//declaring a matrix to store converted image//
		cvtColor(actual_Image, convert_to_HSV, COLOR_BGR2HSV);//converting BGR image to HSV and storing it in convert_to_HSV matrix//
		//Mat detection_screen;//declaring matrix for window where object will be detected//
		inRange(convert_to_HSV, Scalar(Hue_Lower_Value, Saturation_Lower_Value, Value_Lower), Scalar(Hue_Lower_Upper_Value, Saturation_Upper_Value, Value_Upper), detection_screen);//applying track-bar modified value of track-bar//
		erode(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological opening for removing small objects from foreground//
		dilate(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological opening for removing small object from foreground//
		dilate(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological closing for filling up small holes in foreground//
		erode(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//morphological closing for filling up small holes in foreground//
		
			// Top Left Corner
		Point p1(30, 30);

		// Bottom Right Corner
		Point p2(255, 255);

		int thickness = 2;

		// Drawing the Rectangle
		rectangle(actual_Image, p1, p2,
			Scalar(255, 0, 0),
			thickness, LINE_8);
		
		imshow("Colour Detection 1", detection_screen);//showing detected object//
		imshow("Cam 1", actual_Image);//showing actual image//
		*/
		waitKey(25);
	}
	return 0;
}