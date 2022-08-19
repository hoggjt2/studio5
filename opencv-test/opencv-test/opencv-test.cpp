#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define PI 3.14159265

class refPoint {
public:
    double x;
    double y;
};

float slope(float x1, float y1, float x2, float y2)
{
    if (x2 - x1 != 0)
        return (y2 - y1) / (x2 - x1);
    return INT_MAX;
}

void findAngle(double dotProd, double V1, double V2) {

    double angle = (dotProd / (V1 * V2));

    double ret = acos(angle);

    double val = (ret * 180) / PI;

    cout << "Angle: " << val;

}

int main()
{
    //Create an object for the knee reference point
    refPoint knee;
    knee.x = 150;
    knee.y = 50;

    //Create an object for the upper leg reference point 
    refPoint upperLeg;
    upperLeg.x = 350;
    upperLeg.y = 50;

    //Create an object for the calf reference point
    refPoint calf;
    calf.x = 80;
    calf.y = 170;

    array<double, 2> vec1{ (upperLeg.x - knee.x), (upperLeg.y - knee.y) };
    array<double, 2> vec2{ (calf.x - knee.x), (calf.y - knee.y) };

	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(knee.x, knee.y), 10, Scalar(255, 0, 0), -100);
	circle(image, Point(upperLeg.x, upperLeg.y), 10, Scalar(255, 0, 0), -100);
	circle(image, Point(calf.x, calf.y), 10, Scalar(255, 0, 0), -100);
	line(image, Point(knee.x, knee.y), Point(upperLeg.x, upperLeg.y), Scalar(0, 128, 0), 2);
	line(image, Point(knee.x, knee.y), Point(calf.x, calf.y), Scalar(0, 128, 0), 2);
	imshow("Display Window", image);
    double V1, V2;
    double dotProd =((vec1[0] * vec2[0]) + (vec1[1] * vec2[1]));
    V1 = sqrt(pow(vec1[0], 2) + pow(vec1[1], 2));
    V2 = sqrt(pow(vec2[0], 2) + pow(vec2[1], 2));
    findAngle(dotProd, V1, V2);
	waitKey(0);
	return 0;
}
