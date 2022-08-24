//
//  this sample demonstrates the use of pretrained openpose networks with opencv's dnn module.
//
//  it can be used for body pose detection, using either the COCO model(18 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/3.4/testdata/dnn/openpose_pose_coco.prototxt
//
//  or the MPI model(16 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/3.4/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt
//
//  (to simplify this sample, the body models are restricted to a single person.)
//
//
//  you can also try the hand pose model:
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
//  https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt
//
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include<thread>
using namespace cv;
using namespace cv::dnn;
#include <iostream>
using namespace std;
// connection table, in the format [model_id][pair_id][from/to]
// please look at the nice explanation at the bottom of:
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
//
const int POSE_PAIRS[3][20][2] = {
{   // COCO body
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
},
{   // MPI body
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
},
{   // hand
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // pinkie
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
} };

void call_thread(Net n, Mat * result) {
    *result = n.forward();
    cout << "in the thread";
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
        "{ h help           | false     | print this help message }"
        "{ p proto          |           | (required) model configuration, e.g. hand/pose.prototxt }"
        "{ m model          |           | (required) model weights, e.g. hand/pose_iter_102000.caffemodel }"
        "{ i image          |           | (required) path to image file (containing a single person, or hand) }"
        "{ d dataset        |           | specify what kind of model was trained. It could be (COCO, MPI, HAND) depends on dataset. }"
        "{ width            |  368      | Preprocess input image by resizing to a specific width. }"
        "{ height           |  368      | Preprocess input image by resizing to a specific height. }"
        "{ t threshold      |  0.1      | threshold or confidence value for the heatmap }"
        "{ s scale          |  0.003922 | scale for blob }"
    );
    VideoCapture cap(0); //capture the video from web cam
    String modelTxt = "C:/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";//samples::findFile(parser.get<string>("proto"));
    String modelBin = "C:/openpose/models/pose/mpi/pose_iter_160000.caffemodel";//samples::findFile(parser.get<string>("model"));
    String imageFile = "C:/openpose/examples/media/COCO_val2014_000000000294.jpg";//samples::findFile(parser.get<String>("image"));
    String dataset = "MPI";//parser.get<String>("dataset");
    //String modelTxt = "C:/openpose/models/pose/coco/pose_deploy_linevec.prototxt";//samples::findFile(parser.get<string>("proto"));
    //String modelBin = "C:/openpose/models/pose/coco/pose_iter_440000.caffemodel";//samples::findFile(parser.get<string>("model"));
    //String imageFile = "C:/openpose/examples/media/COCO_val2014_000000000294.jpg";//samples::findFile(parser.get<String>("image"));
    //String dataset = "COCO";//parser.get<String>("dataset");
    int W_in = 200;//parser.get<int>("width");
    int H_in = 200;//parser.get<int>("height");
    float thresh = 0.2;// parser.get<float>("threshold");
    float scale = 0.003922;// parser.get<float>("scale");
    if (parser.get<bool>("help") || modelTxt.empty() || modelBin.empty() || imageFile.empty())
    {
        cout << "A sample app to demonstrate human or hand pose detection with a pretrained OpenPose dnn." << endl;
        parser.printMessage();
        return 0;
    }
    int midx, npairs, nparts;
    if (!dataset.compare("COCO")) { midx = 0; npairs = 17; nparts = 18; }
    else if (!dataset.compare("MPI")) { midx = 1; npairs = 14; nparts = 16; }
    else if (!dataset.compare("HAND")) { midx = 2; npairs = 20; nparts = 22; }
    else
    {
        std::cerr << "Can't interpret dataset parameter: " << dataset << std::endl;
        exit(-1);
    }
    // read the network model
    Net net = readNet(modelBin, modelTxt);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    // and the image
    Mat img, result, result_tmp;
    int count = 0;
    bool complete_thread = false;
    do {
        //Mat img = imread(imageFile);
        cap >> img;
        if (img.empty())
        {
            std::cerr << "Can't read image from the file: " << imageFile << std::endl;
            exit(-1);
        }
        // send it through the network
        Mat inputBlob = blobFromImage(img, scale, Size(W_in, H_in), Scalar(0, 0, 0), false, false);
        net.setInput(inputBlob);
        //if (count == 0) {
          //  count = 15;
            //thread thread_object(call_thread, net, &result);
            result = net.forward();
        //}

        //if (count == 15) {
            //count = 10;
          //  cout << "called thread...";
            //Mat* result_ptr = &result_tmp;
            //thread thread_object(call_thread, net, result_ptr);
            //thread_object.join();
            /*if (complete_thread) {
                thread_object.join();
                result = result_tmp.clone();
                complete_thread = false;
            }*/
            //thread_object.detach();
        //}
        //if (count ==   1) {
            
          //  complete_thread = true;
            //count = 16;
        //}

        //count--;
        // the result is an array of "heatmaps", the probability of a body part being in location x,y
        int H = result.size[2];
        int W = result.size[3];
        // find the position of the body parts
        vector<Point> points(22);
        for (int n = 0; n < nparts; n++)
        {
            // Slice heatmap of corresponding body's part.
            Mat heatMap(H, W, CV_32F, result.ptr(0, n));
            // 1 maximum per heatmap
            Point p(-1, -1), pm;
            double conf;
            minMaxLoc(heatMap, 0, &conf, 0, &pm);
            if (n >= 8 && n<=13) {
                if (conf > thresh)
                    p = pm;
            }
            points[n] = p;
        }
        // connect body parts and draw it !
        float SX = float(img.cols) / W;
        float SY = float(img.rows) / H;
        for (int n = 0; n < npairs; n++)
        {
            // lookup 2 connected body/hand parts
            Point2f a = points[POSE_PAIRS[midx][n][0]];
            Point2f b = points[POSE_PAIRS[midx][n][1]];
            // we did not find enough confidence before
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;
            // scale to image size
            a.x *= SX; a.y *= SY;
            b.x *= SX; b.y *= SY;
            line(img, a, b, Scalar(0, 200, 0), 2);
            circle(img, a, 3, Scalar(0, 0, 200), -1);
            circle(img, b, 3, Scalar(0, 0, 200), -1);
        }
        imshow("OpenPose", img);
        waitKey(30);
    } while (1);
    return 0;
}

