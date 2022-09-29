// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
//for landmarks
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
//#include "opencv2/viz.hpp"
//#include <bits/stdc++.h>

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kDetectionsStream[] = "pose_landmarks";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
    "Full path of video to load. "
    "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
    "Full path of where to save result (.mp4 only). "
    "If not provided, show result in a window.");

/*
struct MovementFactor {
    int arr[5];
};
std::list<MovementFactor> left_arm_factors;
int load_factors() {
    int f[5] = { 12,14,16, 0, 30 };
    MovementFactor x;
    for (int i = 0; i < 5; i++) {
        x.arr[i] = f[i];
    }
    left_arm_factors.push_back(x);
    for (MovementFactor mf : left_arm_factors) {
        std::cout << mf.arr[0];
    }
    return 0;
}
*/

// Function to find the magnitude
// of the given vector
double magnitude(double arr[], int N)
{
    // Stores the final magnitude
    double magnitude = 0;
 
    // Traverse the array
    for (int i = 0; i < N; i++)
        magnitude += arr[i] * arr[i];
 
    // Return square root of magnitude
    return sqrt(magnitude);
}
 
// Function to find the dot
// product of two vectors
double dotProduct(double arr[],
                  double brr[], int N)
{
    // Stores dot product
    double product = 0;
 
    // Traverse the array
    for (int i = 0; i < N; i++)
        product = product + arr[i] * brr[i];
 
    // Return the product
    return product;
}
 
double angleBetweenVectors(double arr[],
                         double brr[], int N)
{
    // Stores dot product of two vectors
    double dotProductOfVectors
        = dotProduct(arr, brr, N);
 
    // Stores magnitude of vector A
    double magnitudeOfA
        = magnitude(arr, N);
 
    // Stores magnitude of vector B
    double magnitudeOfB
        = magnitude(brr, N);
 
    // Stores angle between given vectors
    double angle = dotProductOfVectors
                   / (magnitudeOfA * magnitudeOfB);
 
    // Print the angle
    return angle;
}

double get_angle(double a[], double b[], double c[]) {
    double ba[] = {b[0]-a[0], b[1]-a[1]};
    double bc[] = {b[0]-c[0], b[1]-c[1]};
    return angleBetweenVectors(ba, bc, 2);
    //return 1;
}

int draw_bad_form() {
    std::cout << "bad form";
    return 0;
}

int evaluate_points(float landmark_array[33][3], int W, int H) {
/*
    for (MovementFactor mf : left_arm_factors) {
        int H = 100;
        int W = 100;
        //i will be used to loop through list of movement factors
        int point_1[3] = { landmark_array[mf.arr[0]][0], landmark_array[mf.arr[0]][1] , landmark_array[mf.arr[0]][2] };
        int point_2[3] = { landmark_array[mf.arr[1]][0], landmark_array[mf.arr[1]][1] , landmark_array[mf.arr[1]][2] };
        int point_3[3] = { landmark_array[mf.arr[2]][0], landmark_array[mf.arr[2]][1] , landmark_array[mf.arr[2]][2] };
        int ang = get_angle(point_1, point_2, point_3);
        if (ang < mf.arr[3]) {//lower limit
            draw_bad_form();
            return 0;
        }
        if (ang > mf.arr[4]) {//upper limit
            draw_bad_form();
            return 0;
        }
    }
    */
    double A[] = {W, H};
    double B[] = {W, H};
    double C[] = {W, H};
    double D[] = {W, H};

    A[0] *= landmark_array[12][0];
    A[1] *= landmark_array[12][1];
    B[0] *= landmark_array[24][0];
    B[1] *= landmark_array[24][1];
    C[0] *= landmark_array[26][0];
    C[1] *= landmark_array[26][1];
    D[0] *= landmark_array[28][0];
    D[1] *= landmark_array[28][1];

    double hip_angle = get_angle(A, B, C);
    double knee_angle = get_angle(B, C, D);

    //if(hip_angle >= knee_angle-10 && hip_angle <= knee_angle+10){
        //fine form
    //}else{
      //  draw_bad_form();
    //}
    //system("cls");
    if(false){
        std::cout << "Can not see all points" << "\n";
        std::cout << A[0] << "\n";
    }else{
        if(A[0] >= D[0]-20 && A[0] <= D[0]+20){
            std::cout << "GOOD FORM" << "\n";
            //std::cout << A[0] << "\n";
        }else{
            std::cout << "BAD FORM" << "\n";
        }
    }
    //std::cout << landmark_array[12][0] <<"\n";
    //std::cout << A[0] <<"\n";
    return 0;
}




absl::Status RunMPPGraph() {

    //load_factors();


    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: "
        << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Initialize the camera or load the video.";
    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video) {
        capture.open(absl::GetFlag(FLAGS_input_video_path));
    }
    else {
        capture.open(0);
    }
    RET_CHECK(capture.isOpened());

    cv::VideoWriter writer;
    const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
    if (!save_video) {
        cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);

#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
        //capture.set(cv::CAP_PROP_FRAME_WIDTH, 1080);
        //capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1920);
        //capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        //capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(cv::CAP_PROP_FPS, 30);
#endif
    }

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
        graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_detection, graph.AddOutputStreamPoller(kDetectionsStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));


    LOG(INFO) << "Start grabbing and processing frames.";
    bool grab_frames = true;
    int count = 0;
    while (grab_frames) {

        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        //camera_frame_raw = cv::imread("C:/openpose/examples/media/COCO_val2014_000000000564.jpg");
        //capture >> camera_frame_raw;
        capture.read(camera_frame_raw);
        if (camera_frame_raw.empty()) {
            if (!load_video) {
                LOG(INFO) << "Ignore empty frames from camera.";
                continue;
            }
            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        if (!load_video) {
            cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
        }

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release())
            .At(mediapipe::Timestamp(frame_timestamp_us))));

        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet packet;
        if (!poller.Next(&packet)) break;
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();

        mediapipe::Packet detection_packet;
        if (!poller_detection.Next(&detection_packet)) break;
        auto output_landmarks = detection_packet.Get<mediapipe::NormalizedLandmarkList>();
        //landmark log out
        int x = 0;
        float landmark_array[33][3];
        for (mediapipe::NormalizedLandmark lm_point : output_landmarks.landmark())
        {
            //landmark_array = { x, lm_point.x(), lm_point.y(), lm_point.z() };
            //std::cout << landmark_array << "\n";
            //std::cout << "id: " << x << "\n";
            //std::cout << "x: " << lm_point.x() << "\n";
            //std::cout << "y: " << lm_point.y() << "\n";
            //std::cout << "z: " << lm_point.z() << "\n";
            landmark_array[x][0] = lm_point.x();
            landmark_array[x][1] = lm_point.y();
            landmark_array[x][2] = lm_point.z();
            x++;
        }
        /*if (count = 1000000000) {
            for (int i = 0; i < 33; i++) {
                std::cout << i << " " << landmark_array[i][0] << " " << landmark_array[i][1] << " " << landmark_array[i][2] << "\n";
            }*/
            //}
            //std::cout << x;
            //// Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        if (save_video) {
            if (!writer.isOpened()) {
                LOG(INFO) << "Prepare video writer.";
                writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
                RET_CHECK(writer.isOpened());
            }
            writer.write(output_frame_mat);
        }
        else {
            cv::imshow(kWindowName, output_frame_mat);
            // Press any key to exit.
            const int pressed_key = cv::waitKey(30);
            if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
        }

        //Check for bad form
        //Get Mat size size
        int W = camera_frame.cols;
        int H = camera_frame.rows;
        evaluate_points(landmark_array, W, H);

        /*if (count = 1000000000) {
            grab_frames = false;
        }
        count++;*/
    }
        LOG(INFO) << "Shutting down.";
        if (writer.isOpened()) writer.release();
        MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
        return graph.WaitUntilDone();
    
}



int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
