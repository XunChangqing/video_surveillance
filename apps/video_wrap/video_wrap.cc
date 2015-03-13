#include "utils.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

const int kMinLengthSecs = 3;
const int kMaxDistSecs = 5;
const float kMinAreaRatio = 800.0 / (704.0 * 576.0);
const float kFgRatio = 0.9;

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }
  std::string frame_filename = std::string(argv[1]);

  cv::VideoCapture tmp_frame_cap(frame_filename);
  float frame_width = tmp_frame_cap.get(CV_CAP_PROP_FRAME_WIDTH);
  float frame_height = tmp_frame_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  float fps = tmp_frame_cap.get(CV_CAP_PROP_FPS);

  std::string clip_prefix =
      frame_filename.substr(0, frame_filename.rfind('.')) + '/';
  std::string max_area_file_name = clip_prefix + "active_frame.txt";
  std::vector<int> start_frames, end_frames;
  if (!masa_video_surveillance::GetActiveClip(
          max_area_file_name, kMinAreaRatio * frame_width * frame_height,
          kMinLengthSecs * fps, kMaxDistSecs * fps, start_frames, end_frames)) {
    printf("Failed to read active frame file %s\n", max_area_file_name.c_str());
    return 1;
  }

  if (start_frames.size() <= 0) {
    printf("There is no active frames!\n");
    return 1;
  }

  std::string fg_filename = clip_prefix + "fg.avi";

  std::vector<cv::VideoCapture *> frame_cap_arr;
  std::vector<cv::VideoCapture *> fg_cap_arr;
  for (int i = 0; i < start_frames.size(); i++) {
    cv::VideoCapture *frame_cap = new cv::VideoCapture(frame_filename);
    cv::VideoCapture *fg_cap = new cv::VideoCapture(fg_filename);
    if (!frame_cap->isOpened() || !fg_cap->isOpened())
      return 1;
    frame_cap->set(CV_CAP_PROP_POS_FRAMES, start_frames[i] - fps);
    fg_cap->set(CV_CAP_PROP_POS_FRAMES, start_frames[i] - fps);
    frame_cap_arr.push_back(frame_cap);
    fg_cap_arr.push_back(fg_cap);
  }
  // printf("before loop\n");

  cv::Mat oframe(static_cast<int>(frame_height), static_cast<int>(frame_width),
                 CV_8UC3, cv::Scalar(0));
  cv::Mat foreground_sum(static_cast<int>(frame_height),
                         static_cast<int>(frame_width), CV_8UC1, cv::Scalar(0));
  int frame_count = 0;
  for (;;) {
    std::vector<cv::Mat> frame_arr;
    std::vector<cv::Mat> foreground_arr;
    foreground_sum = cv::Scalar(0);

    for (int i = 0; i < start_frames.size(); i++) {
      if (frame_count <= end_frames[i] - start_frames[i] + 2 * fps) {
        cv::Mat frame, foreground;
        *frame_cap_arr[i] >> frame;
        *fg_cap_arr[i] >> foreground;
        if (!frame.empty() && !foreground.empty()) {
          frame_arr.push_back(frame);
          cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
          cv::threshold(foreground, foreground, 100.0, 1.0, cv::THRESH_BINARY);
          cv::add(foreground, foreground_sum, foreground_sum);
          foreground_arr.push_back(foreground);
        }
      }
    }
    // printf("after read frames\n");
    if (frame_arr.size() <= 0)
      break;

    oframe = cv::Scalar(0);

    float num_frame = static_cast<float>(frame_arr.size());
    // printf("Number of frame %f\n", num_frame);
    for (int v = 0; v < static_cast<int>(frame_height); v++) {
      unsigned char *pfg_sum = foreground_sum.ptr(v);
      unsigned char *poframe = oframe.ptr(v);
      std::vector<unsigned char *> pfg_arr;
      std::vector<unsigned char *> pframe_arr;
      for (int i = 0; i < frame_arr.size(); i++) {
        pframe_arr.push_back(frame_arr[i].ptr(v));
        pfg_arr.push_back(foreground_arr[i].ptr(v));
      }
      for (int u = 0; u < static_cast<int>(frame_width); u++) {

        if (pfg_sum[u] <= 0) {
          float weight = 1.0 / num_frame;
          for (int i = 0; i < frame_arr.size(); i++) {
            poframe[3 * u] += static_cast<unsigned char>(
                weight * static_cast<float>(pframe_arr[i][3 * u]));
            poframe[3 * u + 1] += static_cast<unsigned char>(
                weight * static_cast<float>(pframe_arr[i][3 * u + 1]));
            poframe[3 * u + 2] += static_cast<unsigned char>(
                weight * static_cast<float>(pframe_arr[i][3 * u + 2]));
          }
        } else {
          for (int i = 0; i < frame_arr.size(); i++) {
            float weight;
            if (pfg_arr[i][u] > 0)
              weight = kFgRatio / static_cast<float>(pfg_sum[u]);
            else
              weight = (1.0 - kFgRatio) /
                       static_cast<float>(num_frame -
                                          static_cast<float>(pfg_sum[u]));

            poframe[3 * u] += static_cast<unsigned char>(
                weight * static_cast<float>(pframe_arr[i][3 * u]));
            poframe[3 * u + 1] += static_cast<unsigned char>(
                weight * static_cast<float>(pframe_arr[i][3 * u + 1]));
            poframe[3 * u + 2] += static_cast<unsigned char>(
                weight * static_cast<float>(pframe_arr[i][3 * u + 2]));
          }
        }
        // for(int i=0;i<weight.size();i++)
        // printf("%f ", weight[i]);
        // printf("\n");
      }
    }

    //cv::imshow("frame", frame_arr[2]);
    cv::imshow("oframe", oframe);

    char key = cv::waitKey(1);
    if (key == 27) {
      break;
    }
    ++frame_count;
  }
  return 0;
}

//#include <opencv2/opencv.hpp>

// using namespace cv;

// int main(int argc, const char *argv[]) {
// VideoCapture cap(argv[1]);
////cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
////cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

// if (!cap.isOpened())
// return -1;

// Mat img;
// namedWindow("opencv", CV_WINDOW_AUTOSIZE);
// HOGDescriptor hog;
// hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

// while (true) {
// cap >> img;
// cv::resize(img, img, cv::Size(352, 288));
// if (img.empty())
// continue;

// vector<Rect> found, found_filtered;
// hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
// size_t i, j;
// for (i = 0; i < found.size(); i++) {
// Rect r = found[i];
// for (j = 0; j < found.size(); j++)
// if (j != i && (r & found[j]) == r)
// break;
// if (j == found.size())
// found_filtered.push_back(r);
//}

// for (i = 0; i < found_filtered.size(); i++) {
// Rect r = found_filtered[i];
// r.x += cvRound(r.width * 0.1);
// r.width = cvRound(r.width * 0.8);
// r.y += cvRound(r.height * 0.07);
// r.height = cvRound(r.height * 0.8);
// rectangle(img, r.tl(), r.br(), Scalar(0, 255, 0), 3);
//}

// imshow("opencv", img);
// if (waitKey(1) >= 0)
// break;
//}
// return 0;
//}
