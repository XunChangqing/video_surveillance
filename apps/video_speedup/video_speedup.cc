#include "vibe.h"
#include "foreground_blob_detector.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/filesystem.hpp>

const int kMinLengthSecs = 3;
const int kMaxDistSecs = 5;
const float kMinAreaRatio = 800.0 / (704.0 * 576.0);

const float kFrameStride = 10;

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }
  std::string frame_filename = std::string(argv[1]);
  cv::VideoCapture frame_cap(frame_filename);
  // frame_cap.set(CV_CAP_PROP_POS_FRAMES, 2534);
  // fg_cap.set(CV_CAP_PROP_POS_FRAMES, 2534);
  float frame_width = frame_cap.get(CV_CAP_PROP_FRAME_WIDTH);
  float frame_height = frame_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  float fps = frame_cap.get(CV_CAP_PROP_FPS);

  std::string clip_prefix =
      frame_filename.substr(0, frame_filename.rfind('.')) + '/';
  std::string fg_filename = clip_prefix + "fg.avi";
  cv::VideoCapture fg_cap(fg_filename);

  if (!frame_cap.isOpened() || !fg_cap.isOpened())
    return 1;

  std::string max_area_file_name = clip_prefix + "active_frame.txt";
  std::vector<int> start_frames, end_frames;
  if (!masa_video_surveillance::GetActiveClip(
          max_area_file_name, kMinAreaRatio * frame_width * frame_height,
          kMinLengthSecs * fps, kMaxDistSecs * fps, start_frames, end_frames)) {
    printf("Failed to read active frame file %s\n", max_area_file_name.c_str());
    return 1;
  }
  // for (int i = 0; i < start_frames.size(); i++) {
  // printf("%d %d\n", start_frames[i], end_frames[i]);
  //}

  int frame_count = 0;
  cv::Mat frame, foreground;
  for (;;) {
    frame_cap >> frame;
    // fg_cap >> foreground;
    // if (frame.empty() || foreground.empty())
    if (frame.empty())
      break;

    // cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
    // cv::threshold(foreground, foreground, 100.0, 255.0, cv::THRESH_BINARY);

    bool active = false;
    for (int i = 0; i < start_frames.size(); i++) {
      if (frame_count >= start_frames[i] - fps &&
          frame_count <= end_frames[i] + fps) {
        active = true;
        break;
      }
    }

    //active = true;
    if (active) {
      if (frame_count % 10 > 5)
        cv::circle(frame, cv::Point(30, 30), 20, cv::Scalar(0, 0, 255), 3);
    }

    cv::imshow("frame", frame);
    // cv::imshow("foreground", foreground);

    if (!active) {
      for (int i = 0; i < kFrameStride; i++)
        frame_cap >> frame;
      frame_count += kFrameStride;
    }

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
