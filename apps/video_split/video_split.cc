#include "vibe.h"
#include "foreground_blob_detector.h"
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <boost/filesystem.hpp>

// const float kMinAreaRatio = 800.0 / (704.0 * 576.0);

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }
  std::string frame_filename = std::string(argv[1]);
  cv::VideoCapture frame_cap(frame_filename);
  if (!frame_cap.isOpened())
    return 1;

  float frame_width = frame_cap.get(CV_CAP_PROP_FRAME_WIDTH);
  float frame_height = frame_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  float fps = frame_cap.get(CV_CAP_PROP_FPS);
  std::string cache_dir_path =
      frame_filename.substr(0, frame_filename.rfind('.')) + '/';

  boost::filesystem::path cache_dir(cache_dir_path);
  boost::filesystem::create_directories(cache_dir);

  std::string fg_filename = cache_dir_path + "fg.avi";

  cv::VideoWriter fg_writer(fg_filename, CV_FOURCC('X', 'V', 'I', 'D'),
                            frame_cap.get(CV_CAP_PROP_FPS),
                            cv::Size(frame_width, frame_height));

  FILE *active_frame_file =
      fopen((cache_dir_path + "active_frame.txt").c_str(), "w");
  if (!active_frame_file)
    return 1;

  masa_video_surveillance::ForegroundBlobDetector::Params params;
  params.filterByArea = true;
  params.filterByColor = false;
  params.filterByInertia = false;
  params.filterByConvexity = false;
  masa_video_surveillance::ForegroundBlobDetector blob_detector(params);

  int channels = 1;
  masa_video_surveillance::VIBE vibe_bgs(channels, 20, 4, 17, 2, 16);
  //masa_video_surveillance::VIBE vibe_bgs(channels, 20, 4, 30, 2, 16);
  bool background_inited = false;

  int64 s_tick = cv::getTickCount();
  double tick_freq = cv::getTickFrequency();
  int frame_count = 0;
  cv::Mat frame, gray, foreground;
  for (;;) {
    frame_cap >> frame;
    if (frame.empty())
      break;

    // fg_cap >> foreground;
    // cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
    // cv::threshold(foreground, foreground, 100.0, 255.0, cv::THRESH_BINARY);
    cv::cvtColor(frame, gray, CV_BGR2GRAY);
    //cv::cvtColor(frame, gray, CV_BGR2HSV);
    //gray = frame.clone();
    if (!background_inited) {
      vibe_bgs.init(gray);
      background_inited = true;
    } else {
      vibe_bgs.update(gray);
    }

    cv::Mat foreground = vibe_bgs.getMask();

    cv::Mat kernel(cv::Size(5, 5), CV_8UC1);
    kernel.setTo(cv::Scalar(1));
    cv::dilate(foreground, foreground, kernel);
    cv::erode(foreground, foreground, kernel);

    std::vector<masa_video_surveillance::ForegroundBlobDetector::Center> centers;
    blob_detector.FindBlobs(frame, foreground, centers);
    double max_area = -1;
    int idx_max_area = -1;
    for (int i = 0; i < centers.size(); i++) {
      if (centers[i].area > max_area) {
        max_area = centers[i].area;
        idx_max_area = i;
      }
    }
    if (max_area > 0) {
      fprintf(active_frame_file, "%f\n", max_area);
      cv::putText(frame, std::to_string(max_area), cv::Point(20, 50),
                  CV_FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 255));
      cv::circle(frame, centers[idx_max_area].location,
                 centers[idx_max_area].radius, cv::Scalar(0, 0, 255), 1);
    } else {
      fprintf(active_frame_file, "%f\n", 0.0);
    }

    cv::Mat vout;
    cv::cvtColor(foreground, vout, CV_GRAY2BGR);
    fg_writer << vout;

    cv::imshow("frame", frame);
    cv::imshow("foreground", foreground);

    char key = cv::waitKey(1);
    if (key == 27) {
      break;
    }
    // printf("%d\n", frame_count);
    if (frame_count % static_cast<int>(fps * 5) == 0) {
      double secs =
          static_cast<double>(cv::getTickCount() - s_tick) / tick_freq;
      printf("fps %f\n", static_cast<double>(frame_count) / secs);
    }
    ++frame_count;
  }
  fg_writer.release();
  fclose(active_frame_file);
  printf("Release video writer!\n");
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
