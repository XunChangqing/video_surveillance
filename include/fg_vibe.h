#ifndef VIDEO_COMPRESSION_VIBE_H
#define VIDEO_COMPRESSION_VIBE_H

#include "utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/blobtrack.hpp>
#include <memory>

#define RANDOM_BUFFER_SIZE (65535)

namespace masa_video_surveillance {
struct FGVIBEParams {
  int channels;
  int samples;
  int pixel_neighbor;
  int distance_threshold;
  int matching_threshold;
  int update_factor;
};

class FGVIBE : public CvFGDetector {
public:
  FGVIBE(void *params);
  ~FGVIBE();
  virtual void Process(IplImage *pImg);
  virtual IplImage *GetMask();
  virtual void Release();

private:
  void init(const cv::Mat &img);
  void update(const cv::Mat &img);
  cv::Vec2i getRndNeighbor(int i, int j);

  FGVIBEParams params_;
  int samples_;
  int channels_;
  int pixel_neighbor_;

  cv::Size size_;
  unsigned char *model_;

  cv::Mat mask_;
  IplImage mask_img_;

  unsigned int rng_[RANDOM_BUFFER_SIZE];
  int rng_idx_;

  DISALLOW_COPY_AND_ASSIGN(FGVIBE);
};
}

#endif //_VIDEO_COMPRESSION_VIBE_H
