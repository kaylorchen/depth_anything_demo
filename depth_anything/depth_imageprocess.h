//
// Created by kaylor on 7/10/24.
//

#pragma once
#include "opencv2/opencv.hpp"
#include "ai_instance.h"
class DepthImageprocess {
 public:
  DepthImageprocess(const ai_framework::Engine &engine, int target_size,
                    std::vector<float> mean_values = {0.0, 0.0, 0.0},
                    std::vector<float> std_values = {1.0, 1.0, 1.0},
                    bool debug = false);
  void PreProcess(const std::vector<cv::Mat> &input, uint8_t *tenssors[]);
  void PostProcess(const uint8_t *const tensors[]);
  bool GetResult(cv::Mat &result);
  const int &get_target_size() const { return target_size_; }

 private:
  void MakeSquare(const cv::Mat &src, cv::Mat &dst);
  uint64_t PopulateData(const cv::Mat &data, float *dst);
  int target_size_;
  std::vector<float> mean_values_;
  std::vector<float> std_values_;
  bool debug_{false};
  std::queue<cv::Mat> depth_queue_;
  std::mutex depth_queue_mutex_;
  ModelFormat model_format_;
  std::vector<cv::Size> input_mat_size_;
  std::vector<cv::Size> resize_mat_size_;
  std::vector<float> output_scale_;
  std::vector<int> output_zero_points_;
  std::vector<std::vector<int64_t>> input_shape_;
#ifdef RK3588
  std::vector<bool> input_width_equal_stride_;
  std::vector<uint32_t> input_stride_;
#endif
};
