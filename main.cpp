#include "kaylordut/log/logger.h"
#include "ai_instance.h"
#include "yaml-cpp/yaml.h"
#include "depth_anything/depth_imageprocess.h"
#include "kaylordut/time/time.h"

int main(int argc, char **argv) {
  YAML::Node config = YAML::LoadFile(argv[1]);
  auto model_path = config["model_path"].as<std::string>();
  auto image_path = config["image_path"].as<std::string>();
  auto input_size = config["input_size"].as<int>();
  KAYLORDUT_LOG_INFO("model_path: {}\n image_path: {}", model_path, image_path);
  ai_framework::Engine engine(ModelFormat::TRT_FORMAT, model_path.c_str());
  DepthImageprocess depth_imageprocess(engine, input_size, {0.485, 0.456, 0.406},
                                       {0.229, 0.224, 0.225});
    std::vector<cv::Mat> input(1);
  input.at(0) = cv::imread(image_path);
  if (input.at(0).empty()) {
    KAYLORDUT_LOG_ERROR("Could not read image");
    return -1;
  }
  KAYLORDUT_LOG_INFO("Begin...");
  depth_imageprocess.PreProcess(input, engine.get_input_tensor_ptr());
  KAYLORDUT_TIME_COST_INFO("DoInference()", engine.DoInference());
  depth_imageprocess.PostProcess(engine.get_output_tensor_ptr());
  cv::Mat res;
  if (!depth_imageprocess.GetResult(res)) {
    KAYLORDUT_LOG_ERROR("No result");
  }
  cv::Mat display;
  cv::hconcat(input.at(0), res, display);
  kaylordut::Time time;
  auto filename = time.now_str() + ".jpg";
//  cv::imwrite(filename, display);
  cv::imshow("result", display);
  cv::waitKey(0);
  return 0;
}
