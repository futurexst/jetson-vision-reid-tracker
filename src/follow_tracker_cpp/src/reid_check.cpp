#include "follow_tracker_cpp/reid_engine.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cstdio>

// 사용법: reid_check <engine_path> <crop_image_path> [out_txt]
// crop 이미지를 CLIP 전처리 -> 임베딩 추출 -> 값 출력(비교용)
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  if (argc < 3) {
    printf("usage: reid_check <engine> <crop_image> [out_txt]\n");
    return 1;
  }
  std::string engine_path = argv[1];
  std::string img_path = argv[2];
  std::string out_txt = (argc >= 4) ? argv[3] : "";

  follow_tracker_cpp::ReidEngine reid;
  if (!reid.load(engine_path)) {
    printf("ENGINE LOAD FAIL\n");
    return 2;
  }

  cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);  // BGR
  if (img.empty()) {
    printf("IMAGE READ FAIL: %s\n", img_path.c_str());
    return 3;
  }
  printf("crop size: %dx%d\n", img.cols, img.rows);

  auto feat = reid.extract(img);
  if (feat.empty()) {
    printf("EXTRACT FAIL\n");
    return 4;
  }
  printf("feat_dim=%d\n", static_cast<int>(feat.size()));

  // 앞 8개 값 출력
  printf("first8:");
  for (int i = 0; i < 8 && i < static_cast<int>(feat.size()); ++i) {
    printf(" %.6f", feat[i]);
  }
  printf("\n");

  // L2 norm 확인 (정규화됐으니 ~1.0)
  double norm = 0.0;
  for (float v : feat) {norm += static_cast<double>(v) * v;}
  printf("L2norm=%.6f\n", std::sqrt(norm));

  // 파일로 전체 저장 (파이썬 비교용)
  if (!out_txt.empty()) {
    FILE * f = fopen(out_txt.c_str(), "w");
    if (f) {
      for (size_t i = 0; i < feat.size(); ++i) {
        fprintf(f, "%.8f\n", feat[i]);
      }
      fclose(f);
      printf("saved: %s\n", out_txt.c_str());
    }
  }

  rclcpp::shutdown();
  return 0;
}
