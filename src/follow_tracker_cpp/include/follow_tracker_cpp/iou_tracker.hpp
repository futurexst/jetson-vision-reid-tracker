#ifndef FOLLOW_TRACKER_CPP__IOU_TRACKER_HPP_
#define FOLLOW_TRACKER_CPP__IOU_TRACKER_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

namespace follow_tracker_cpp
{

// 트래킹된 사람 1명 (track_id + bbox + 3D + 상태)
struct Track
{
  int id;
  cv::Rect box;          // 원본 픽셀 bbox
  float conf;
  float px, py, pz;      // 3D 좌표 (m, 카메라 광학)
  int cx, cy;            // bbox 중심 픽셀
  int age;               // 생성 후 누적 프레임
  int hits;              // 누적 매칭 횟수
  int time_since_update; // 마지막 매칭 후 경과 프레임
};

// 검출 입력 1건 (DetectionSet.detections 원소 대응)
struct Det
{
  cv::Rect box;
  float conf;
  float px, py, pz;
};

// 단순 IoU 트래커 (BoxMOT 트래커 대체, track_id 부여 목적)
// - IoU 최대 매칭(greedy), 임계값 미달 시 신규 트랙 생성
// - max_age 초과 미매칭 트랙 폐기
class IouTracker
{
public:
  IouTracker() = default;

  // 파라미터 (Python tracker 기본 대응)
  float iou_thresh = 0.3f;
  int max_age = 300;    // 20Hz 기준 약 15초 (Python max_age=300)
  int min_hits = 2;     // Python min_hits=2

  // 검출 목록으로 트랙 갱신 -> 현재 살아있는 트랙 반환
  std::vector<Track> update(const std::vector<Det> & dets);

private:
  static float iou(const cv::Rect & a, const cv::Rect & b);

  std::vector<Track> tracks_;
  int next_id_ = 1;
};

}  // namespace follow_tracker_cpp

#endif  // FOLLOW_TRACKER_CPP__IOU_TRACKER_HPP_
