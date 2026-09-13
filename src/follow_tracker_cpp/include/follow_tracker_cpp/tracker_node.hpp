#ifndef FOLLOW_TRACKER_CPP__TRACKER_NODE_HPP_
#define FOLLOW_TRACKER_CPP__TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <follow_tracker_msgs/msg/detection_set.hpp>

#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>
#include <string>
#include <vector>

#include "follow_tracker_cpp/iou_tracker.hpp"
#include "follow_tracker_cpp/reid_engine.hpp"

namespace follow_tracker_cpp
{

class TrackerNode : public rclcpp::Node
{
public:
  explicit TrackerNode(const rclcpp::NodeOptions & options);

private:
  // ===== 콜백 =====
  void on_detections(const follow_tracker_msgs::msg::DetectionSet::SharedPtr msg);
  void on_follow_enable(const std_msgs::msg::Bool::SharedPtr msg);
  void on_agv_mode(const std_msgs::msg::Int32::SharedPtr msg);

  // ===== 발행 유틸 =====
  void publish_cmd(float mode, float distance, float angle, bool force = false);
  void stop_cmd();

  // ===== Re-ID 로직 =====
  std::vector<float> extract_feat(const cv::Rect & box, const cv::Mat & img);
  void update_locked_feat(const std::vector<float> & feat);
  cv::Mat extract_hist(const cv::Rect & box, const cv::Mat & img);
  void update_locked_hist(const cv::Mat & hist);
  float color_sim(const cv::Rect & box, const cv::Mat & img);
  // 후보 중 locked 타겟 재식별. 반환: track_id(-1=없음), sim은 out
  int best_match_id(const std::vector<Track> & tracks, const cv::Mat & img, float & out_sim);

  float pixel_to_angle(int cx) const;

  // ===== 구독/발행 =====
  rclcpp::Subscription<follow_tracker_msgs::msg::DetectionSet>::SharedPtr det_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr agv_mode_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

  // ===== 트래커/ReID =====
  IouTracker tracker_;
  ReidEngine reid_;

  // ===== 파라미터 (Python follow_node / handoff 기준) =====
  static constexpr int CENTER_X = 320;
  static constexpr int IMG_WIDTH = 640;
  static constexpr double HFOV_DEG = 69.0;
  static constexpr int LOCK_FRAMES = 15;      // 워밍업
  static constexpr int LOST_DELAY = 8;        // 사라진 후 탐색 시작 지연
  static constexpr double FIND_HOLD_SEC = 0.5;

  // 모드 플래그 (linear.z)
  static constexpr float MODE_FOLLOW = 0.0f;
  static constexpr float MODE_SEARCH_LEFT = 1.0f;
  static constexpr float MODE_SEARCH_RIGHT = 2.0f;
  static constexpr float MODE_FIND = 3.0f;

  // Re-ID 게이트 (handoff 스펙)
  static constexpr float REID_SIM_THRESH = 0.55f;   // C++ CLIP 실측: 본인0.79+ vs 타인0.51- 경계
  static constexpr float OSNET_STRONG = 0.78f;   // 색상게이트 면제
  static constexpr float OSNET_MID = 0.55f;
  static constexpr float REID_EMA = 0.9f;
  static constexpr float EMA_UPDATE_GATE = 0.7f;   // 오염 방지: 처음 락 대상 지문 고정(이 미만 유사도면 갱신 스킵)
  static constexpr float COLOR_SIM_THRESH = 0.5f;
  static constexpr float COLOR_EMA = 0.9f;
  static constexpr int RELOCK_STREAK = 5;        // 연속 매칭 확정 프레임
  static constexpr int MISS_LIMIT = 60;          // 폐기 (20Hz=3초)

  // 공간 게이트
  static constexpr int SIDE_MARGIN = 80;
  static constexpr double DIST_GATE = 0.8;
  // 가림 블랙리스트
  static constexpr int OCC_CX_MARGIN = 150;
  static constexpr double OCC_DIST_MARGIN = 1.0;
  static constexpr int OCC_EDGE_MARGIN = 70;   // 가장자리 walk-out은 가림 아님(블랙리스트 제외)

  // ===== 상태 =====
  bool enabled_ = true;
  bool follow_cat_ = false;
  bool guide_mode_ = false;
  bool pickup_cat_ = false;

  int locked_id_ = -1;
  bool has_locked_once_ = false;
  int target_miss_ = 0;
  int relock_streak_ = 0;       // 재식별 후보 연속 매칭 카운트
  int relock_cand_id_ = -1;     // 연속 매칭 중인 후보 id

  std::vector<float> locked_feat_;   // L2 정규화 임베딩 (비었으면 없음)
  cv::Mat locked_hist_;              // HSV 색상 지문

  std::string last_seen_side_;       // "left"/"right"/""
  int last_cx_ = -1;
  double last_dist_ = -1.0;
  int lost_count_ = 0;
  bool searching_ = false;
  double find_until_ = 0.0;
  bool started_ = false;
  int frame_count_ = 0;

  // 가림 블랙리스트: track_id 집합
  std::vector<int> occ_blacklist_;

  // 통계
  int relock_count_ = 0;
  int unlock_count_ = 0;

  // ===== MJPEG 스트림 =====
  void start_stream_server(int port);
  void draw_overlay(cv::Mat & img, const std::vector<Track> & people,
    int target_id, const std::string & status, float distance, float angle);
  std::thread stream_thread_;
};

}  // namespace follow_tracker_cpp

#endif  // FOLLOW_TRACKER_CPP__TRACKER_NODE_HPP_
