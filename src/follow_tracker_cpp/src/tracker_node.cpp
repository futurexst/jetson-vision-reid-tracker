#include "follow_tracker_cpp/tracker_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <csignal>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

namespace follow_tracker_cpp
{

// ===== MJPEG 스트림 전역 버퍼 =====
namespace
{
std::mutex g_frame_mtx;
cv::Mat g_frame;

void mjpeg_serve(int port)
{
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) {return;}
  int opt = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (bind(srv, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    close(srv);
    return;
  }
  listen(srv, 4);
  while (true) {
    int cli = accept(srv, nullptr, nullptr);
    if (cli < 0) {continue;}
    const char * header =
      "HTTP/1.0 200 OK\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (write(cli, header, std::strlen(header)) < 0) {close(cli); continue;}
    std::vector<uchar> buf;
    bool alive = true;
    while (alive) {
      cv::Mat frame;
      {
        std::lock_guard<std::mutex> lk(g_frame_mtx);
        if (!g_frame.empty()) {frame = g_frame.clone();}
      }
      if (!frame.empty()) {
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
        cv::imencode(".jpg", frame, buf, params);
        char part[128];
        int n = snprintf(part, sizeof(part),
          "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
          buf.size());
        if (write(cli, part, n) < 0) {alive = false; break;}
        if (write(cli, reinterpret_cast<char *>(buf.data()), buf.size()) < 0) {
          alive = false; break;
        }
        if (write(cli, "\r\n", 2) < 0) {alive = false; break;}
      }
      usleep(100000);   // ~10fps
    }
    close(cli);
  }
}
}  // namespace

TrackerNode::TrackerNode(const rclcpp::NodeOptions & options)
: Node("tracker_node", options)
{
  // ReID 엔진 로드
  const std::string engine_path =
    "models/clip_msmt17_fp16.engine";
  if (!reid_.load(engine_path)) {
    RCLCPP_ERROR(get_logger(), "ReID 엔진 로드 실패: %s", engine_path.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "ReID 엔진 로드 완료 (feat_dim=%d)", reid_.feat_dim());
  }

  // 트래커 파라미터 (Python max_age/min_hits 대응)
  tracker_.max_age = MISS_LIMIT;   // track 소멸 기준
  tracker_.min_hits = 2;

  // 구독
  det_sub_ = create_subscription<follow_tracker_msgs::msg::DetectionSet>(
    "/camera_infer_node/detections", 10,
    std::bind(&TrackerNode::on_detections, this, std::placeholders::_1));
  enable_sub_ = create_subscription<std_msgs::msg::Bool>(
    "/follow_enable", 10,
    std::bind(&TrackerNode::on_follow_enable, this, std::placeholders::_1));
  agv_mode_sub_ = create_subscription<std_msgs::msg::Int32>(
    "/agv_mode", 10,
    std::bind(&TrackerNode::on_agv_mode, this, std::placeholders::_1));

  // 발행
  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/target_cmd", 10);

  // MJPEG write 중 클라이언트 끊김(SIGPIPE)으로 프로세스 죽는 것 방지
  signal(SIGPIPE, SIG_IGN);
  start_stream_server(8080);
  RCLCPP_INFO(get_logger(), "tracker_node started");
}

// ===== 콜백: enable / agv_mode =====
void TrackerNode::on_follow_enable(const std_msgs::msg::Bool::SharedPtr msg)
{
  enabled_ = msg->data;
  RCLCPP_INFO(get_logger(), "[ENABLE] follow_enable = %s", enabled_ ? "true" : "false");
  if (!enabled_) {
    stop_cmd();
  }
}

void TrackerNode::on_agv_mode(const std_msgs::msg::Int32::SharedPtr msg)
{
  const int cat = msg->data / 10;
  const bool new_follow = (cat == 1);
  // 추종 새로 진입하는 엣지에서 락 상태 리셋 -> 새 타겟 재락 허용
  if (new_follow && !follow_cat_) {
    locked_id_ = -1;
    locked_feat_.clear();
    locked_hist_.release();
    has_locked_once_ = false;
    target_miss_ = 0;
    relock_streak_ = 0;
    relock_cand_id_ = -1;
    occ_blacklist_.clear();
    RCLCPP_INFO(get_logger(), "[MODE] 추종 재진입 -> 락 상태 리셋");
  }
  follow_cat_ = new_follow;
  guide_mode_ = (cat == 2);
  pickup_cat_ = (cat == 5);
}

// ===== 발행 유틸 =====
void TrackerNode::publish_cmd(float mode, float distance, float angle, bool force)
{
  if ((!enabled_ || pickup_cat_) && !force) {
    return;
  }
  geometry_msgs::msg::Twist msg;
  msg.linear.x = distance;
  msg.linear.z = mode;
  msg.angular.z = angle;
  cmd_pub_->publish(msg);
}

void TrackerNode::stop_cmd()
{
  publish_cmd(MODE_FIND, 0.0f, 0.0f, true);
}

float TrackerNode::pixel_to_angle(int cx) const
{
  const float cx_err = static_cast<float>(cx - CENTER_X);
  return -(cx_err / static_cast<float>(IMG_WIDTH)) *
         static_cast<float>(HFOV_DEG * M_PI / 180.0);
}

// ===== ReID 임베딩 헬퍼 =====
std::vector<float> TrackerNode::extract_feat(const cv::Rect & box, const cv::Mat & img)
{
  cv::Rect safe = box & cv::Rect(0, 0, img.cols, img.rows);
  if (safe.width <= 0 || safe.height <= 0) {
    return {};
  }
  cv::Mat crop = img(safe);
  return reid_.extract(crop);   // 내부에서 L2 정규화됨
}

void TrackerNode::update_locked_feat(const std::vector<float> & feat)
{
  if (feat.empty()) {
    return;
  }
  if (locked_feat_.empty()) {
    locked_feat_ = feat;
    return;
  }
  // 오염 방지: 현재 임베딩이 기존 locked_feat와 충분히 유사할 때만 갱신
  float _sim = 0.0f;
  for (size_t i = 0; i < locked_feat_.size(); ++i) {
    _sim += locked_feat_[i] * feat[i];
  }
  if (_sim < EMA_UPDATE_GATE) {
    return;   // 타인 의심 -> EMA 오염 차단(갱신 스킵)
  }
  // EMA 갱신 후 L2 재정규화
  double norm = 0.0;
  for (size_t i = 0; i < locked_feat_.size(); ++i) {
    locked_feat_[i] = REID_EMA * locked_feat_[i] + (1.0f - REID_EMA) * feat[i];
    norm += static_cast<double>(locked_feat_[i]) * locked_feat_[i];
  }
  norm = std::sqrt(norm);
  if (norm > 0.0) {
    for (auto & v : locked_feat_) {
      v = static_cast<float>(v / norm);
    }
  }
}

// ===== 색상 히스토그램 헬퍼 =====
cv::Mat TrackerNode::extract_hist(const cv::Rect & box, const cv::Mat & img)
{
  // 가장자리 15% 잘라 배경/팔 영향 줄임
  int mx = static_cast<int>(box.width * 0.15);
  int my = static_cast<int>(box.height * 0.15);
  cv::Rect inner(box.x + mx, box.y + my, box.width - 2 * mx, box.height - 2 * my);
  inner &= cv::Rect(0, 0, img.cols, img.rows);
  if (inner.width <= 0 || inner.height <= 0) {
    return cv::Mat();
  }
  cv::Mat hsv;
  cv::cvtColor(img(inner), hsv, cv::COLOR_BGR2HSV);
  cv::Mat hist;
  const int channels[] = {0, 1};
  const int histSize[] = {30, 32};
  const float hrange[] = {0, 180};
  const float srange[] = {0, 256};
  const float * ranges[] = {hrange, srange};
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
  cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
  return hist;
}

void TrackerNode::update_locked_hist(const cv::Mat & hist)
{
  if (hist.empty()) {
    return;
  }
  if (locked_hist_.empty()) {
    locked_hist_ = hist.clone();
  } else {
    locked_hist_ = COLOR_EMA * locked_hist_ + (1.0f - COLOR_EMA) * hist;
  }
}

float TrackerNode::color_sim(const cv::Rect & box, const cv::Mat & img)
{
  if (locked_hist_.empty()) {
    return 1.0f;   // 색상 기준 없으면 게이트 통과
  }
  cv::Mat h = extract_hist(box, img);
  if (h.empty()) {
    return 0.0f;
  }
  return static_cast<float>(
    cv::compareHist(locked_hist_, h, cv::HISTCMP_CORREL));
}


// ===== 게이트 판정 (Python _gate_decide 이식) =====
// 반환: 0=PASS(osnet-strong), 1=PASS(color), 2=BLOCK(side), 3=BLOCK(dist), 4=BLOCK
namespace
{
enum GateResult { G_PASS_STRONG = 0, G_PASS_COLOR = 1, G_BLOCK_SIDE = 2,
  G_BLOCK_DIST = 3, G_BLOCK = 4 };
inline bool gate_is_block(int g) {return g >= G_BLOCK_SIDE;}
}  // namespace

// ===== best_match_id (Python _best_match_id 이식) =====
// people: 트랙 목록. 반환: locked 타겟 track_id(-1=없음), out_sim에 최고 유사도
int TrackerNode::best_match_id(
  const std::vector<Track> & tracks, const cv::Mat & img, float & out_sim)
{
  out_sim = -1.0f;
  if (locked_feat_.empty()) {
    return -1;
  }
  int best_tid = -1;
  float best_sim = -1.0f;

  for (const auto & t : tracks) {
    const int tid = t.id;
    std::vector<float> f = extract_feat(t.box, img);
    if (f.empty()) {
      continue;
    }
    // 코사인 유사도 (양쪽 L2 정규화 -> 내적)
    float sim = 0.0f;
    for (size_t i = 0; i < locked_feat_.size(); ++i) {
      sim += locked_feat_[i] * f[i];
    }

    // 블랙리스트 track: osnet-strong이면 조기해제, 아니면 재락 금지
    auto it = std::find(occ_blacklist_.begin(), occ_blacklist_.end(), tid);
    if (it != occ_blacklist_.end()) {
      if (sim >= OSNET_STRONG) {
        occ_blacklist_.erase(it);
        RCLCPP_INFO(get_logger(),
          "[OCC-RELEASE] id=%d osnet=%.2f >= strong -> 블랙리스트 조기해제",
          tid, sim);
      } else {
        continue;   // 타인 추정 -> 재락 금지 유지
      }
    }

    const int p_cx = t.cx;
    const double p_dist = static_cast<double>(t.pz);

    // 방향 게이트
    bool side_block = false;
    if (last_seen_side_ == "left" && p_cx > CENTER_X + SIDE_MARGIN) {
      side_block = true;
    } else if (last_seen_side_ == "right" && p_cx < CENTER_X - SIDE_MARGIN) {
      side_block = true;
    }
    // 거리 게이트
    bool dist_block = false;
    if (last_dist_ > 0.0 && p_dist > 0.0) {
      if (std::abs(p_dist - last_dist_) > DIST_GATE) {
        dist_block = true;
      }
    }
    // 공간 게이트 확정 시 색상 연산 생략 (부하 절감)
    float csim;
    if (side_block || dist_block) {
      csim = 0.0f;
    } else {
      csim = color_sim(t.box, img);
    }
    // 게이트 판정 (side/dist 최우선)
    int gate;
    if (side_block) {
      gate = G_BLOCK_SIDE;
    } else if (dist_block) {
      gate = G_BLOCK_DIST;
    } else if (sim >= OSNET_STRONG) {
      gate = G_PASS_STRONG;
    } else if (csim >= COLOR_SIM_THRESH) {
      gate = G_PASS_COLOR;
    } else {
      gate = G_BLOCK;
    }
    RCLCPP_INFO(get_logger(),
      "[CAND] id=%d osnet=%.2f color=%.2f gate=%d", tid, sim, csim, gate);
    if (gate_is_block(gate)) {
      continue;
    }
    if (sim > best_sim) {
      best_sim = sim;
      best_tid = tid;
    }
  }

  out_sim = best_sim;
  if (best_tid != -1 && best_sim >= REID_SIM_THRESH) {
    return best_tid;
  }
  return -1;
}


// ===== 메인 루프: DetectionSet 수신 (Python loop() 이식) =====
void TrackerNode::on_detections(
  const follow_tracker_msgs::msg::DetectionSet::SharedPtr msg)
{
  // 추종/안내/픽업 카테고리 아니면 트래킹 스킵
  if (!(follow_cat_ || guide_mode_ || pickup_cat_)) {
    return;
  }

  // color 프레임 복원 (ReID crop / 색상 히스토그램용)
  cv::Mat img;
  try {
    img = cv_bridge::toCvCopy(msg->color, "bgr8")->image;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge 변환 실패: %s", e.what());
    return;
  }

  // DetectionSet -> Det 목록
  std::vector<Det> dets;
  dets.reserve(msg->detections.size());
  for (const auto & d : msg->detections) {
    Det det;
    det.box = cv::Rect(d.x, d.y, d.w, d.h);
    det.conf = d.conf;
    det.px = d.px;
    det.py = d.py;
    det.pz = d.pz;
    dets.push_back(det);
  }

  // IoU 트래커 update -> track_id 부여
  std::vector<Track> tracks = tracker_.update(dets);
  frame_count_ += 1;

  // 워밍업
  if (!started_) {
    if (frame_count_ >= LOCK_FRAMES) {
      started_ = true;
      RCLCPP_INFO(get_logger(), "[START] 추종 시작");
    }
    {
      std::lock_guard<std::mutex> lk(g_frame_mtx);
      g_frame = img.clone();
    }
    return;
  }

  // 안내 모드는 후방 감시(별도 노드) -> 여기선 추종 로직만. 정지 유지.
  if (guide_mode_) {
    return;
  }

  // 유효 사람 목록 (pz>0)
  std::vector<Track> people;
  for (const auto & t : tracks) {
    if (t.pz > 0.0f) {
      people.push_back(t);
    }
  }

  // 살아있는 track_id 집합 (블랙리스트 화면이탈 해제 / 락 소멸 판정)
  std::vector<int> alive_ids;
  for (const auto & t : tracks) {
    alive_ids.push_back(t.id);
  }

  // 블랙리스트 화면이탈 해제: alive에 없는 blacklist id 제거
  if (!occ_blacklist_.empty()) {
    occ_blacklist_.erase(
      std::remove_if(occ_blacklist_.begin(), occ_blacklist_.end(),
        [&alive_ids](int id) {
          return std::find(alive_ids.begin(), alive_ids.end(), id)
                 == alive_ids.end();
        }),
      occ_blacklist_.end());
  }

  float distance = 0.0f, angle = 0.0f;
  double now = this->now().seconds();

  // ===== find(재발견 정지) 유지 중 =====
  if (find_until_ > 0.0) {
    if (now < find_until_) {
      publish_cmd(MODE_FIND, 0.0f, 0.0f);
      return;
    } else {
      find_until_ = 0.0;
      searching_ = people.empty();
    }
  }

  // 거리 오름차순 정렬 (가까운 사람 우선)
  std::sort(people.begin(), people.end(),
    [](const Track & a, const Track & b) {return a.pz < b.pz;});

  // target 선택 (-1 인덱스 = 미확정)
  int target_idx = -1;

  if (!people.empty()) {
    if (locked_id_ == -1) {
      // ===== UNLOCK: 재식별 시도 =====
      float sim;
      int rematch_tid = best_match_id(people, img, sim);
      if (rematch_tid != -1) {
        // 연속 프레임 재락: 같은 후보 RELOCK_STREAK회 연속 PASS
        if (rematch_tid == relock_cand_id_) {
          relock_streak_ += 1;
        } else {
          relock_cand_id_ = rematch_tid;
          relock_streak_ = 1;
        }
        if (relock_streak_ >= RELOCK_STREAK) {
          locked_id_ = rematch_tid;
          for (size_t i = 0; i < people.size(); ++i) {
            if (people[i].id == rematch_tid) {target_idx = static_cast<int>(i);}
          }
          RCLCPP_INFO(get_logger(), "[RELOCK] id=%d sim=%.2f (streak=%d)",
            locked_id_, sim, relock_streak_);
          relock_count_ += 1;
          relock_cand_id_ = -1;
          relock_streak_ = 0;
        } else {
          RCLCPP_INFO(get_logger(), "[RELOCK-WAIT] id=%d sim=%.2f streak=%d/%d",
            rematch_tid, sim, relock_streak_, RELOCK_STREAK);
        }
      } else {
        // 재식별 실패: 최초 1회만 새 타겟 락
        if (!has_locked_once_) {
          target_idx = 0;
          locked_id_ = people[0].id;
          locked_feat_.clear();
          locked_hist_.release();
          has_locked_once_ = true;
          RCLCPP_INFO(get_logger(), "[LOCK] target id=%d", locked_id_);
          update_locked_feat(extract_feat(people[0].box, img));
          update_locked_hist(extract_hist(people[0].box, img));
        } else {
          RCLCPP_INFO(get_logger(), "[WAIT] 타겟 재식별 대기 (새 사람 무시)");
        }
      }
    } else {
      // ===== LOCK: locked_id 매칭만 선택 =====
      int matched_idx = -1;
      for (size_t i = 0; i < people.size(); ++i) {
        if (people[i].id == locked_id_) {matched_idx = static_cast<int>(i); break;}
      }
      if (matched_idx != -1) {
        target_miss_ = 0;
        target_idx = matched_idx;
        // EMA 갱신: 5프레임마다만 CLIP forward (부하 절감)
        if (frame_count_ % 5 == 0) {
          update_locked_feat(extract_feat(people[matched_idx].box, img));
          update_locked_hist(extract_hist(people[matched_idx].box, img));
        }
      } else {
        // 락된 사람 안 보임 -> 블랙리스트 등록 판정 후 UNLOCK
        bool locked_alive =
          std::find(alive_ids.begin(), alive_ids.end(), locked_id_)
          != alive_ids.end();
        if (!locked_alive) {
          std::vector<int> occluders;
          // 가장자리 walk-out은 가림 아님: 마지막 위치가 화면 양끝이면 블랙리스트 스킵
          bool edge_out = (last_cx_ >= 0) &&
            (last_cx_ < OCC_EDGE_MARGIN ||
             last_cx_ > IMG_WIDTH - OCC_EDGE_MARGIN);
          if (last_cx_ >= 0 && last_dist_ > 0.0 && !edge_out) {
            for (const auto & t : tracks) {
              bool near_cx = std::abs(t.cx - last_cx_) <= OCC_CX_MARGIN;
              double tdepth = static_cast<double>(t.pz);
              bool near_d = (tdepth <= 0.0 ||
                std::abs(tdepth - last_dist_) <= OCC_DIST_MARGIN);
              if (near_cx && near_d) {
                occluders.push_back(t.id);
                if (std::find(occ_blacklist_.begin(), occ_blacklist_.end(), t.id)
                    == occ_blacklist_.end()) {
                  occ_blacklist_.push_back(t.id);
                }
              }
            }
          }
          if (!occluders.empty()) {
            std::string ids;
            for (int oid : occluders) {ids += std::to_string(oid) + " ";}
            RCLCPP_INFO(get_logger(),
              "[OCCLUSION] id=%d 가림 감지 -> 재락금지 occ_ids=[%s] last_cx=%d last_dist=%.2f",
              locked_id_, ids.c_str(), last_cx_, last_dist_);
          } else if (edge_out) {
            RCLCPP_INFO(get_logger(),
              "[WALKOUT] id=%d 가장자리 이탈(last_cx=%d) -> 블랙리스트 스킵",
              locked_id_, last_cx_);
          }
          RCLCPP_INFO(get_logger(),
            "[UNLOCK] id=%d track 소멸 -> 재식별 대기(feat 유지)", locked_id_);
          unlock_count_ += 1;
          locked_id_ = -1;
          target_miss_ = 0;
        }
        people.clear();   // lost/search 경로로
      }
    }
  }

  // ===== 타겟 확정: 추종 발행 =====
  if (target_idx != -1 && !people.empty()) {
    const Track & tgt = people[target_idx];
    if (searching_) {
      find_until_ = now + FIND_HOLD_SEC;
      searching_ = false;
      RCLCPP_INFO(get_logger(), "[FIND] 사람 재발견 -> 정지");
      publish_cmd(MODE_FIND, 0.0f, 0.0f);
    } else {
      distance = tgt.pz;
      angle = pixel_to_angle(tgt.cx);
      lost_count_ = 0;
      last_cx_ = tgt.cx;
      last_dist_ = static_cast<double>(tgt.pz);
      if (tgt.cx < CENTER_X - 40) {
        last_seen_side_ = "left";
      } else if (tgt.cx > CENTER_X + 40) {
        last_seen_side_ = "right";
      }
      publish_cmd(MODE_FOLLOW, distance, angle);
    }
  }

  // ===== 사람 없음: lost/search =====
  std::string status = "FOLLOWING";
  if (people.empty()) {
    lost_count_ += 1;
    if (lost_count_ >= LOST_DELAY) {
      searching_ = true;
      if (last_seen_side_ == "right") {
        status = "SEARCH_RIGHT";
        publish_cmd(MODE_SEARCH_RIGHT, 0.0f, 0.0f);
      } else {
        status = "SEARCH_LEFT";
        publish_cmd(MODE_SEARCH_LEFT, 0.0f, 0.0f);
      }
    } else {
      status = "LOST_WAIT";
    }
  } else if (target_idx == -1) {
    status = (locked_id_ == -1) ? "RELOCK_WAIT" : "WAIT";
  }

  // ===== 오버레이 그리기 + 스트림 송출 =====
  int tgt_id = (target_idx != -1 && !people.empty())
    ? people[target_idx].id : locked_id_;
  draw_overlay(img, people, tgt_id, status, distance, angle);
}


// ===== MJPEG 서버 시작 =====
void TrackerNode::start_stream_server(int port)
{
  stream_thread_ = std::thread(mjpeg_serve, port);
  stream_thread_.detach();
  RCLCPP_INFO(get_logger(),
    "MJPEG stream: http://192.168.0.103:%d/", port);
}

// ===== fancy 브래킷 박스 (Python draw_fancy_box 이식) =====
namespace
{
void draw_fancy_box(
  cv::Mat & img, const cv::Rect & box, const std::string & label,
  const cv::Scalar & color, int thickness, bool filled_label)
{
  const int x1 = box.x, y1 = box.y;
  const int x2 = box.x + box.width, y2 = box.y + box.height;
  const int L = std::max(15, static_cast<int>(box.width * 0.18));

  // 얇은 전체 사각형
  cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 1, cv::LINE_AA);

  // 모서리 L자 브래킷
  const int corners[4][4] = {
    {x1, y1, 1, 1}, {x2, y1, -1, 1}, {x1, y2, 1, -1}, {x2, y2, -1, -1}};
  for (const auto & c : corners) {
    int px = c[0], py = c[1], dx = c[2], dy = c[3];
    cv::line(img, cv::Point(px, py), cv::Point(px + dx * L, py),
      color, thickness, cv::LINE_AA);
    cv::line(img, cv::Point(px, py), cv::Point(px, py + dy * L),
      color, thickness, cv::LINE_AA);
  }

  // 라벨
  int baseline = 0;
  cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
  int ly = std::max(y1 - 8, ts.height + 6);
  cv::Scalar txt_color;
  if (filled_label) {
    cv::rectangle(img, cv::Point(x1, ly - ts.height - 6),
      cv::Point(x1 + ts.width + 10, ly + 2), color, -1, cv::LINE_AA);
    txt_color = cv::Scalar(255, 255, 255);
  } else {
    txt_color = color;
  }
  cv::putText(img, label, cv::Point(x1 + 5, ly - 3),
    cv::FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv::LINE_AA);
}

// 반투명 배경 패널 위 텍스트
void draw_text_panel(
  cv::Mat & img, const std::string & text, int x, int y,
  const cv::Scalar & color, double scale)
{
  int baseline = 0;
  cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
  cv::Rect bg(x - 4, y - ts.height - 6, ts.width + 12, ts.height + 12);
  bg &= cv::Rect(0, 0, img.cols, img.rows);
  if (bg.width > 0 && bg.height > 0) {
    cv::Mat roi = img(bg);
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(dark, 0.45, roi, 0.55, 0, roi);
  }
  cv::putText(img, text, cv::Point(x, y),
    cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}
}  // namespace

// ===== 오버레이 그리기 =====
void TrackerNode::draw_overlay(
  cv::Mat & img, const std::vector<Track> & people,
  int target_id, const std::string & status, float distance, float angle)
{
  // 중앙 세로선 (파랑)
  cv::line(img, cv::Point(CENTER_X, 0), cv::Point(CENTER_X, 480),
    cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

  // 사람 박스: 타겟=빨강 굵게 filled, 후보=회색
  for (const auto & p : people) {
    char lab[64];
    if (p.id == target_id) {
      snprintf(lab, sizeof(lab), "TARGET LOCKED  %.2fm", p.pz);
      draw_fancy_box(img, p.box, lab, cv::Scalar(0, 0, 255), 3, true);
    } else {
      snprintf(lab, sizeof(lab), "ID:%d  %.2fm", p.id, p.pz);
      draw_fancy_box(img, p.box, lab, cv::Scalar(255, 255, 255), 2, false);
    }
  }

  // 상태 텍스트 (반투명 패널 위, 차분한 초록)
  draw_text_panel(img, status, 10, 34, cv::Scalar(130, 210, 102), 0.7);

  // dist / ang
  char da[96];
  snprintf(da, sizeof(da), "dist %.2fm   ang %+.3frad", distance, angle);
  draw_text_panel(img, da, 10, 66, cv::Scalar(90, 220, 255), 0.6);

  // 락 상태
  char lk[80];
  snprintf(lk, sizeof(lk), "locked_id=%d   blacklist=%zu",
    locked_id_, occ_blacklist_.size());
  draw_text_panel(img, lk, 10, 96, cv::Scalar(255, 200, 90), 0.55);

  std::lock_guard<std::mutex> lk2(g_frame_mtx);
  g_frame = img.clone();
}

}  // namespace follow_tracker_cpp

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(follow_tracker_cpp::TrackerNode)
