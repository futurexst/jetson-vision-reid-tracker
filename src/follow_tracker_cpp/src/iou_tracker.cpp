#include "follow_tracker_cpp/iou_tracker.hpp"
#include <algorithm>

namespace follow_tracker_cpp
{

float IouTracker::iou(const cv::Rect & a, const cv::Rect & b)
{
  const int x1 = std::max(a.x, b.x);
  const int y1 = std::max(a.y, b.y);
  const int x2 = std::min(a.x + a.width, b.x + b.width);
  const int y2 = std::min(a.y + a.height, b.y + b.height);
  const int iw = x2 - x1;
  const int ih = y2 - y1;
  if (iw <= 0 || ih <= 0) {
    return 0.0f;
  }
  const float inter = static_cast<float>(iw) * static_cast<float>(ih);
  const float area_a = static_cast<float>(a.width) * static_cast<float>(a.height);
  const float area_b = static_cast<float>(b.width) * static_cast<float>(b.height);
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Track> IouTracker::update(const std::vector<Det> & dets)
{
  const int n_tr = static_cast<int>(tracks_.size());
  const int n_det = static_cast<int>(dets.size());

  std::vector<bool> det_used(n_det, false);
  std::vector<bool> tr_matched(n_tr, false);

  // 기존 트랙 age 증가 (매칭되면 아래에서 리셋)
  for (auto & t : tracks_) {
    t.age += 1;
    t.time_since_update += 1;
  }

  // ===== 1) IoU greedy 매칭 =====
  struct Pair
  {
    float iou;
    int tr;
    int det;
  };
  std::vector<Pair> pairs;
  pairs.reserve(n_tr * n_det);
  for (int i = 0; i < n_tr; ++i) {
    for (int j = 0; j < n_det; ++j) {
      const float v = iou(tracks_[i].box, dets[j].box);
      if (v >= iou_thresh) {
        pairs.push_back({v, i, j});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(),
    [](const Pair & a, const Pair & b) {return a.iou > b.iou;});

  for (const auto & p : pairs) {
    if (tr_matched[p.tr] || det_used[p.det]) {
      continue;
    }
    tr_matched[p.tr] = true;
    det_used[p.det] = true;
    Track & t = tracks_[p.tr];
    const Det & d = dets[p.det];
    t.box = d.box;
    t.conf = d.conf;
    t.px = d.px;
    t.py = d.py;
    t.pz = d.pz;
    t.cx = d.box.x + d.box.width / 2;
    t.cy = d.box.y + d.box.height / 2;
    t.hits += 1;
    t.time_since_update = 0;
  }

  // ===== 2) 미매칭 검출: 신규 트랙 생성 (재식별은 tracker_node RELOCK이 담당) =====
  for (int j = 0; j < n_det; ++j) {
    if (det_used[j]) {
      continue;
    }
    const Det & d = dets[j];
    Track t;
    t.id = next_id_++;
    t.box = d.box;
    t.conf = d.conf;
    t.px = d.px;
    t.py = d.py;
    t.pz = d.pz;
    t.cx = d.box.x + d.box.width / 2;
    t.cy = d.box.y + d.box.height / 2;
    t.age = 1;
    t.hits = 1;
    t.time_since_update = 0;
    tracks_.push_back(t);
  }

  // ===== 3) max_age 초과 미매칭 트랙 폐기 =====
  tracks_.erase(
    std::remove_if(tracks_.begin(), tracks_.end(),
      [this](const Track & t) {return t.time_since_update > max_age;}),
    tracks_.end());

  // ===== 4) 출력: 이번 프레임 실제로 보이는 track만 (유령 track 제외) =====
  std::vector<Track> out;
  out.reserve(tracks_.size());
  for (const auto & t : tracks_) {
    if (t.time_since_update == 0 && t.hits >= min_hits) {
      out.push_back(t);
    }
  }
  return out;
}

}  // namespace follow_tracker_cpp
