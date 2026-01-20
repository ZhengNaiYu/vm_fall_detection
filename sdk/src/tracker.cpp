#include "tracker.h"
#include <algorithm>

using visionmatrixsdk::falldetection::InferItem;

float SimpleByteTracker::IoU(const InferItem& a, const InferItem& b) const {
    int x1 = std::max(a.x1, b.x1);
    int y1 = std::max(a.y1, b.y1);
    int x2 = std::min(a.x2, b.x2);
    int y2 = std::min(a.y2, b.y2);
    int inter_w = std::max(0, x2 - x1);
    int inter_h = std::max(0, y2 - y1);
    int inter = inter_w * inter_h;
    int area_a = std::max(0, a.x2 - a.x1) * std::max(0, a.y2 - a.y1);
    int area_b = std::max(0, b.x2 - b.x1) * std::max(0, b.y2 - b.y1);
    int uni = area_a + area_b - inter;
    if (uni <= 0) return 0.0f;
    return static_cast<float>(inter) / static_cast<float>(uni);
}

void SimpleByteTracker::update(std::vector<InferItem>& detections) {
    // Greedy IoU matching: tracks -> detections
    std::vector<int> det_assigned(detections.size(), -1);

    for (auto& track : tracks_) {
        float best_iou = iou_threshold_;
        int best_det = -1;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (det_assigned[i] != -1) continue;
            float iou = IoU(track.state, detections[i]);
            if (iou > best_iou) {
                best_iou = iou;
                best_det = static_cast<int>(i);
            }
        }

        if (best_det >= 0) {
            // Match found
            track.state = detections[best_det];
            track.state.id = track.id;
            track.time_since_update = 0;
            track.hits += 1;
            det_assigned[best_det] = track.id;
        } else {
            // No match
            track.time_since_update += 1;
        }
    }

    // Create new tracks for unmatched detections
    for (size_t i = 0; i < detections.size(); ++i) {
        if (det_assigned[i] != -1) continue;
        Track t;
        t.state = detections[i];
        t.id = next_id_++;
        t.state.id = t.id;
        t.time_since_update = 0;
        t.hits = 1;
        tracks_.push_back(t);
    }

    // Remove stale tracks
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& t) {
        return t.time_since_update > max_age_ || t.hits < min_hits_;
    }), tracks_.end());

    // Write back ids to detections
    for (auto& det : detections) {
        // Find matching track by IoU with highest score
        int assigned = det.id; // may already set if matched earlier
        if (assigned != 0) continue;
        float best_iou = iou_threshold_;
        int best_id = 0;
        for (const auto& t : tracks_) {
            float iou = IoU(t.state, det);
            if (iou > best_iou) {
                best_iou = iou;
                best_id = t.id;
            }
        }
        if (best_id != 0) det.id = best_id;
    }
}

void SimpleByteTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}
