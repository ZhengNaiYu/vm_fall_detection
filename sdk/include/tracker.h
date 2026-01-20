// Simple ByteTrack-style tracker (greedy IoU assignment)
#ifndef SIMPLE_BYTE_TRACKER_H
#define SIMPLE_BYTE_TRACKER_H

#include "vmsdk.h"
#include <vector>

class SimpleByteTracker {
public:
    struct Track {
        visionmatrixsdk::falldetection::InferItem state;
        int id;
        int time_since_update;
        int hits;
    };

    SimpleByteTracker(float iou_threshold = 0.3f, int max_age = 30, int min_hits = 1)
        : iou_threshold_(iou_threshold), max_age_(max_age), min_hits_(min_hits), next_id_(1) {}

    // Assign track ids to detections in-place.
    void update(std::vector<visionmatrixsdk::falldetection::InferItem>& detections);

    void reset();

private:
    float IoU(const visionmatrixsdk::falldetection::InferItem& a,
              const visionmatrixsdk::falldetection::InferItem& b) const;

    std::vector<Track> tracks_;
    float iou_threshold_;
    int max_age_;
    int min_hits_;
    int next_id_;
};

#endif // SIMPLE_BYTE_TRACKER_H