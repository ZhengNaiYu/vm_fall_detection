import collections
import numpy as np

class PoseBuffer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.keypoints = {}  # 存储展平后的关键点 (34,)

    def update(self, pid, keypoints, bbox):
        if pid not in self.pose:
            self.pose[pid] = collections.deque(maxlen=self.window_size)
            self.center_y[pid] = collections.deque(maxlen=self.window_size)
            self.angle[pid] = collections.deque(maxlen=self.window_size)

        self.pose[pid].append(keypoints)

        _, y, _, h = bbox
        self.center_y[pid].append(y + h / 2)
