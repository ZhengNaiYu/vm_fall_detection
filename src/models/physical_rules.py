import numpy as np


def detect_fall_by_physics(box_history, PHYSICS_THRESHOLD=0.2, PHYSICS_FRAMES=10):
    """
    物理规则: 基于框高度变化检测跌倒
    如果最近的高度相比平均历史高度下降超过阈值，则判定跌倒
    box_history: deque of [x, y, w, h]
    """
    if len(box_history) < PHYSICS_FRAMES:
        return False
    
    recent_boxes = list(box_history)[-PHYSICS_FRAMES:]
    recent_heights = [box[3] for box in recent_boxes]
    
    current_height = recent_heights[-1]
    prev_avg_height = np.mean(recent_heights[:-1]) if len(recent_heights) > 1 else recent_heights[0]
    
    # 计算高度下降比例
    height_drop_ratio = (prev_avg_height - current_height) / prev_avg_height if prev_avg_height > 0 else 0
    
    return height_drop_ratio > PHYSICS_THRESHOLD


def body_tilt_angle(kp):
    """
    返回角度（degree）
    """
    l_sh, r_sh = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]

    shoulder = (l_sh + r_sh) / 2
    hip = (l_hip + r_hip) / 2

    vec = shoulder - hip
    angle = np.degrees(np.arctan2(abs(vec[0]), abs(vec[1])))
    return angle


def is_fall(center_y_seq, angle_seq,
            angle_th=45,
            drop_th=0.15,
            min_frames=8):
    """
    center_y_seq: list of normalized center y
    angle_seq: list of angle
    """
    if len(center_y_seq) < min_frames:
        return False

    # 1️⃣ 姿态角度
    angle_flag = angle_seq[-1] > angle_th

    # 2️⃣ 重心下降速度
    drop = center_y_seq[-1] - center_y_seq[-min_frames]
    drop_flag = drop > drop_th

    return angle_flag and drop_flag
