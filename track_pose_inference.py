import argparse
import collections

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

from src.utils.pose_buffer import PoseBuffer
from src.models import FallDetectionLSTM, ImprovedLSTM, TransformerEncoder
from src.models import detect_fall_by_physics


# ---------------------------------------------------------
# Helpers for feature processing (与训练端完全一致)
# ---------------------------------------------------------
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


def to_relative_xy(kp_xy: np.ndarray) -> np.ndarray:
    """将 17x2 关键点转换为相对坐标（中心=髋或肩，尺度=肩/髋距离最大值）。
    kp_xy: (17, 2)
    Returns: (34,) flattened
    """
    pts = kp_xy.reshape(17, 2)
    ls, rs = pts[COCO_LEFT_SHOULDER], pts[COCO_RIGHT_SHOULDER]
    lh, rh = pts[COCO_LEFT_HIP], pts[COCO_RIGHT_HIP]

    # 中心
    center = np.array([0.0, 0.0])
    if not (np.isnan(lh).any() or np.isnan(rh).any()):
        center = (lh + rh) / 2.0
    elif not (np.isnan(ls).any() or np.isnan(rs).any()):
        center = (ls + rs) / 2.0
    else:
        valid = pts[~np.isnan(pts).any(axis=1)]
        center = valid.mean(axis=0) if len(valid) > 0 else np.array([0.0, 0.0])

    # 尺度
    def _dist(a, b):
        if np.isnan(a).any() or np.isnan(b).any():
            return np.nan
        return float(np.linalg.norm(a - b))

    d1 = _dist(ls, rs)
    d2 = _dist(lh, rh)
    scale = np.nanmax([d1, d2])
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    rel = (pts - center) / scale
    return rel.reshape(-1)  # 34


def build_features(seq_rel: np.ndarray) -> np.ndarray:
    """根据相对坐标序列构建特征：rel_xy + velocity."""
    T = seq_rel.shape[0]
    vel = np.zeros_like(seq_rel)
    vel[1:] = seq_rel[1:] - seq_rel[:-1]
    return np.concatenate([seq_rel, vel], axis=1)  # (T, 68)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_activity_model(cfg):
    infer_cfg = cfg["track_pose_inference"]
    model_type = infer_cfg.get("model_type", "LSTM")
    
    params = dict(
        input_size=infer_cfg["input_size"],
        hidden_size=infer_cfg["hidden_size"],
        num_layers=infer_cfg["num_layers"],
        num_classes=infer_cfg["num_classes"],
        dropout_prob=infer_cfg.get("dropout_prob", 0.5)
    )
    
    if model_type == "ImprovedLSTM":
        activity_model = ImprovedLSTM(**params)
    elif model_type == "Transformer":
        params['nhead'] = infer_cfg.get('nhead', 4)
        activity_model = TransformerEncoder(**params)
    else:  # 默认LSTM
        activity_model = FallDetectionLSTM(**params)

    ckpt_path = infer_cfg.get("lstm_model_path")
    if ckpt_path:
        if not torch.cuda.is_available():
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            state_dict = torch.load(ckpt_path)
        activity_model.load_state_dict(state_dict)
        print(f"加载模型 ({model_type}): {ckpt_path}")
    else:
        print("警告: 未指定模型路径")

    activity_model.eval()
    return activity_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (optional, overrides config).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_cfg = cfg["track_pose_inference"]

    # 支持命令行覆盖视频路径
    video_in = args.video if args.video else infer_cfg["video_in"]
    video_out = infer_cfg["video_out"]
    window = int(infer_cfg["window"])
    use_lstm = bool(infer_cfg["use_lstm"])
    physics_threshold = float(infer_cfg["physics_threshold"])
    physics_frames = int(infer_cfg["physics_frames"])
    center_y_threshold = float(infer_cfg.get("center_y_threshold", 0.7))
    tracker_config = infer_cfg["tracker_config"]
    conf = float(infer_cfg["conf"])
    yolo_model_path = infer_cfg["yolo_model_path"]
    num_classes = int(infer_cfg.get("num_classes", 9))
    sequence_length = int(infer_cfg.get("sequence_length", 10))
    class_names = infer_cfg.get("class_names", {0: "Unknown"})
    
    # 确保 class_names 是正确的字典格式
    if isinstance(class_names, dict):
        class_names = {int(k): v for k, v in class_names.items()}
    
    print(f"\n" + "=" * 70)
    print("姿态动作推理")
    print("=" * 70)
    print(f"输入视频: {video_in}")
    print(f"输出视频: {video_out}")
    print(f"窗口大小: {window}")
    print(f"模型类别: {num_classes}")
    print(f"使用 LSTM: {use_lstm}")
    print(f"=" * 70 + "\n")

    model = YOLO(yolo_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_buffer = PoseBuffer(window_size=window)
    activity_model = build_activity_model(cfg).to(device) if use_lstm else None

    box_history = {}  # pid -> deque of [x, y, w, h]

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: FPS={fps}, 宽={W}, 高={H}\n")

    writer = cv2.VideoWriter(
        video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    results = model.track(
        source=video_in,
        persist=True,
        tracker=tracker_config,
        conf=conf
    )

    frame_cnt = 0
    for r in results:
        frame = r.orig_img
        frame_cnt += 1
        print(f"处理帧 {frame_cnt}")

        if r.boxes.id is None:
            writer.write(frame)
            continue

        ids = r.boxes.id.cpu().numpy()
        boxes = r.boxes.xywh.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy()

        # 可选：过滤掉太小的检测框（远处的人或误检）
        min_box_area = infer_cfg.get("min_box_area", 0)  # 最小框面积（像素），0表示不过滤
        
        filtered_data = []
        for pid, box, kp in zip(ids, boxes, kps):
            x, y, w, h = box
            
            # 大小过滤：过滤掉太小的检测框
            if min_box_area > 0 and (w * h) < min_box_area:
                continue
            
            filtered_data.append((pid, box, kp))
        
        for pid, box, kp in filtered_data:
            x, y, w, h = box

            # 关键点相对坐标（与训练端一致）
            kp_rel_flat = to_relative_xy(kp)

            if pid not in pose_buffer.keypoints:
                pose_buffer.keypoints[pid] = collections.deque(maxlen=window)
            if pid not in box_history:
                box_history[pid] = collections.deque(maxlen=physics_frames)

            pose_buffer.keypoints[pid].append(kp_rel_flat)
            box_history[pid].append(np.array([x, y, w, h]))

            if use_lstm:
                if len(pose_buffer.keypoints[pid]) >= window:
                    seq_rel = np.array(list(pose_buffer.keypoints[pid]))  # (T,34)
                    seq_feat = build_features(seq_rel)  # (T,68)
                    # 归一化（与训练端一致）
                    max_abs = np.max(np.abs(seq_feat))
                    if np.isfinite(max_abs) and max_abs > 0:
                        seq_feat = seq_feat / max_abs
                    input_tensor = torch.tensor(seq_feat, dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        output = activity_model(input_tensor)
                        probs = F.softmax(output, dim=1)
                        pred = probs.argmax(dim=1).item()
                        prob = probs[0, pred].item()

                    action = class_names.get(pred, f"Unknown({pred})")
                    # 对于 9 分类，判断是否为跌倒（此处不涉及，可根据需要修改）
                    is_fall = False
                else:
                    action = "Pending"
                    is_fall = False
                    probs = None
            else:
                is_fall = detect_fall_by_physics(box_history[pid], physics_threshold, physics_frames, center_y_threshold, H)
                action = "Fall" if is_fall else "Normal"

            color = (0, 0, 255) if is_fall else (0, 255, 0)
            if use_lstm and len(pose_buffer.keypoints[pid]) >= window:
                label = f"ID {int(pid)} {action} {prob:.2f} Frame {frame_cnt}"
            else:
                label = f"ID {int(pid)} {action} Frame {frame_cnt}"

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # 在框左侧输出各类别概率
            if use_lstm and probs is not None:
                prob_vec = probs.squeeze(0).cpu().numpy()
                # 从上到下逐行绘制
                line_y = y1
                line_x = x1 - 90  # 左侧留出空间
                for cid in sorted(class_names.keys()):
                    pname = class_names.get(cid, f"C{cid}")
                    pval = prob_vec[cid] if cid < len(prob_vec) else 0.0
                    text = f"{pname}: {pval:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (line_x, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )
                    line_y += 14

        writer.write(frame)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n推理完成!")
    print(f"输出视频已保存: {video_out}\n")


if __name__ == "__main__":
    main()
