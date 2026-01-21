import argparse
import collections

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.utils.pose_buffer import PoseBuffer
from src.models import FallDetectionLSTM
from src.models import detect_fall_by_physics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_fall_model(cfg):
    infer_cfg = cfg["track_fall_inference"]

    fall_model = FallDetectionLSTM(
        input_size=infer_cfg["input_size"],
        hidden_size=infer_cfg["hidden_size"],
        num_layers=infer_cfg["num_layers"],
        num_classes=infer_cfg["num_classes"]
    )

    ckpt_path = infer_cfg.get("lstm_model_path")
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        fall_model.load_state_dict(state_dict)

    fall_model.eval()
    return fall_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_cfg = cfg["track_fall_inference"]

    video_in = infer_cfg["video_in"]
    video_out = infer_cfg["video_out"]
    window = int(infer_cfg["window"])
    use_lstm = bool(infer_cfg["use_lstm"])
    physics_threshold = float(infer_cfg["physics_threshold"])
    physics_frames = int(infer_cfg["physics_frames"])
    tracker_config = infer_cfg["tracker_config"]
    conf = float(infer_cfg["conf"])
    yolo_model_path = infer_cfg["yolo_model_path"]

    model = YOLO(yolo_model_path)
    pose_buffer = PoseBuffer(window_size=window)
    fall_model = build_fall_model(cfg) if use_lstm else None

    box_history = {}  # pid -> deque of [x, y, w, h]

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        print(f"Processing frame {frame_cnt}")

        if r.boxes.id is None:
            writer.write(frame)
            continue

        ids = r.boxes.id.cpu().numpy()
        boxes = r.boxes.xywh.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy()

        for pid, box, kp in zip(ids, boxes, kps):
            x, y, w, h = box

            kp_flat = kp.flatten()

            if pid not in pose_buffer.keypoints:
                pose_buffer.keypoints[pid] = collections.deque(maxlen=window)
            if pid not in box_history:
                box_history[pid] = collections.deque(maxlen=physics_frames)

            pose_buffer.keypoints[pid].append(kp_flat)
            box_history[pid].append(np.array([x, y, w, h]))

            if use_lstm:
                if len(pose_buffer.keypoints[pid]) >= window:
                    seq = np.array(list(pose_buffer.keypoints[pid]))
                    input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        output = fall_model(input_tensor)
                        pred = output.argmax(dim=1).item()

                    fall = (pred == 0)
                else:
                    fall = False
            else:
                fall = detect_fall_by_physics(box_history[pid], physics_threshold, physics_frames)

            color = (0, 0, 255) if fall else (0, 255, 0)
            label = f"ID {int(pid)} FALL Frame {frame_cnt}" if fall else f"ID {int(pid)} Frame {frame_cnt}"

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

        writer.write(frame)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
