"""可视化关键点序列，生成骨骼视频
- 输入 keypoints 和 labels 的 .npy 文件
- 自动读取 config.yaml 中的 class_names（yolo_pose_inference_action_videos）
- 长度假设为 (N, T, 34) 对应 17 个关键点 (x, y)
"""
import argparse
import os
from pathlib import Path
import yaml
import cv2
import numpy as np

# COCO 17 关键点骨架连接
COCO_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (0, 14), (0, 15), (14, 16), (15, 17)  # 14,15,16,17 不存在于17点，留作扩展
]

# 实际 17 点，上面 (14,15,16,17) 不使用，过滤掉越界连接
COCO_EDGES = [(i, j) for (i, j) in COCO_EDGES if i < 17 and j < 17]


def load_class_names(config_path: str):
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("yolo_pose_inference_action_videos", {}).get("class_names", {})
        return {int(k): v for k, v in names.items()}
    except Exception:
        return {}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def draw_skeleton(frame, kpts, color_edges=(255, 0, 0), color_points=(0, 255, 0)):
    """在 frame 上绘制单帧骨骼"""
    for i, j in COCO_EDGES:
        if kpts[i] is None or kpts[j] is None:
            continue
        cv2.line(frame, kpts[i], kpts[j], color_edges, 2, cv2.LINE_AA)
    for p in kpts:
        if p is None:
            continue
        cv2.circle(frame, p, 3, color_points, -1, cv2.LINE_AA)


def normalize_keypoints(seq: np.ndarray, canvas_w=640, canvas_h=480):
    """将关键点归一化到画布大小；返回像素坐标列表列表"""
    xs = seq[:, 0::2]
    ys = seq[:, 1::2]
    valid = (xs != 0) | (ys != 0)
    if not valid.any():
        return [[None] * 17 for _ in range(seq.shape[0])], canvas_w, canvas_h

    x_min, x_max = xs[valid].min(), xs[valid].max()
    y_min, y_max = ys[valid].min(), ys[valid].max()
    # 避免除零
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    norm_frames = []
    for t in range(seq.shape[0]):
        kpts = []
        for i in range(17):
            x = seq[t, 2 * i]
            y = seq[t, 2 * i + 1]
            if x == 0 and y == 0:
                kpts.append(None)
                continue
            xn = int((x - x_min) / (x_max - x_min) * (canvas_w * 0.8) + canvas_w * 0.1)
            yn = int((y - y_min) / (y_max - y_min) * (canvas_h * 0.8) + canvas_h * 0.1)
            kpts.append((xn, yn))
        norm_frames.append(kpts)
    return norm_frames, canvas_w, canvas_h


def visualize_sequences(keypoints_path: str, labels_path: str, out_dir: str, config_path: str,
                        fps: int = 10, max_sequences: int = 50):
    keypoints = np.load(keypoints_path)
    # input(keypoints.shape)
    labels = np.load(labels_path)

    assert keypoints.shape[0] == labels.shape[0], "keypoints 与 labels 数量不一致"
    assert keypoints.shape[2] == 34, "关键点维度应为 34 (17*2)"

    class_names = load_class_names(config_path)

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    total = min(max_sequences, keypoints.shape[0])
    print(f"可视化前 {total} 个序列 / 共 {keypoints.shape[0]} 个")

    for idx in range(total):
        seq = keypoints[idx]  # (T, 34)
        label_id = int(labels[idx]) if idx < len(labels) else -1
        label_name = class_names.get(label_id, f"class_{label_id}")

        norm_frames, W, H = normalize_keypoints(seq)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = out_dir / f"seq_{idx:04d}_{label_name}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

        for kpts in norm_frames:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            draw_skeleton(frame, kpts)
            cv2.putText(frame, label_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(frame)

        writer.release()
        print(f"保存: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化关键点序列，生成骨骼视频")
    parser.add_argument("--keypoints", type=str, default="./data/processed/keypoints_sequences_cls9_fps10.npy")
    parser.add_argument("--labels", type=str, default="./data/processed/labels_cls9_fps10.npy")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--out_dir", type=str, default="./data/processed/vis_cls9_fps10")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_sequences", type=int, default=50, help="最多导出多少个序列视频")
    args = parser.parse_args()

    visualize_sequences(
        keypoints_path=args.keypoints,
        labels_path=args.labels,
        out_dir=args.out_dir,
        config_path=args.config,
        fps=args.fps,
        max_sequences=args.max_sequences,
    )
