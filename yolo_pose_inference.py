import yaml
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from sklearn.impute import KNNImputer

# from src.utils import extract_keypoints_from_video

# ---------------------------------------------------------
# Helpers for feature processing
# ---------------------------------------------------------
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


def to_relative_xy(xy_seq: np.ndarray) -> np.ndarray:
    """Convert xy to per-frame relative coords using hip/shoulder scale.
    xy_seq: (T, 17, 2)
    Returns: (T, 17, 2)
    """
    T = xy_seq.shape[0]
    rel = np.zeros_like(xy_seq)
    for t in range(T):
        pts = xy_seq[t]
        ls, rs = pts[COCO_LEFT_SHOULDER], pts[COCO_RIGHT_SHOULDER]
        lh, rh = pts[COCO_LEFT_HIP], pts[COCO_RIGHT_HIP]

        # center: midpoint of hips (fallback to shoulders or mean of valid)
        center = np.array([0.0, 0.0])
        if not (np.isnan(lh).any() or np.isnan(rh).any()):
            center = (lh + rh) / 2.0
        elif not (np.isnan(ls).any() or np.isnan(rs).any()):
            center = (ls + rs) / 2.0
        else:
            # mean of available joints
            valid = pts[~np.isnan(pts).any(axis=1)]
            center = valid.mean(axis=0) if len(valid) > 0 else np.array([0.0, 0.0])

        # scale: max of hip distance and shoulder distance
        def _dist(a, b):
            if np.isnan(a).any() or np.isnan(b).any():
                return np.nan
            return float(np.linalg.norm(a - b))

        d1 = _dist(ls, rs)
        d2 = _dist(lh, rh)
        scale = np.nanmax([d1, d2])
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        rel[t] = (pts - center) / scale
    return rel


def build_features(seq_rel: np.ndarray) -> np.ndarray:
    """根据相对坐标序列构建特征：rel_xy + velocity."""
    T = seq_rel.shape[0]
    vel = np.zeros_like(seq_rel)
    vel[1:] = seq_rel[1:] - seq_rel[:-1]
    return np.concatenate([seq_rel, vel], axis=1)  # (T, 68)


def compute_velocity(seq: np.ndarray) -> np.ndarray:
    """First-order temporal difference with zero padding at start.
    seq: (T, D)
    Returns: (T, D)
    """
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return vel


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_video_fps(video_path):
    """获取视频的帧率"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def extract_keypoints_from_action_videos(video_dir, output_dir, cfg):
    """
    从 action_videos 目录提取 pose 数据，自动检测帧率
    长视频会被分割成多个 sequence_length 的片段
    
    Args:
        video_dir: action_videos 目录
        output_dir: 输出目录
        cfg: 配置字典
    """
    video_path = Path(video_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载 YOLO 模型
    yolo_model_path = cfg["yolo_model_path"]
    print(f"加载 YOLO 模型: {yolo_model_path}")
    model = YOLO(yolo_model_path)
    
    # 获取所有类别目录
    class_dirs = sorted([d for d in video_path.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(f"错误: {video_dir} 中没有找到类别目录")
        return
    
    # 建立类别到标签的映射（优先使用配置的 class_names）
    class_names_cfg = cfg.get("class_names")
    class_to_label = {}

    if class_names_cfg:
        # 将配置中的映射转换为 name->id（不区分大小写）
        class_names_cfg = {int(k): v for k, v in class_names_cfg.items()}
        name_to_id = {v.lower(): k for k, v in class_names_cfg.items()}

        next_id = max(class_names_cfg.keys()) + 1 if class_names_cfg else 0
        for d in class_dirs:
            lbl = name_to_id.get(d.name.lower())
            if lbl is None:
                # 未在配置中找到的类，追加新的 id
                lbl = next_id
                next_id += 1
                print(f"警告: {d.name} 未在 class_names 中配置，自动分配标签 {lbl}")
            class_to_label[d.name] = lbl

        num_classes = max(class_to_label.values()) + 1
        # print(f'使用配置的 class_names，类别数: {num_classes}, 映射: {class_to_label}')
        # input("按回车键继续...")
    else:
        # 回退到按目录顺序分配
        class_to_label = {d.name: i for i, d in enumerate(class_dirs)}
        num_classes = len(class_to_label)
    
    print(f"\n找到 {num_classes} 个类别:")
    for class_name, label in sorted(class_to_label.items()):
        print(f"  {label}: {class_name}")
    
    # 自动检测帧率
    fps_values = set()
    for class_dir in class_dirs:
        videos = list(class_dir.glob("*.mp4"))
        if videos:
            fps = get_video_fps(str(videos[0]))
            fps_values.add(fps)
    
    print(f"\n检测到的帧率: {sorted(fps_values)}")
    
    if len(fps_values) > 1:
        print(f"警告: 检测到多个帧率，将使用 {max(fps_values)} fps")
        fps = max(fps_values)
    else:
        fps = list(fps_values)[0] if fps_values else 30
    
    fps_int = int(fps)
    print(f"使用帧率: {fps_int} fps")
    
    # 统计视频数量
    total_videos = sum(len(list(d.glob("*.mp4"))) for d in class_dirs)
    print(f"\n找到 {total_videos} 个视频")
    
    # 提取参数
    sequence_length = cfg.get("sequence_length", 30)
    n_neighbors = cfg.get("n_neighbors", 5)
    normalize = cfg.get("normalize", True)
    overlap_ratio = float(cfg.get("overlap_ratio", 0.5))
    overlap_ratio = max(0.0, min(0.9, overlap_ratio))
    stride = max(1, int(round(sequence_length * (1.0 - overlap_ratio))))

    # 提取关键点
    print(f"\n开始提取 pose 数据 (sequence_length={sequence_length})...")
    print("注意: 长视频将被分割成多个片段")
    keypoints_sequences = []
    labels = []
    segment_stats = {}  # 统计每个视频分割成的片段数
    
    pbar = tqdm(total=total_videos, desc="处理进度")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_label = class_to_label[class_name]
        
        videos = sorted(class_dir.glob("*.mp4"))
        
        for video_path_item in videos:
            try:
                # 提取视频的所有关键点（不限制长度）
                cap = cv2.VideoCapture(str(video_path_item))
                all_keypoints = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 使用 YOLO 检测姿态
                    results = model(frame, verbose=False)
                    
                    if results and len(results) > 0:
                        keypoints = results[0].keypoints
                        
                        if keypoints is not None and len(keypoints.xy) > 0:
                            # 提取第一个检测到的人的关键点 (xy 17x2) 与置信度 (17)
                            kpts_xy = keypoints.xy[0].cpu().numpy().reshape(-1, 2)
                            frame_vec = kpts_xy.flatten()
                            all_keypoints.append(frame_vec)
                        else:
                            all_keypoints.append(np.zeros(34))
                    else:
                        all_keypoints.append(np.zeros(34))
                
                cap.release()
                
                # 将关键点序列分割成 sequence_length 的片段
                if len(all_keypoints) > 0:
                    all_keypoints_array = np.array(all_keypoints)
                    video_length = len(all_keypoints_array)
                    # print(f"处理视频: {video_path_item.name}, 帧数: {video_length}, shape: {all_keypoints_array.shape}")
                    # 统计与分段（带重叠滑窗）
                    if video_length < sequence_length:
                        # 计算需要重复多少次才能达到目标长度, 重复整个序列并截取到目标长度
                        repeats_needed = int(np.ceil(sequence_length / video_length))
                        repeated_keypoints = np.tile(all_keypoints_array, (repeats_needed, 1))
                        keypoints = repeated_keypoints[:sequence_length]
                        keypoints_sequences.append(keypoints)
                        labels.append(class_label)
                        segment_stats[str(video_path_item.name)] = 1
                        # print(f"视频过短重复序列以达到目标长度: 原始帧数 {video_length} -> 重复后帧数 {repeated_keypoints.shape[0]}, shape: {repeated_keypoints.shape}")
                    else:
                        # 使用滑动窗口 + 重叠
                        starts = list(range(0, video_length - sequence_length + 1, stride))
                        if len(starts) == 0:
                            starts = [0]
                        last_start = video_length - sequence_length
                        if starts[-1] != last_start:
                            starts.append(last_start)

                        for start_idx in starts:
                            end_idx = start_idx + sequence_length
                            segment = all_keypoints_array[start_idx:end_idx]
                            keypoints_sequences.append(segment)
                            labels.append(class_label)
                        segment_stats[str(video_path_item.name)] = len(starts)
                
            except Exception as e:
                print(f"\n警告: 处理 {video_path_item} 时出错: {e}")
            
            pbar.update(1)
    
    pbar.close()
    
    keypoints_sequences = np.array(keypoints_sequences)
    labels = np.array(labels)
    
    print(f"\n提取完成!")
    print(f"  原始视频数: {total_videos}")
    print(f"  生成的片段数: {len(keypoints_sequences)}") # 关键点形状: (1382, 15, 51)
    print(f"  关键点形状: {keypoints_sequences.shape}")
    print(f"  标签形状: {labels.shape}")
    
    # 显示分割统计
    if segment_stats:
        max_segments = max(segment_stats.values())
        multi_segment_videos = sum(1 for v in segment_stats.values() if v > 1)
        if multi_segment_videos > 0:
            print(f"  分割成多个片段的视频: {multi_segment_videos} 个")
            print(f"  最多分割片段数: {max_segments}")
    
    # 开始特征处理流水线
    print("\n开始特征处理（插值/平滑/相对坐标/速度）...")
    sequences_processed = []
    for seq in keypoints_sequences:
        # seq: (T, 34)
        T = seq.shape[0]
        xy = seq[:, :34]
        # 将 0 视为缺失
        xy = xy.astype(np.float32)
        xy[xy == 0] = np.nan
        # KNN Imputation
        imp = KNNImputer(n_neighbors=n_neighbors)
        xy_imp = imp.fit_transform(xy)
        # 转换为相对坐标（关键修复！）
        xy_frames = xy_imp.reshape(T, 17, 2)
        xy_rel_frames = to_relative_xy(xy_frames)
        xy_rel = xy_rel_frames.reshape(T, 34)
        # 构建特征：rel_xy + velocity
        seq_out = build_features(xy_rel)
        sequences_processed.append(seq_out)

    keypoints_sequences_processed = np.asarray(sequences_processed, dtype=np.float32)

    # 归一化（对坐标与速度部分做最大绝对值归一化）
    if normalize:
        print("归一化...")
        max_abs = np.max(np.abs(keypoints_sequences_processed))
        if np.isfinite(max_abs) and max_abs > 0:
            keypoints_sequences_processed = keypoints_sequences_processed / max_abs
        keypoints_sequences_processed = np.nan_to_num(keypoints_sequences_processed)
    else:
        keypoints_sequences_processed = np.nan_to_num(keypoints_sequences_processed)
    
    # 保存数据
    print("\n保存数据...")
    
    # 根据帧率和类别数生成文件名
    save_keypoints_template = cfg.get("save_keypoints", "./data/processed/keypoints_sequences_cls{cls}_fps{fps}.npy")
    save_labels_template = cfg.get("save_labels", "./data/processed/labels_cls{cls}_fps{fps}.npy")
    
    keypoints_file = save_keypoints_template.format(cls=num_classes, fps=fps_int)
    labels_file = save_labels_template.format(cls=num_classes, fps=fps_int)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(keypoints_file) if os.path.dirname(keypoints_file) else ".", exist_ok=True)
    
    np.save(keypoints_file, keypoints_sequences_processed)
    np.save(labels_file, labels)
    
    print(f"  关键点已保存: {keypoints_file}")
    print(f"  标签已保存: {labels_file}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  类别数: {num_classes}")
    print(f"  帧率: {fps_int} fps")
    print(f"  样本数: {len(keypoints_sequences_processed)}")
    print(f"  序列长度: {sequence_length}")
    print(f"  特征维度: {keypoints_sequences_processed.shape[-1]}")
    
    # 打印每个类别的样本数
    print(f"\n各类别样本数:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        class_name = [k for k, v in class_to_label.items() if v == label][0]
        print(f"  {class_name}: {count}")


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def main():
    """Extract pose keypoints from organized video directories."""
    # ---------------------------
    # Parse command-line argument
    # ---------------------------
    parser = argparse.ArgumentParser(
        description="Extract pose keypoints from videos using YOLO"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config file")
    args = parser.parse_args()

    # ---------------------------
    # Load YAML config
    # ---------------------------
    config = load_config(args.config)
    exp_cfg = config["yolo_pose_inference"]
    
    print("=" * 70)
    print("YOLO Pose Keypoints Extraction")
    print("=" * 70)
    
    # Extract from organized video directories
    video_dir = exp_cfg["video_dir"]
    output_dir = exp_cfg.get("output_dir", "./data/processed")
    
    print(f"视频目录: {video_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    
    extract_keypoints_from_action_videos(video_dir, output_dir, exp_cfg)


if __name__ == "__main__":
    main()