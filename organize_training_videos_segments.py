"""
根据 valid_images.txt 中指定的每个帧范围片段，生成独立的训练视频
每个帧范围都保存为一个单独的视频文件
"""
import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm


def parse_valid_images(valid_images_path):
    """
    解析 valid_images.txt 文件
    
    返回格式: [
        {
            'folder': 'jump_03-02-12-34-01-795',
            'class': 'jump',
            'start': 52,
            'end': 59,
            'segment_id': 1
        },
        ...
    ]
    """
    segments = []
    
    with open(valid_images_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    i = 0
    while i < len(lines):
        # 第一行是文件夹名
        folder_name = lines[i]
        
        # 提取类别名（文件夹名格式: class_timestamp）
        class_name = folder_name.split('_')[0]
        
        segment_id = 1
        i += 1
        
        # 读取所有帧范围
        while i < len(lines):
            parts = lines[i].split()
            
            # 如果不是两个数字，说明是下一个文件夹名
            if len(parts) != 2:
                break
            
            try:
                start_idx = int(parts[0])
                end_idx = int(parts[1])
                
                segments.append({
                    'folder': folder_name,
                    'class': class_name,
                    'start': start_idx,
                    'end': end_idx,
                    'segment_id': segment_id
                })
                
                segment_id += 1
                i += 1
            except ValueError:
                # 不是数字，说明是下一个文件夹名
                break
    
    return segments


def create_video_from_segment(source_folder, start_idx, end_idx, output_path, 
                              image_format="{:05d}.jpg", fps=30):
    """
    根据指定的帧范围创建视频
    
    Args:
        source_folder: 源图像文件夹路径
        start_idx: 起始帧索引
        end_idx: 结束帧索引
        output_path: 输出视频路径
        image_format: 图像文件名格式
        fps: 视频帧率
    """
    # 收集所有需要的图像路径
    image_paths = []
    for frame_idx in range(start_idx, end_idx + 1):
        image_name = image_format.format(frame_idx)
        image_path = os.path.join(source_folder, image_name)
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"警告: 图像不存在 {image_path}")
    
    if not image_paths:
        print(f"错误: 没有找到有效图像 (帧 {start_idx}-{end_idx})")
        return False
    
    # 读取第一张图像以获取尺寸
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        print(f"错误: 无法读取图像 {image_paths[0]}")
        return False
    
    height, width, _ = first_frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 写入所有图像
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)
    
    video.release()
    return True


def organize_training_videos_segments(source_dir, valid_images_file, output_base_dir, 
                                      image_format="{:05d}.jpg", fps=30):
    """
    根据 valid_images.txt 生成训练视频，每个帧范围片段保存为单独的视频
    
    Args:
        source_dir: source_images3 目录路径
        valid_images_file: valid_images.txt 文件路径
        output_base_dir: 输出基础目录
        image_format: 图像文件名格式
        fps: 视频帧率
    """
    source_path = Path(source_dir)
    output_path = Path(output_base_dir)
    
    # 解析 valid_images.txt
    print("解析 valid_images.txt...")
    segments = parse_valid_images(valid_images_file)
    
    print(f"找到 {len(segments)} 个视频片段")
    
    # 统计类别
    class_counts = {}
    for segment in segments:
        class_name = segment['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n各类别片段统计:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  - {class_name}: {count} 个片段")
    
    # 创建输出目录
    for class_name in class_counts.keys():
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个视频片段
    print(f"\n开始生成训练视频片段 (fps={fps})...")
    success_count = 0
    fail_count = 0
    
    # 用于给每个类别的视频编号
    class_video_counter = {}
    
    for segment in tqdm(segments, desc="处理进度"):
        folder_name = segment['folder']
        class_name = segment['class']
        start_idx = segment['start']
        end_idx = segment['end']
        
        # 源图像文件夹
        source_folder = source_path / folder_name
        
        if not source_folder.exists():
            print(f"\n警告: 文件夹不存在 {source_folder}")
            fail_count += 1
            continue
        
        # 为每个类别的视频编号
        if class_name not in class_video_counter:
            class_video_counter[class_name] = 1
        else:
            class_video_counter[class_name] += 1
        
        # 生成视频文件名：class_编号_folder_start-end.mp4
        timestamp = folder_name.split('_', 1)[1] if '_' in folder_name else folder_name
        video_name = f"{class_name}_{class_video_counter[class_name]:04d}_{timestamp}_{start_idx:05d}-{end_idx:05d}.mp4"
        output_video_path = output_path / class_name / video_name
        
        # 创建视频
        if create_video_from_segment(str(source_folder), start_idx, end_idx,
                                    str(output_video_path), image_format, fps):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n处理完成!")
    print(f"成功: {success_count} 个视频片段")
    print(f"失败: {fail_count} 个视频片段")
    print(f"\n输出目录: {output_path}")
    
    # 显示每个类别的视频数量
    print(f"\n各类别视频统计:")
    for class_name in sorted(class_counts.keys()):
        class_dir = output_path / class_name
        videos = list(class_dir.glob("*.mp4"))
        
        # 计算总帧数
        total_frames = 0
        for segment in segments:
            if segment['class'] == class_name:
                total_frames += (segment['end'] - segment['start'] + 1)
        
        print(f"  - {class_name}: {len(videos)} 个视频片段, 总计 {total_frames} 帧")


def main():
    # 配置路径
    source_dir = "/mnt/2/leo/fall_detection/data/source_images3"
    valid_images_file = "/mnt/2/leo/fall_detection/data/source_images3/valid_images.txt"
    output_base_dir = "/mnt/2/leo/fall_detection/data/training_videos_segments"
    
    # 图像格式
    image_format = "{:05d}.jpg"
    
    # 设置视频帧率（降低帧率以避免快进效果）
    fps = 10
    
    print(f"源目录: {source_dir}")
    print(f"配置文件: {valid_images_file}")
    print(f"输出目录: {output_base_dir}")
    print(f"图像格式: {image_format}")
    print(f"视频帧率: {fps} fps\n")
    
    # 执行转换和整理
    organize_training_videos_segments(source_dir, valid_images_file, output_base_dir, 
                                     image_format, fps)


if __name__ == "__main__":
    main()
