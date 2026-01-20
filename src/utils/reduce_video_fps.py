import os
import cv2
import numpy as np

def reduce_video_fps(input_folder, output_folder, target_fps=5, overwrite=False):
    """
    Reduce FPS of all videos in a folder and save to output folder.
    
    Args:
        input_folder (str): Path to input video folder
        output_folder (str): Path to save processed videos
        target_fps (int): Desired FPS (e.g., 5 or 10)
        overwrite (bool): Whether to overwrite existing files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for filename in video_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {filename}, already exists.")
            continue
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open {filename}")
            continue
        
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30  # fallback
        frame_interval = max(int(round(orig_fps / target_fps)), 1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"Saved {filename} at {target_fps} FPS")