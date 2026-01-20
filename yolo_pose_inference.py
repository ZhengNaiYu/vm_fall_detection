import yaml
import os
import numpy as np
import argparse
from ultralytics import YOLO
import supervision as sv
from sklearn.impute import KNNImputer

from src.utils import extract_keypoints_from_video


# ---------------------------------------------------------
# Load Config
# ---------------------------------------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def main():
    # ---------------------------
    # Parse command-line argument
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config file")
    args = parser.parse_args()

    # ---------------------------
    # Load YAML config
    # ---------------------------
    config = load_config(args.config)
    exp_cfg = config["yolo_pose_inference"]

    YOLO_MODEL_PATH = exp_cfg["yolo_model_path"]
    SEQUENCE_LENGTH = exp_cfg["sequence_length"]
    N_NEIGHBORS = exp_cfg["n_neighbors"]
    NORMALIZE = exp_cfg["normalize"]
    VIDEO_FOLDERS = exp_cfg["video_folders"]

    SAVE_KEYPOINTS = exp_cfg["save_keypoints"].format(
        cls=exp_cfg["num_classes"], fps=exp_cfg["fps"]
    )
    SAVE_LABELS = exp_cfg["save_labels"].format(
        cls=exp_cfg["num_classes"], fps=exp_cfg["fps"]
    )

    # ---------------------------------------------------------
    # 2. Load YOLO pose model + tracker
    # ---------------------------------------------------------
    model = YOLO(YOLO_MODEL_PATH)
    byte_tracker = sv.ByteTrack()

    # ---------------------------------------------------------
    # 3. Extract keypoints
    # ---------------------------------------------------------
    keypoints_sequences = []
    labels = []

    fps = exp_cfg["fps"]
    for base_folder, label in VIDEO_FOLDERS.items():
        folder = f"{base_folder}_fps{fps}"
        for filename in os.listdir(folder):
            if filename.endswith(".mp4"):
                video_path = os.path.join(folder, filename)

                keypoints = extract_keypoints_from_video(
                    video_path, model, sequence_length=SEQUENCE_LENGTH
                )
                keypoints_sequences.append(keypoints)
                labels.append(label)

    keypoints_sequences = np.array(keypoints_sequences)
    labels = np.array(labels)

    # ---------------------------------------------------------
    # 4. Replace 0 with NaN (missing keypoints)
    # ---------------------------------------------------------
    keypoints_sequences[keypoints_sequences == 0] = np.nan

    # ---------------------------------------------------------
    # 5. KNN Imputation
    # ---------------------------------------------------------
    imputer = KNNImputer(n_neighbors=N_NEIGHBORS)

    # reshape: (samples * seq_len, 34)
    flat = keypoints_sequences.reshape(-1, keypoints_sequences.shape[-1])

    flat_imputed = imputer.fit_transform(flat)

    keypoints_sequences_imputed = flat_imputed.reshape(
        keypoints_sequences.shape[0],
        keypoints_sequences.shape[1],
        keypoints_sequences.shape[2]
    )

    # ---------------------------------------------------------
    # 6. Normalize if enabled
    # ---------------------------------------------------------
    if NORMALIZE:
        max_val = np.nanmax(keypoints_sequences_imputed)
        keypoints_sequences_normalized = keypoints_sequences_imputed / max_val
        keypoints_sequences_normalized = np.nan_to_num(keypoints_sequences_normalized)
    else:
        keypoints_sequences_normalized = np.nan_to_num(keypoints_sequences_imputed)

    # ---------------------------------------------------------
    # 7. Save output
    # ---------------------------------------------------------
    np.save(SAVE_KEYPOINTS, keypoints_sequences_normalized)
    np.save(SAVE_LABELS, labels)

    print(f"Saved keypoints: {SAVE_KEYPOINTS}")
    print(f"Saved labels: {SAVE_LABELS}")


if __name__ == "__main__":
    main()