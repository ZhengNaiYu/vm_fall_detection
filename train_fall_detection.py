import os
import yaml
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.models import FallDetectionGRU, FallDetectionLSTM
from src.training import train_model, plot_training_history
from src.evaluation import evaluate_model


# -------------------------------------------------------
# Load Config
# -------------------------------------------------------
def load_config(path="./config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------
# Select Device
# -------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------
# Build Dynamic Paths Using Template
# -------------------------------------------------------
def build_paths(cfg):
    fps = cfg["fps"]
    cls = cfg["num_classes"]

    keypoints_path = os.path.join(
        cfg["root_dir"],
        cfg["keypoints_template"].format(cls=cls, fps=fps)
    )

    labels_path = os.path.join(
        cfg["root_dir"],
        cfg["labels_template"].format(cls=cls, fps=fps)
    )

    model_save_path = os.path.join(
        cfg["model_dir"],
        cfg["model_name_template"].format(cls=cls, fps=fps)
    )

    history_plot_path = os.path.join(
        cfg["results_dir"],
        cfg["history_plot_template"].format(cls=cls, fps=fps)
    )

    return keypoints_path, labels_path, model_save_path, history_plot_path


# -------------------------------------------------------
# Select and Build Model
# -------------------------------------------------------
def build_model(cfg):
    model_type = cfg["type"]
    output_size = cfg["num_classes"]

    params = dict(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout_prob=cfg["dropout_prob"],
        num_classes=output_size
    )

    if model_type == "LSTM":
        return FallDetectionLSTM(**params)
    elif model_type == "GRU":
        return FallDetectionGRU(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# -------------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------------
def main():
    # ----------------------------
    # Parse Arguments
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train_fall_detection"]
    device = get_device()
    print(f"Using device: {device}")

    # Build dynamic paths
    keypoints_path, labels_path, save_model_path, history_plot_path = build_paths(train_cfg)

    # Load dataset
    keypoints = np.load(keypoints_path)
    labels = np.load(labels_path)
    print(f"Loaded dataset: {keypoints_path}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        keypoints,
        labels,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"]
    )

    # Split val/test
    val_ratio = train_cfg["val_ratio"]
    val_size = int(len(X_test) * val_ratio)

    X_val, y_val = X_test[:val_size], y_test[:val_size]
    X_test, y_test = X_test[val_size:], y_test[val_size:]

    # Convert tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoaders
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Build model
    model = build_model(train_cfg).to(device)

    # Load checkpoint
    ckpt = train_cfg.get("checkpoint", None)
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt}")

    # Evaluate before training
    print("\n== Before Training Evaluation Start ==")
    evaluate_model(model, train_loader, num_classes=train_cfg["num_classes"])
    evaluate_model(model, val_loader, num_classes=train_cfg["num_classes"])
    evaluate_model(model, test_loader, num_classes=train_cfg["num_classes"])
    print("\n== Before Training Evaluation Finish ==")

    # Train model
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        num_classes=int(train_cfg["num_classes"]),
        num_epochs=int(train_cfg["num_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        patience=int(train_cfg["patience"]),
        device=device
    )

    # Save training curves
    plot_training_history(history, save_path=history_plot_path)

    # Save model
    torch.save(trained_model.state_dict(), save_model_path)
    print(f"Model saved → {save_model_path}")

    # Evaluate final model
    print("\n== Final Test Evaluation ==")
    evaluate_model(trained_model, test_loader, num_classes=train_cfg["num_classes"])


if __name__ == "__main__":
    main()
