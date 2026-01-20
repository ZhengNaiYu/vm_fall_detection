import argparse
import yaml
import torch
import os
from ultralytics import YOLO

from src.models import FallDetectionLSTM, FallDetectionGRU


# ----------------------------------------
# Load config
# ----------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------
# Build model from config
# ----------------------------------------
def build_model(export_cfg):
    params = dict(
        input_size=export_cfg["input_size"],
        hidden_size=export_cfg["hidden_size"],
        num_layers=export_cfg["num_layers"],
        dropout_prob=export_cfg["dropout_prob"],
        num_classes=export_cfg["num_classes"],
    )

    if export_cfg["type"] == "LSTM":
        model = FallDetectionLSTM(**params)
    elif export_cfg["type"] == "GRU":
        model = FallDetectionGRU(**params)
    else:
        raise ValueError("Unknown model type: ", export_cfg["type"])

    return model


# ----------------------------------------
# Export ONNX
# ----------------------------------------
def export_onnx():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to trained .pth file (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                        help="output ONNX file path (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_cfg = cfg["export_onnx"]
    
    # Use command-line args if provided, otherwise use config
    checkpoint = args.checkpoint if args.checkpoint else export_cfg["checkpoint"]
    output = args.output if args.output else export_cfg["output"]
    
    model = build_model(export_cfg)

    # Load weights
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # Dummy input: (batch, seq_len, features)
    seq_len = export_cfg["sequence_length"]
    dummy = torch.randn(1, seq_len, export_cfg["input_size"])

    # Export ONNX
    torch.onnx.export(
        model,
        dummy,
        output,
        export_params=True,
        opset_version=export_cfg["opset_version"],
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch"},
        }
    )

    print(f"ONNX model exported → {output}")


def convert_yolo_to_onnx():
    """Convert YOLO model to ONNX format."""
    from ultralytics import YOLO
    
    print("Converting YOLO model to ONNX...")
    
    # Load YOLO model
    model = YOLO('models/yolo11n-pose.pt')
    
    # Export to ONNX
    model.export(format='onnx', opset=11, simplify=True)
    
    print("YOLO model exported to: models/yolo11n-pose.onnx")


if __name__ == "__main__":
    # model = YOLO("models/yolo11x-pose.pt")
    # model.export(
    #     format="onnx",
    #     opset=17,          # 推荐 16 或 17（兼容度最强）
    #     simplify=True,     # 自动用 onnxsim 简化
    #     dynamic=True       # 支持动态输入尺寸
    # )

    # export_onnx()

    # Example usage:
    # python export_onnx.py \
    # --config config.yaml \
    # --checkpoint models/lstm_3cls_fps5.pth \
    # --output models/lstm_3cls_fps5.onnx

    convert_yolo_to_onnx()