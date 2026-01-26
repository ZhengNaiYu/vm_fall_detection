import argparse
import yaml
import torch
import os
from ultralytics import YOLO

from src.models import LSTM, GRU, BiLSTMAttention, Transformer


# ----------------------------------------
# Load config
# ----------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------
# Build classifier model from config
# ----------------------------------------
def build_classifier_model(export_cfg):
    """Build a classification model (LSTM, GRU, BiLSTMAttention, or Transformer)."""
    params = dict(
        input_size=export_cfg["input_size"],
        hidden_size=export_cfg["hidden_size"],
        num_layers=export_cfg["num_layers"],
        dropout_prob=export_cfg.get("dropout_prob", 0.5),
        num_classes=export_cfg["num_classes"],
    )

    model_type = export_cfg.get("type", "LSTM")
    
    if model_type == "LSTM":
        model = LSTM(**params)
    elif model_type == "GRU":
        model = GRU(**params)
    elif model_type == "BiLSTMAttention" or model_type == "ImprovedLSTM":
        # Support both new and old names
        model = BiLSTMAttention(**params)
    elif model_type == "Transformer":
        params['nhead'] = export_cfg.get("nhead", 4)
        model = Transformer(**params)
    else:
        raise ValueError(f"Unknown classifier model type: {model_type}")

    return model


# ----------------------------------------
# Export classifier model to ONNX
# ----------------------------------------
def export_classifier_onnx(export_cfg, checkpoint, output):
    """Export trained classifier model to ONNX format."""
    print(f"\n{'='*70}")
    print("导出分类模型到 ONNX")
    print(f"{'='*70}")
    
    model = build_classifier_model(export_cfg)

    # Load weights
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print(f"✓ 加载模型: {checkpoint}")

    # Dummy input: (batch, seq_len, features)
    seq_len = export_cfg.get("sequence_length", 15)
    dummy = torch.randn(1, seq_len, export_cfg["input_size"])

    # Export ONNX
    torch.onnx.export(
        model,
        dummy,
        output,
        export_params=True,
        opset_version=export_cfg.get("opset_version", 17),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch"},
        }
    )

    print(f"✓ ONNX 模型已导出 → {output}")
    print(f"{'='*70}\n")


# ----------------------------------------
# Export YOLO pose model to ONNX
# ----------------------------------------
def export_yolo_onnx(yolo_cfg):
    """Export YOLO pose estimation model to ONNX format."""
    print(f"\n{'='*70}")
    print("导出 YOLO 姿态检测模型到 ONNX")
    print(f"{'='*70}")
    
    model_path = yolo_cfg.get("yolo_model_path", "models/yolo11n-pose.pt")
    output_path = yolo_cfg.get("yolo_output_onnx", "models/yolo11n-pose.onnx")
    opset_version = yolo_cfg.get("opset_version", 17)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    
    # Load YOLO model
    model = YOLO(model_path)
    print(f"✓ 加载 YOLO 模型: {model_path}")
    
    # Export to ONNX
    model.export(
        format='onnx',
        opset=opset_version,
        simplify=True,
        dynamic=True
    )
    
    print(f"✓ ONNX 模型已导出 → {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["classifier", "yolo", "all"],
        default="classifier",
        help="Which model to export: classifier (分类模型), yolo (YOLO检测模型), or all (两个都导出)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained classifier .pth file (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path for classifier (overrides config)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # 导出分类模型
    if args.model in ["classifier", "all"]:
        export_cfg = cfg.get("export_onnx", {})
        if not export_cfg:
            print("❌ 错误: 配置文件中未找到 export_onnx 配置")
        else:
            # Use command-line args if provided, otherwise use config
            checkpoint = args.checkpoint if args.checkpoint else export_cfg.get("checkpoint")
            output = args.output if args.output else export_cfg.get("output")
            
            if not checkpoint or not output:
                print("❌ 错误: 未指定 checkpoint 或 output 路径")
                print("   请通过配置文件或命令行参数指定")
            else:
                try:
                    export_classifier_onnx(export_cfg, checkpoint, output)
                except Exception as e:
                    print(f"❌ 导出分类模型失败: {e}")
    
    # 导出 YOLO 模型
    if args.model in ["yolo", "all"]:
        yolo_cfg = cfg.get("export_onnx", {})
        if not yolo_cfg:
            print("❌ 错误: 配置文件中未找到 yolo_pose_inference 配置")
        else:
            try:
                export_yolo_onnx(yolo_cfg)
            except Exception as e:
                print(f"❌ 导出 YOLO 模型失败: {e}")

    print("\n✓ 导出完成!")