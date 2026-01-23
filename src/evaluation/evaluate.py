import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_loader, num_classes=3, device=None):
    """
    Evaluates a trained model on test data.

    Args:
        model (torch.nn.Module)
        test_loader (DataLoader)
        num_classes (int)
        device (str or torch.device)

    Returns:
        dict: evaluation metrics (test_loss, accuracy, precision, recall, f1_score)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)

            if num_classes == 1:
                batch_y = batch_y.float().unsqueeze(1)
                loss = criterion(outputs, batch_y)
                predicted = (torch.sigmoid(outputs) >= 0.5).long().squeeze()
            else:
                batch_y = batch_y.long()
                loss = criterion(outputs, batch_y)
                predicted = torch.argmax(outputs, dim=1)

            test_loss += loss.item()
            # Robust to batch_size=1 (avoid 0-d arrays)
            all_preds.extend(predicted.view(-1).cpu().tolist())
            all_labels.extend(batch_y.view(-1).cpu().tolist())

    avg_test_loss = test_loss / len(test_loader)

    if num_classes == 1:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=1)
        recall = recall_score(all_labels, all_preds, zero_division=1)
        f1 = f1_score(all_labels, all_preds, zero_division=1)
    else:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='micro', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=1)

    print(f"🔍 Evaluation:")
    print(f"📉 Test Loss: {avg_test_loss:.4f}")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔄 Recall: {recall:.4f}")
    print(f"⭐ F1 Score: {f1:.4f}")

    return {
        "test_loss": avg_test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
