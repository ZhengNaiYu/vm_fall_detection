import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model(model, train_loader, val_loader, num_classes=3, num_epochs=200, learning_rate=0.0001, 
                weight_decay=1e-4, patience=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains a detection and recognition model using LSTM or GRU.
    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_classes (int): Number of output classes.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        weight_decay (float): L2 regularization parameter.
        patience (int): Number of epochs without improvement before early stopping.
        device (str): Device to use ("cuda" or "cpu").
    Returns:
        model: Trained model.
        history: Dictionary with training and validation metrics.
    """

    model.to(device)

    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss() 
    else:
        criterion = nn.CrossEntropyLoss()  
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Track metrics
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            # outputs = model(batch_X).squeeze()
            outputs = model(batch_X)

            if num_classes == 1:
                batch_y = batch_y.float().unsqueeze(1)  # shape (B,1)
                loss = criterion(outputs, batch_y)
                predicted = (torch.sigmoid(outputs) >= 0.5).long().squeeze()
            else:
                loss = criterion(outputs, batch_y)
                predicted = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (predicted == batch_y.squeeze()).sum().item()
            total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)

        # Evaluation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        val_inference_time = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device) # torch.Size([8, 20, 34]) torch.Size([8])
                # outputs = model(batch_X).squeeze()

                start_time = time.time()
                outputs = model(batch_X)
                val_inference_time += time.time() - start_time

                if num_classes == 1:
                    batch_y = batch_y.float().unsqueeze(1)
                    loss = criterion(outputs, batch_y)
                    predicted = (torch.sigmoid(outputs) >= 0.5).long().squeeze()
                else:
                    loss = criterion(outputs, batch_y)
                    predicted = torch.argmax(outputs, dim=1)

                val_loss += loss.item()
                correct += (predicted == batch_y.squeeze()).sum().item()
                total += batch_y.size(0)


        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"Val batch time: {val_inference_time/len(val_loader)*1000:.2f} ms")

        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # if patience_counter >= patience:
        #     print(f"⏹️ Early stopping at epoch {epoch+1}")
        #     break

   # --- restore best model ---
    if best_model:
        model.load_state_dict(best_model)
        print("✔️ Model restored to best validation loss state.")

    return model, history
