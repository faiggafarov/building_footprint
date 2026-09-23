import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.building_dataset import get_dataloaders
from models.unet import UNet
from models.losses import DiceBCELoss

# ---------- Metrics ----------
def compute_dice(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + eps)

def compute_iou(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)


def compute_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float().sum()
    return correct / targets.numel()

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ---------- Hyperparameters ----------
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4

# ---------- Paths ----------
IMAGE_DIR = "data/mass_tiles_npz/images"
LABEL_DIR = "data/mass_tiles_npz/labels"

print("Found images:", len(os.listdir(IMAGE_DIR)) if os.path.exists(IMAGE_DIR) else 0)
print("Found labels:", len(os.listdir(LABEL_DIR)) if os.path.exists(LABEL_DIR) else 0)

# ---------- Data ----------
train_loader, val_loader, test_loader = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE)

# ---------- Model ----------
# Massachusetts verisi 3 kanallı (RGB) olduğu için in_channels=3 olmalıdır.
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- Training ----------
def train():
    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = "logs/unet_metrics.csv"
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc", "Val Dice"])  # full metrics header

        best_val_dice = 0.0
        patience = 7
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            total_loss = 0
            total_train_acc = 0
            loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}", leave=False)

            for imgs, masks in loop:
                imgs = imgs.to(device)
                masks = masks.to(device).unsqueeze(1)

                preds = model(imgs)
                loss = criterion(preds, masks)
                train_acc = compute_accuracy(preds, masks).item()
                total_train_acc += train_acc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            avg_train_acc = total_train_acc / len(train_loader)

            # ---------- Validation ----------
            model.eval()
            dice_scores = []
            iou_scores = []
            val_loss_total = 0
            val_acc_total = 0

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device).unsqueeze(1)

                    preds = model(imgs)
                    loss = criterion(preds, masks)
                    val_loss_total += loss.item()
                    val_acc_total += compute_accuracy(preds, masks).item()
                    
                    dice = compute_dice(preds, masks).item()
                    iou = compute_iou(preds, masks).item()
                    dice_scores.append(dice)
                    iou_scores.append(iou)

            val_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
            val_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
            val_loss_avg = val_loss_total / len(val_loader) if len(val_loader) > 0 else 0
            val_acc_avg = val_acc_total / len(val_loader) if len(val_loader) > 0 else 0

            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | Val Loss = {val_loss_avg:.4f} | Train Acc = {avg_train_acc:.4f} | Val Acc = {val_acc_avg:.4f}", flush=True)
            writer.writerow([epoch + 1, avg_loss, val_loss_avg, avg_train_acc, val_acc_avg, val_dice])
            file.flush()
            
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                epochs_no_improve = 0
                torch.save(model.state_dict(), "checkpoints/unet_best.pth")
                print("Saved new best model.", flush=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.", flush=True)
                    break
            model.train()

    torch.save(model.state_dict(), "checkpoints/unet.pth")
    print("Model saved to checkpoints/unet.pth", flush=True)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()