import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import copy
import time
import torch
import random
import numpy as np
import scipy.ndimage as nd
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
train_bs = 32
test_bs = 64
num_epochs = 200
patience = 15
resize_dim = 96
task_idx = 2  # MRT column index
num_workers = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL COMPONENTS 
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SlicingNet(nn.Module):
    """Same exact slice encoder from your pairwise script."""
    def __init__(self, in_channels=1, base_channels=8):
        super().__init__()
        self.block1 = ResBlock(in_channels, base_channels, stride=2)
        self.block2 = ResBlock(base_channels, base_channels)
        self.block3 = ResBlock(base_channels, base_channels)
        self.block4 = ResBlock(base_channels, base_channels, stride=2)
        self.block5 = ResBlock(base_channels, base_channels, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = base_channels

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.dropout(out)
        out = self.pool(out)
        return out.flatten(1)


class RegressionNet(nn.Module):
    """
    Identical to RankingNet EXCEPT:
    - in_channels = 1
    - final output is scalar
    """
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.block1 = ResBlock(in_channels, base_channels, stride=2)
        self.block2 = ResBlock(base_channels, base_channels*2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(base_channels*2, 1),
            #nn.Tanh()                          # bound output to [-1,1]
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.dropout(out)
        out = self.pool(out)
        # return self.fc(out.flatten(1)) 
        #return 4.0 * self.fc(out.flatten(1))   # map to [-4,4]
        return self.fc(out.flatten(1)) / 20


class BrainRegressor(nn.Module):
    """
    Full model:
    scan → slice encoder → [C, D] embedding map → RegressionNet → scalar
    """
    def __init__(self, slicer, regressor):
        super().__init__()
        self.slicer = slicer
        self.regressor = regressor

    def encode_scan(self, scan3d):
        """
        scan3d: [1, 96, 96, 96]
        Returns embedding map: [1, C, D] = [1, 8, 96]
        """
        scan3d = scan3d.squeeze(0)  # shape [96,96,96]
        slices = scan3d  # D,H,W

        feats = []
        for i in range(slices.shape[0]):
            sl = slices[i].unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
            emb = self.slicer(sl)  # [1, C]
            feats.append(emb)

        feats = torch.stack(feats, dim=2)  # [1, C, D]
        return feats

    def forward(self, scan3d):
        # scan3d: [B, 1, 96,96,96]
        B = scan3d.shape[0]
        outputs = []

        for b in range(B):
            f = self.encode_scan(scan3d[b])  # [1, C, D]
            f = f.unsqueeze(1)               # [1, 1, C, D]  <-- add channel dim
            out = self.regressor(f)          # OK for conv2d
            outputs.append(out)

        return torch.cat(outputs, dim=0)     # [B, 1]


# ============================================================
# DATASET
# ============================================================

class MRIDatasetRegression(Dataset):
    """
    Returns:
       resized_volume_tensor [1, 96,96,96]
       scalar MRT score
    """
    def __init__(self, split="train", n_test_subjects=8):
        self.split = split
        df = pd.read_csv("./indivs_192/combined.csv")
        df = df.dropna(subset=[df.columns[task_idx]])

        # --- EXCLUDE SUBJECTS BELOW -3.6 ---
        score_col = df.columns[task_idx]
        df = df[df[score_col] >= -3.6].reset_index(drop=True)

        self.scan_paths = df.iloc[:, 0].tolist()
        self.scores = df.iloc[:, task_idx].tolist()

        n = len(self.scan_paths)
        ids = list(range(n))
        sorted_ids = sorted(ids, key=lambda i: self.scores[i])
        step = max(1, len(sorted_ids) // n_test_subjects)

        self.test_subjects = set(sorted_ids[::step][:n_test_subjects])
        self.train_subjects = set(ids) - self.test_subjects

        if split == "train":
            self.indices = list(self.train_subjects)
        else:
            self.indices = list(self.test_subjects)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        score = float(self.scores[i])
        path = self.scan_paths[i]

        scan = np.load(path).astype(np.float32)
        labels = np.load(path[:-6] + "labels.npy").astype(np.int32)

        # Resize
        scan_t = torch.from_numpy(scan).unsqueeze(0).unsqueeze(0)
        lab_t  = torch.from_numpy(labels).unsqueeze(0).unsqueeze(0)

        scan_r = F.interpolate(scan_t, (96,96,96), mode="trilinear",
                               align_corners=False)[0,0]
        lab_r  = F.interpolate(lab_t.float(), (96,96,96), mode="nearest")[0,0].to(torch.int32)

        scan_r[lab_r == 0] = 0.0

        return scan_r.unsqueeze(0), torch.tensor([score], dtype=torch.float32)


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for scan, target in tqdm(loader, desc="Train", leave=False):
        scan = scan.to(device)
        target = target.to(device)

        pred = model(scan)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def test_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        for scan, target in tqdm(loader, desc="Test", leave=False):
            scan = scan.to(device)
            target = target.to(device)

            pred = model(scan)  # shape [B,1]

            loss = criterion(pred, target)
            total_loss += loss.item()

            preds.extend(pred.squeeze(1).cpu().tolist())   # FIXED
            trues.extend(target.squeeze(1).cpu().tolist()) # FIXED

    preds = np.array(preds)
    trues = np.array(trues)
    mae = np.mean(np.abs(preds - trues))

    return total_loss / len(loader), mae, preds.tolist(), trues.tolist()


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":
    slicer = SlicingNet(in_channels=1, base_channels=8).to(device)
    regressor = RegressionNet(in_channels=1, base_channels=16).to(device)
    model = BrainRegressor(slicer, regressor).to(device)

    train_ds = MRIDatasetRegression("train")
    test_ds  = MRIDatasetRegression("test")

    train_loader = DataLoader(train_ds, batch_size=train_bs,
                              shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=1,
                              shuffle=False, num_workers=num_workers)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    best_mae = float("inf")
    best_state = None
    epochs_no_improve = 0

    train_losses, test_losses, test_maes = [], [], []
    train_maes = []
    
    for epoch in range(1, num_epochs+1):
        print(f"\n===== Epoch {epoch} =====")
        
       # -------- Train --------
        tr_step_loss = train_epoch(model, train_loader, criterion, optimizer)

        # Compute TRAIN full-dataset MSE and MAE
        tr_loss_tmp, tr_mae_tmp, tr_preds, tr_trues = test_epoch(model, train_loader, criterion)
        tr_loss = tr_loss_tmp
        tr_mae  = tr_mae_tmp

        # -------- Test --------
        te_loss, te_mae, preds, trues = test_epoch(model, test_loader, criterion)

        print(f"Train Loss: {tr_loss:.4f} | Train MAE: {tr_mae:.4f}")
        print(f"Test  Loss: {te_loss:.4f} | Test  MAE: {te_mae:.4f}")

        # Store the full losses (NOT the per-step train loss)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_maes.append(tr_mae)
        test_maes.append(te_mae)

        # Early stopping
        if te_mae < best_mae:
            best_mae = te_mae
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    print(f"\nBest Test MAE = {best_mae:.4f}")

    # ------------------------------------------------------------
    # SCATTER PLOT — TRAIN = BLUE, TEST = ORANGE
    # ------------------------------------------------------------
    plt.figure(figsize=(6,6))

    # TRAIN POINTS
    plt.scatter(tr_trues, tr_preds, color="blue", alpha=0.6, label="Train")

    # TEST POINTS
    plt.scatter(trues, preds, color="orange", alpha=0.7, label="Test")

    # 45° line
    all_vals = tr_trues + trues + tr_preds + preds
    minn, maxx = min(all_vals), max(all_vals)
    plt.plot([minn, maxx], [minn, maxx], "r--")

    plt.xlabel("True MRT")
    plt.ylabel("Predicted MRT")
    plt.title(f"MSE Regression | Train MAE={tr_mae:.3f} | Test MAE={best_mae:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_scatter.png", dpi=200)
    plt.show()

    # ------------------------------------------------------------
    # TRAINING CURVES (LOSS + MAE)
    # ------------------------------------------------------------
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10,4))

    # Loss curves
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    # MAE curves
    plt.subplot(1,2,2)
    plt.plot(epochs, train_maes, label="Train MAE")
    plt.plot(epochs, test_maes, label="Test MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    plt.show()
