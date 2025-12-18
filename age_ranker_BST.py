# age_ranker.py
# Self-contained: downloads UTKFace via KaggleHub, trains pairwise age-ranker
# Includes deterministic age-balanced split, early stopping, and learning-curve plotting

import os, glob, random
import kagglehub
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


# ------------------------------------------------------------
# 1. Dataset 
# ------------------------------------------------------------
class PairwiseUTKFace(Dataset):
    def __init__(self, split="train", resize=192, max_images=2000, test_images=200,
                 augment=True, tau=8.0, max_diff=60):
        super().__init__()

        path = kagglehub.dataset_download("jangedoo/utkface-new")
        print(f"✅ UTKFace dataset path: {path}")

        # --- collect all image paths and ages ---
        files = sorted(glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True))
        samples = []
        for f in files:
            try:
                age = int(os.path.basename(f).split("_")[0])
                samples.append((f, age))
            except Exception:
                continue

        random.seed(0)
        random.shuffle(samples)

        # --- Split strategy ---
        test_split = random.sample(samples, test_images)
        test_paths = {f for f, _ in test_split}
        remaining = [(f, a) for f, a in samples if f not in test_paths]

        # --- Balanced age-bin sampling for training ---
        bins = [[] for _ in range(10)]
        for f, age in remaining:
            bin_idx = min(age // 10, 9)
            bins[bin_idx].append((f, age))

        per_bin = max_images // len(bins)
        train_split = []
        for b in bins:
            random.shuffle(b)
            if len(b) >= per_bin:
                train_split.extend(b[:per_bin])
            else:
                if len(b) > 0:
                    train_split.extend(random.choices(b, k=per_bin))
        random.shuffle(train_split)

        # --- Select which split to use ---
        chosen = train_split if split == "train" else test_split
        self.samples = chosen
        self.ages = np.array([a for _, a in self.samples])
        self.tau = tau
        self.max_diff = max_diff

        # --- base transform (resize + tensor + normalize) ---
        self.base_tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # --- independent augmentation setup ---
        self.augment = augment and (split == "train")
        self.flip_tf = transforms.RandomHorizontalFlip(p=0.5)

        # --- caching setup (grayscale precache) ---
        self.cache = {}
        self.preload_limit = 50000 # not really necessary anymore....
        n_preload = min(len(self.samples), self.preload_limit)
        print(f"⚡ Preloading first {n_preload}/{len(self.samples)} grayscale images into memory...")
        for f, _ in tqdm(self.samples[:n_preload], desc="Preload", leave=False):
            img = Image.open(f).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            gray_img = Image.fromarray((gray * 255).astype(np.uint8))
            tensor = self.base_tf(gray_img)
            self.cache[f] = tensor

        print(f"✅ Built {split} set with {len(chosen)} images (pairs sampled online)")

    def _get_tensor(self, path):
        if path in self.cache:
            return self.cache[path].clone()
        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        gray_img = Image.fromarray((gray * 255).astype(np.uint8))
        return self.base_tf(gray_img)

    def _sample_pair_indices(self):
        """Draw (i, j) uniformly over |Δage| ∈ [0, max_diff]."""
        N = len(self.samples)
        i = np.random.randint(0, N)
        a_i = self.ages[i]

        # 1️⃣  Uniformly sample an absolute age difference
        target_diff = np.random.uniform(0, self.max_diff)

        # 2️⃣  Randomly decide whether to go older or younger
        sign = np.random.choice([-1, 1])
        desired_age = a_i + sign * target_diff

        # 3️⃣  Find candidates whose age is within ±2 yrs of desired_age
        candidates = np.where(np.abs(self.ages - desired_age) < 2.0)[0]
        if len(candidates) == 0:
            j = np.random.randint(0, N)
        else:
            j = np.random.choice(candidates)

        # 4️⃣  Avoid i == j edge case
        if j == i:
            j = (j + 1) % N

        return i, j

    def __getitem__(self, idx):
        i, j = self._sample_pair_indices()
        f_i, age_i = self.samples[i]
        f_j, age_j = self.samples[j]
        x_i = self._get_tensor(f_i)
        x_j = self._get_tensor(f_j)

        if self.augment:
            x_i = self.flip_tf(x_i)
            x_j = self.flip_tf(x_j)

        x = torch.cat([x_i, x_j], dim=0)  # [2, H, W]
        diff = age_i - age_j
        p_younger = 1 / (1 + np.exp(-diff / self.tau))  # smooth label in [0,1]
        label = torch.tensor([1 - p_younger, p_younger], dtype=torch.float32)  # [P(older), P(younger)]
        return x, label

    def __len__(self):
        return 100000 # hyperparameter controls how often you test

# ------------------------------------------------------------
# 2. Model definition (ResNet-10-style, 2-channel grayscale input)
# ------------------------------------------------------------
class ResBlock(nn.Module):
    """Basic 2-conv residual block."""
    def __init__(self, in_ch, out_ch, stride=1, dropout_p=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class AgeRanker(nn.Module):
    """ResNet-10-style ranker (2-channel grayscale input)."""
    def __init__(self, base_ch=32, num_classes=2, layers=[1, 1, 1, 1], dropout_p=0.2):
        super().__init__()

        # Stem: 2-channel grayscale pair input
        self.stem = nn.Sequential(
            nn.Conv2d(2, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 stages × 1 block each → total ~10 conv layers
        self.layer1 = self._make_layer(base_ch,   base_ch,   blocks=layers[0], stride=1, dropout_p=dropout_p)
        self.layer2 = self._make_layer(base_ch,   base_ch*2, blocks=layers[1], stride=2, dropout_p=dropout_p)
        self.layer3 = self._make_layer(base_ch*2, base_ch*4, blocks=layers[2], stride=2, dropout_p=dropout_p)
        self.layer4 = self._make_layer(base_ch*4, base_ch*8, blocks=layers[3], stride=2, dropout_p=dropout_p)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(base_ch*8, 1)  # single output logit


    def _make_layer(self, in_ch, out_ch, blocks, stride, dropout_p):
        layers = [ResBlock(in_ch, out_ch, stride, dropout_p)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch, 1, dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)  # shape [B]
    
# ------------------------------------------------------------
# 3. Train/test helpers
# ------------------------------------------------------------
def run_epoch(model, loader, opt, criterion, device, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0
    loop = tqdm(loader, desc="Train" if train else "Eval", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y[:, 1])  # train on P(younger)
            if train:
                loss.backward()
                opt.step()
        preds = (torch.sigmoid(out) > 0.5).long()
        # preds = [1, 1, 0, 1]
        # y     = [1, 0, 0, 1]
        # preds == y [True, False, True, True]
        # (preds == y).sum() = 3 = correct
        # y.shape = [B, 1]
        # total = y.size(0) = B = 4
        # acc = correct / total = 3 / 4 = 75%
        correct += (preds == (y[:, 1] > 0.5)).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        loop.set_postfix(loss=f"{loss_sum/total:.3f}", acc=f"{100*correct/total:.2f}")
    return loss_sum / total, 100 * correct / total


# ------------------------------------------------------------
# 4. Main training logic
# ------------------------------------------------------------
def main():
    epochs = 1000 # we use early stopping so set large
    batch_size = 32
    resize = 192
    patience = 10
    eval_interval = 1.0 # test every full epoch (or eg 0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = PairwiseUTKFace("train", resize=resize, max_images=4000, test_images=500)
    test_ds = PairwiseUTKFace("test", resize=resize, max_images=4000, test_images=500)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AgeRanker(base_ch=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0
    checks_no_improve = 0 # patience now counts interim evals
    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total, correct, loss_sum = 0, 0, 0
        loop = tqdm(enumerate(train_dl), total=len(train_dl), desc="Train", leave=False)

        for batch_i, (x, y) in loop:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y[:, 1])  # train on P(younger)
            loss.backward()
            opt.step()

            preds = (torch.sigmoid(out) > 0.5).long()
            correct += (preds == (y[:, 1] > 0.5)).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
            loop.set_postfix(loss=f"{loss_sum/total:.3f}", acc=f"{100*correct/total:.2f}")

            # --- periodic evaluation ---
            if (batch_i + 1) % max(1, int(len(train_dl) * eval_interval)) == 0:
                test_loss, acc = run_epoch(model, test_dl, opt, criterion, device, train=False)
                print(f"  🔍 interim eval — acc: {acc:.2f}%, loss: {test_loss:.3f}")
                test_losses.append(test_loss)
                test_accs.append(acc)

                # --- early-stopping check (per interim eval) ---
                if acc > best_acc:
                    best_acc = acc
                    checks_no_improve = 0
                    torch.save(model.state_dict(), "age_ranker_best_resnet10_4ktrain_500test_soft_logits.pth")
                else:
                    checks_no_improve += 1
                    if checks_no_improve >= patience:
                        print(f"⏹️ Early stopping triggered after {checks_no_improve} evals "
                              f"(best acc {best_acc:.2f}%)")
                        plt.figure(figsize=(8, 5))
                        plt.plot(train_losses, label="Train loss")
                        plt.plot(test_losses, label="Test loss")
                        plt.plot(test_accs, label="Test acc (%)")
                        plt.xlabel("Eval step")
                        plt.ylabel("Loss / Accuracy")
                        plt.title("UTKFace Pairwise Age Ranking (ResNet-34)")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig("learning_curve.png", dpi=150)
                        plt.show()
                        return  # exit training early

        # --- end-of-epoch bookkeeping only ---
        train_loss = loss_sum / total
        train_losses.append(train_loss)
        print(f"Epoch {epoch} finished. (best acc so far {best_acc:.2f}%)")

# ------------------------------------------------------------
# 5. Relative-ranking age regression 
# ------------------------------------------------------------
@torch.no_grad()
def regress_age_sliding_window(model, train_ds, test_ds, device, kernel_size=5):
    """
    Estimate each test face's age using a sliding-window kernel over the
    integer ages present in the training set. The kernel compares the
    test face to younger and older bins while skipping the middle one.
    The vote curve V(age) reflects how consistent the model's pairwise
    predictions are with the hypothesis "test ≈ age".
    """
    model.eval()

    # --- Build lookup of training faces by age ---
    age_to_imgs = {}
    for f, age in train_ds.samples:
        age_to_imgs.setdefault(age, []).append(f)
    ages = sorted(age_to_imgs.keys())
    if len(ages) < kernel_size:
        raise ValueError("Training ages fewer than kernel size!")

    half = kernel_size // 2
    test_ages = np.array([a for _, a in test_ds.samples], dtype=float)
    pred_ages = np.zeros_like(test_ages)

    # --- For visualization of a few examples ---
    example_idxs = np.linspace(0, len(test_ds.samples)-1, num=5, dtype=int)
    example_votes = {}

    BATCH_SIZE = 128

    for ti, (f_test, true_age) in enumerate(tqdm(test_ds.samples, desc="Sliding-window regression")):
        x_test = test_ds._get_tensor(f_test).unsqueeze(0).to(device)
        votes = []

        for center_idx in range(len(ages)):
            a = ages[center_idx]
            younger = ages[max(0, center_idx - half):center_idx]
            older   = ages[center_idx + 1:center_idx + 1 + half]

            younger_refs = [f for age in younger for f in age_to_imgs[age]]
            older_refs   = [f for age in older for f in age_to_imgs[age]]
            if not younger_refs or not older_refs:
                votes.append(0.0)
                continue

            # ---- Batch comparisons (younger) ----
            probs_younger = []
            for k in range(0, len(younger_refs), BATCH_SIZE):
                batch_refs = younger_refs[k:k+BATCH_SIZE]
                ref_tensors = torch.stack([train_ds._get_tensor(f) for f in batch_refs]).to(device)
                x_batch = x_test.repeat(len(ref_tensors), 1, 1, 1)
                out = model(torch.cat([x_batch, ref_tensors], dim=1))
                p = torch.softmax(out, dim=1)[:, 0]  # prob(test older)
                probs_younger.extend(p.detach().cpu().tolist())

            # ---- Batch comparisons (older) ----
            probs_older = []
            for k in range(0, len(older_refs), BATCH_SIZE):
                batch_refs = older_refs[k:k+BATCH_SIZE]
                ref_tensors = torch.stack([train_ds._get_tensor(f) for f in batch_refs]).to(device)
                x_batch = x_test.repeat(len(ref_tensors), 1, 1, 1)
                out = model(torch.cat([x_batch, ref_tensors], dim=1))
                p = torch.softmax(out, dim=1)[:, 1]  # prob(test younger)
                probs_older.extend(p.detach().cpu().tolist())

            mean_vote = np.mean(probs_younger + probs_older)
            votes.append(mean_vote)

        votes = np.array(votes)
        pred_age = ages[np.argmax(votes)]
        pred_ages[ti] = pred_age

        if ti in example_idxs:
            example_votes[ti] = (true_age, ages, votes)

    # --- Compute overall MAE ---
    mae = np.mean(np.abs(pred_ages - test_ages))
    print(f"✅ Sliding-window MAE: {mae:.2f} years")

    # --- Print some examples ---
    for ti, (true_age, ages_axis, votes) in example_votes.items():
        print(f"Test #{ti} → True {true_age:.1f}, Pred {pred_ages[ti]:.1f}")
        plt.figure(figsize=(6, 3))
        plt.bar(ages_axis, votes, color='skyblue', width=1.0)
        plt.axvline(true_age, color='red', linestyle='--', label='True age')
        plt.axvline(pred_ages[ti], color='green', linestyle=':', label='Pred age')
        plt.xlabel("Candidate age")
        plt.ylabel("Vote strength")
        plt.title(f"Sliding-window voting for test {ti}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Global scatter ---
    plt.figure(figsize=(6,6))
    plt.scatter(test_ages, pred_ages, alpha=0.6)
    plt.plot([test_ages.min(), test_ages.max()],
             [test_ages.min(), test_ages.max()], 'r--', label='Ideal')
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Sliding-window Age Regression | MAE = {mae:.2f} yrs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("age_regression_sliding_window.png", dpi=150)
    plt.show()

    return pred_ages, mae

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

@torch.no_grad()
def analyze_ranking_errors(model, test_ds, device, max_diff=40, sample_pairs=30000, bin_size=10):
    """
    Visualize where ranking errors occur for a trained AgeRanker.
    Produces:
      1. Accuracy vs. |Δage|
      2. Signed bias histogram
      3. Coarse 10-year bin confusion heatmap
    """
    model.eval()
    results = []
    pairs = random.sample(test_ds.pairs, min(sample_pairs, len(test_ds.pairs)))

    for (i, j) in tqdm(pairs, desc="Evaluating random test pairs"):
        f_i, age_i = test_ds.samples[i]
        f_j, age_j = test_ds.samples[j]
        true_label = 0 if age_i > age_j else 1

        x_i = test_ds._get_tensor(f_i).unsqueeze(0).to(device)
        x_j = test_ds._get_tensor(f_j).unsqueeze(0).to(device)
        out = model(torch.cat([x_i, x_j], dim=1))
        pred = out.argmax(1).item()
        #p_older = torch.softmax(out, dim=1)[0, 0].item()
        p_older = torch.sigmoid(out).item()            # shape [B]

        results.append({
            "age_i": age_i,
            "age_j": age_j,
            "pred": pred,
            "true": true_label,
            "correct": int(pred == true_label),
            "d_age": age_i - age_j,
            "abs_diff": abs(age_i - age_j),
            "p_older": p_older
        })

    df = pd.DataFrame(results)

    # ------------------------
    # 1️⃣  Accuracy vs |Δage|
    # ------------------------
    bins = np.arange(0, max_diff + 1)
    acc_by_diff = df.groupby(pd.cut(df.abs_diff, bins))["correct"].mean()
    plt.figure(figsize=(8, 4))
    acc_by_diff.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.ylabel("Pairwise Accuracy")
    plt.xlabel("|Δ Age| (years)")
    plt.title("Pairwise Accuracy vs. Absolute Age Difference")
    plt.tight_layout()
    plt.savefig("acc_vs_delta_age.png", dpi=150)
    plt.show()

    # ------------------------
    # 2️⃣  Signed bias histogram
    # ------------------------
    df["signed_err"] = np.sign(df["d_age"]) * (1 - df["correct"])
    plt.figure(figsize=(6, 4))
    sns.histplot(df["signed_err"], bins=41, kde=True, color="coral")
    plt.xlabel("Signed Error (positive = overestimation)")
    plt.title("Directional Bias in Pairwise Predictions")
    plt.tight_layout()
    plt.savefig("signed_bias_hist.png", dpi=150)
    plt.show()

    # ------------------------
    # 3️⃣  Coarse bin confusion heatmap
    # ------------------------
    bins = list(range(0, 101, bin_size))
    n_bins = len(bins) - 1
    mat = np.zeros((n_bins, n_bins))
    counts = np.zeros_like(mat)

    for (i, j) in random.sample(test_ds.pairs, min(sample_pairs, len(test_ds.pairs))):
        f_i, age_i = test_ds.samples[i]
        f_j, age_j = test_ds.samples[j]
        bi, bj = min(age_i // bin_size, n_bins-1), min(age_j // bin_size, n_bins-1)
        if bi == bj:
            continue
        x_i = test_ds._get_tensor(f_i).unsqueeze(0).to(device)
        x_j = test_ds._get_tensor(f_j).unsqueeze(0).to(device)
        out = model(torch.cat([x_i, x_j], dim=1))
        pred = out.argmax(1).item()
        # increment both halves symmetrically for readability
        mat[bi, bj] += (pred == 0)
        mat[bj, bi] += (pred == 1)
        counts[bi, bj] += 1
        counts[bj, bi] += 1

    acc = mat / (counts + 1e-8)
    plt.figure(figsize=(7, 6))
    sns.heatmap(acc, vmin=0.4, vmax=1.0, cmap="viridis",
                xticklabels=[f"{b}-{b+bin_size-1}" for b in bins[:-1]],
                yticklabels=[f"{b}-{b+bin_size-1}" for b in bins[:-1]])
    plt.xlabel("Younger bin")
    plt.ylabel("Older bin")
    plt.title(f"Pairwise Correctness between {bin_size}-year Age Bins")
    plt.tight_layout()
    plt.savefig("bin_confusion_heatmap.png", dpi=150)
    plt.show()

    print("✅ Diagnostic plots saved:")
    print(" - acc_vs_delta_age.png")
    print(" - signed_bias_hist.png")
    print(" - bin_confusion_heatmap.png")
    return df

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.ndimage as nd

@torch.no_grad()
def infer_age_by_entropy(model, train_ds, f_test, device, smooth_sigma=1.0, batch_size=256):
    """
    Estimate the true age of a test face as the region of highest uncertainty (max entropy)
    across all 1-year age bins in the training dataset.

    Args:
        model: trained pairwise ranking model (2-channel input).
        train_ds: training dataset with .samples = [(path, age), ...].
        f_test: path to the test image.
        device: torch device.
        smooth_sigma: Gaussian smoothing sigma for vote curve.
        batch_size: number of pairwise comparisons to batch for speed.

    Returns:
        pred_age (float): estimated age of highest entropy.
        ages (np.ndarray): sorted ages used for bins.
        votes (np.ndarray): per-age mean "older" probability.
        entropies (np.ndarray): per-age local entropy values.
    """
    model.eval()

    # --- Build lookup of training faces by age ---
    age_to_imgs = {}
    for f, age in train_ds.samples:
        age_to_imgs.setdefault(age, []).append(f)
    ages = sorted(age_to_imgs.keys())

    # --- Get the test tensor once ---
    x_test = train_ds._get_tensor(f_test).unsqueeze(0).to(device)

    # --- Compute average "older" probability for each age bin ---
    votes = []
    for a in tqdm(ages, desc="Comparing to age bins", leave=False):
        refs = age_to_imgs[a]
        probs = []
        for k in range(0, len(refs), batch_size):
            batch_refs = refs[k:k+batch_size]
            ref_tensors = torch.stack([train_ds._get_tensor(f) for f in batch_refs]).to(device)
            x_batch = x_test.repeat(len(ref_tensors), 1, 1, 1)
            out = model(torch.cat([x_batch, ref_tensors], dim=1))
            #p_older = torch.softmax(out, dim=1)[:, 0]  # prob(test older)
            p_older = torch.sigmoid(out)
            probs.extend(p_older.detach().cpu().tolist())
        votes.append(np.mean(probs))
    votes = np.array(votes)

    # --- Smooth the vote curve ---
    if smooth_sigma > 0:
        votes = nd.gaussian_filter1d(votes, sigma=smooth_sigma)

    # --- Compute entropy per age ---
    p = np.clip(votes, 1e-6, 1 - 1e-6)
    entropies = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # --- Find age of highest entropy (most uncertain region) ---
    max_idx = np.argmax(entropies)
    pred_age = ages[max_idx]

    # --- Visualization ---
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ages, votes, 'b-', label="P(test older)")
    ax1.set_xlabel("Reference age (years)")
    ax1.set_ylabel("P(test older)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(ages, entropies, 'r--', label="Entropy")
    ax2.set_ylabel("Entropy", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.axvline(pred_age, color='g', linestyle=':', label=f"Pred age = {pred_age}")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Entropy-based Age Estimation")
    plt.tight_layout()
    plt.show()

    return pred_age, np.array(ages), votes, entropies

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.ndimage as nd
import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def infer_all_by_entropy_with_images(model, train_ds, test_ds, device,
                                     smooth_sigma=1.0, batch_size=256,
                                     n_examples=5,
                                     use_filter=False,
                                     use_aug8x=False,
                                     drop_frac=0.1):
    model.eval()

    # --- Optional reference filtering FIRST ---
    if use_filter:
        print(f"🧩 Applying self-consistency filtering (drop_frac={drop_frac})...")
        train_ds = filter_reference_pool(model, train_ds, device,
                                         drop_frac=drop_frac,
                                         batch_size=batch_size)

    # --- Build age_to_imgs from possibly filtered dataset ---
    age_to_imgs = {}
    for f, age in train_ds.samples:
        age_to_imgs.setdefault(age, []).append(f)
    ages = sorted(age_to_imgs.keys())

    pred_ages, true_ages = [], []
    example_indices = np.linspace(0, len(test_ds.samples) - 1, n_examples, dtype=int)
    examples = {}

    for ti, (f_test, true_age) in enumerate(tqdm(test_ds.samples, desc="Entropy inference")):
        x_test = test_ds._get_tensor(f_test).unsqueeze(0).to(device)

        # --- mean P(test older) per reference age ---
        votes = []
        for a in ages:
            refs = age_to_imgs[a]
            probs = []
            if use_aug8x:
                probs = []
                flip_h = transforms.RandomHorizontalFlip(p=1.0)
                for k in range(0, len(refs), batch_size):
                    batch_refs = refs[k:k + batch_size]
                    for f in batch_refs:
                        x_ref = train_ds._get_tensor(f).to(device)

                        # --- 8× variants ---
                        aug_pairs = []
                        for t_flip in [False, True]:
                            for r_flip in [False, True]:
                                for swap in [False, True]:
                                    x_t = flip_h(x_test.squeeze(0)) if t_flip else x_test.squeeze(0)
                                    x_r = flip_h(x_ref) if r_flip else x_ref
                                    pair = torch.cat([x_t, x_r], dim=0)
                                    if swap:
                                        pair = torch.flip(pair, dims=[0])  # swap channels
                                    aug_pairs.append((pair, swap))

                        # --- Batch inference for this test/ref pair ---
                        x_aug = torch.stack([p[0] for p in aug_pairs]).to(device)
                        out = model(x_aug)
                        p_older = torch.sigmoid(out).cpu().numpy()
                        # p_older = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
                        for p_val, (_, swapped) in zip(p_older, aug_pairs):
                            # when swapped, invert meaning of older/younger
                            probs.append(1 - p_val if swapped else p_val)
                votes.append(np.mean(probs))
            else:
                # --- original unaugmented version ---
                probs = []
                for k in range(0, len(refs), batch_size):
                    batch_refs = refs[k:k + batch_size]
                    ref_tensors = torch.stack([train_ds._get_tensor(f) for f in batch_refs]).to(device)
                    x_batch = x_test.repeat(len(ref_tensors), 1, 1, 1)
                    out = model(torch.cat([x_batch, ref_tensors], dim=1))
                    p_older = torch.sigmoid(out)  # prob(test older)
                    #p_older = torch.softmax(out, dim=1)[:, 0]  # prob(test older)
                    probs.extend(p_older.detach().cpu().tolist())
                votes.append(np.mean(probs))

        votes = np.array(votes)
        if smooth_sigma > 0:
            votes = nd.gaussian_filter1d(votes, sigma=smooth_sigma)

        # --- compute entropy & find highest point ---
        p = np.clip(votes, 1e-6, 1 - 1e-6)
        entropies = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        pred_age = ages[np.argmax(entropies)]

        pred_ages.append(pred_age)
        true_ages.append(true_age)

        # save example for plotting later
        if ti in example_indices:
            examples[ti] = dict(
                path=f_test,
                true_age=true_age,
                pred_age=pred_age,
                ages=ages,
                votes=votes,
                entropies=entropies
            )

    # --- Compute global MAE ---
    pred_ages = np.array(pred_ages, dtype=float)
    true_ages = np.array(true_ages, dtype=float)
    mae = np.mean(np.abs(pred_ages - true_ages))
    print(f"\n✅ Global MAE = {mae:.2f} years over {len(true_ages)} test faces")

    # --- Scatter plot overall ---
    plt.figure(figsize=(6, 6))
    plt.scatter(true_ages, pred_ages, alpha=0.6)
    plt.plot([true_ages.min(), true_ages.max()],
             [true_ages.min(), true_ages.max()], 'r--', label='Ideal')
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Entropy-based Age Regression | MAE = {mae:.2f} yrs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot a few detailed examples with image thumbnails ---
    for ti, ex in examples.items():
        ages, votes, entropies = ex["ages"], ex["votes"], ex["entropies"]
        true_age, pred_age = ex["true_age"], ex["pred_age"]
        img_path = ex["path"]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(ages, votes, 'b-', label="P(test older)")
        ax1.axvline(pred_age, color='green', linestyle=':', label=f"Pred age = {pred_age:.1f}")
        ax1.axvline(true_age, color='black', linestyle='--', label=f"True age = {true_age}")
        ax1.set_xlabel("Reference age (years)")
        ax1.set_ylabel("P(test older)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(ages, entropies, 'r--', label="Entropy")
        ax2.set_ylabel("Entropy", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(f"Test #{ti} — True {true_age}, Pred {pred_age:.1f}")
        fig.tight_layout()

        # --- Add the test image as a small inset ---
        try:
            img = mpimg.imread(img_path)
            # convert to grayscale preview if desired:
            # img = np.mean(img, axis=2)
            imagebox = OffsetImage(img, zoom=0.6)
            ab = AnnotationBbox(imagebox, (0.85, 0.75), xycoords='axes fraction',
                                box_alignment=(1, 1), frameon=True, pad=0.3)
            ax1.add_artist(ab)
        except Exception as e:
            print(f"[⚠️] Could not load image {img_path}: {e}")

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

    return pred_ages, true_ages, mae

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.no_grad()
def analyze_loss_vs_age(model, dataset, device, batch_size=256):
    """
    Evaluate pairwise ranking loss and accuracy as a function of:
      1. true age (per-person),
      2. absolute age difference (per-pair).
    """

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    # --- collect all ages ---
    ages = np.array([age for _, age in dataset.samples])
    n = len(dataset.samples)

    # --- random subset of pairs to keep it fast ---
    n_pairs = 5000
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    pairs = random.sample(pairs, min(n_pairs, len(pairs)))

    all_losses, all_accs, all_age_diffs, all_age_means = [], [], [], []

    for k in tqdm(range(0, len(pairs), batch_size), desc="Analyzing losses"):
        batch = pairs[k:k+batch_size]
        x_batch, y_batch, age_diffs, age_means = [], [], [], []
        for i, j in batch:
            (f_i, age_i), (f_j, age_j) = dataset.samples[i], dataset.samples[j]
            x_i = dataset._get_tensor(f_i)
            x_j = dataset._get_tensor(f_j)
            x_batch.append(torch.cat([x_i, x_j], dim=0))
            y_batch.append(0 if age_i > age_j else 1)
            age_diffs.append(abs(age_i - age_j))
            age_means.append((age_i + age_j) / 2)

        x_batch = torch.stack(x_batch).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)
        out = model(x_batch)
        loss = criterion(out, y_batch[:, 1])  # train on P(younger)
        preds = (torch.sigmoid(out) > 0.5).long()
        acc = (preds == (y_batch[:, 1] > 0.5)).float()

        all_losses.extend(loss.detach().cpu().tolist())
        all_accs.extend(acc.detach().cpu().tolist())
        all_age_diffs.extend(age_diffs)
        all_age_means.extend(age_means)

    all_losses = np.array(all_losses)
    all_accs = np.array(all_accs)
    all_age_diffs = np.array(all_age_diffs)
    all_age_means = np.array(all_age_means)

    # --- 1. Mean loss vs true age (using age mean as proxy) ---
    bins = np.arange(0, np.max(all_age_means)+5, 5)
    bin_means = (bins[:-1] + bins[1:]) / 2
    loss_vs_age, _, _ = binned_statistic(all_age_means, all_losses, bins=bins, statistic="mean")
    acc_vs_age, _, _ = binned_statistic(all_age_means, all_accs, bins=bins, statistic="mean")

    # --- 2. Mean loss vs age difference ---
    diffs = np.arange(0, np.max(all_age_diffs)+5, 5)
    diff_centers = (diffs[:-1] + diffs[1:]) / 2
    loss_vs_diff, _, _ = binned_statistic(all_age_diffs, all_losses, bins=diffs, statistic="mean")
    acc_vs_diff, _, _ = binned_statistic(all_age_diffs, all_accs, bins=diffs, statistic="mean")

    # --- Plot ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(bin_means, loss_vs_age, 'r-', label="Mean loss")
    ax[0].set_xlabel("True age (mean of pair)")
    ax[0].set_ylabel("Cross-entropy loss")
    ax[0].set_title("Loss vs age region")
    ax[0].legend()

    ax[1].plot(diff_centers, acc_vs_diff, 'b-', label="Accuracy")
    ax[1].set_xlabel("Age difference between pair (years)")
    ax[1].set_ylabel("Pairwise accuracy")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Accuracy vs. |Δ age|")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Overall pairwise accuracy: {np.mean(all_accs)*100:.2f}%")

# ------------------------------------------------------------
# 5b. Optional self-consistency filtering of reference pool
# ------------------------------------------------------------
@torch.no_grad()
def filter_reference_pool(model, train_ds, device, drop_frac=0.1, batch_size=256):
    """
    Evaluate each training sample by inferring its own age using all others as references.
    Drop the top drop_frac with the highest self-MAE.
    Returns a *new* filtered dataset (train_ds is not modified in-place).
    """
    model.eval()
    n = len(train_ds.samples)
    errors = []

    print(f"🔍 Evaluating {n} training refs for self-consistency...")
    for i, (f_i, true_age) in enumerate(tqdm(train_ds.samples, desc="Self-consistency")):
        refs = [f for j, (f, _) in enumerate(train_ds.samples) if j != i]
        age_to_imgs = {}
        for f in refs:
            age = int(os.path.basename(f).split("_")[0])
            age_to_imgs.setdefault(age, []).append(f)

        ages = sorted(age_to_imgs.keys())
        x_test = train_ds._get_tensor(f_i).unsqueeze(0).to(device)

        votes = []
        for a in ages:
            ref_list = age_to_imgs[a]
            probs = []
            for k in range(0, len(ref_list), batch_size):
                batch_refs = ref_list[k:k+batch_size]
                ref_tensors = torch.stack([train_ds._get_tensor(f) for f in batch_refs]).to(device)
                x_batch = x_test.repeat(len(ref_tensors), 1, 1, 1)
                out = model(torch.cat([x_batch, ref_tensors], dim=1))
                p_older = torch.sigmoid(out)
                # p_older = torch.softmax(out, dim=1)[:, 0]
                probs.extend(p_older.detach().cpu().tolist())
            votes.append(np.mean(probs))

        # entropy-based predicted age
        p = np.clip(np.array(votes), 1e-6, 1 - 1e-6)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        pred_age = ages[np.argmax(ent)]
        err = abs(pred_age - true_age)
        errors.append((i, err))

    errors.sort(key=lambda x: x[1], reverse=True)
    n_drop = int(len(errors) * drop_frac)
    drop_ids = set(idx for idx, _ in errors[:n_drop])

    print(f"🧹 Dropping {n_drop}/{n} refs (~{100*drop_frac:.1f}%) with highest self-MAE")
    filtered = [s for i, s in enumerate(train_ds.samples) if i not in drop_ids]
    new_ds = train_ds
    new_ds.samples = filtered
    #new_ds.pairs = [(i, j) for i in range(len(filtered)) for j in range(len(filtered)) if i != j]
    return new_ds

# ------------------------------------------------------------
# 5c. Inference-time 8× augmentation
# ------------------------------------------------------------
def get_augmented_pairs(x_test, x_ref):
    """
    Generate 8 variants of the (test, ref) comparison:
      - original vs flip for both test and ref
      - swapped channel order (reverses label meaning)
    Returns: list of (pair_tensor, flip_label)
             where flip_label=True means output should be inverted (1↔0)
    """
    aug_pairs = []
    flip_h = transforms.RandomHorizontalFlip(p=1.0)

    # base and flipped versions
    t_base = x_test
    t_flip = flip_h(x_test)
    r_base = x_ref
    r_flip = flip_h(x_ref)

    combos = [
        (t_base, r_base, False),
        (t_base, r_flip, False),
        (t_flip, r_base, False),
        (t_flip, r_flip, False),
        (r_base, t_base, True),
        (r_flip, t_base, True),
        (r_base, t_flip, True),
        (r_flip, t_flip, True),
    ]
    for t, r, flipped in combos:
        aug_pairs.append((torch.cat([t, r], dim=0), flipped))
    return aug_pairs

@torch.no_grad()
def infer_all_by_binary_search(model, train_ds, test_ds, device,
                               start_age=60, max_depth=6, local_window=5,
                               batch_size=256, n_examples=5):
    """
    Coarse-to-fine age inference using binary search over reference ages.
    For each test image:
        1. Do binary search (≈log2 N) using mean P(test older)
        2. Then refine locally ±local_window years via entropy
    """

    model.eval()

    # --- prepare references by age ---
    age_to_imgs = {}
    for f, age in train_ds.samples:
        age_to_imgs.setdefault(age, []).append(f)
    ages = sorted(age_to_imgs.keys())

    pred_ages, true_ages = [], []
    example_idxs = np.linspace(0, len(test_ds.samples)-1, num=n_examples, dtype=int)
    examples = {}

    def mean_prob(x_test, age_idx):
        """Mean P(test older) for all refs of given age index."""
        refs = age_to_imgs[ages[age_idx]]
        probs = []
        for k in range(0, len(refs), batch_size):
            batch_refs = refs[k:k+batch_size]
            ref_tensors = torch.stack([train_ds._get_tensor(f) for f in batch_refs]).to(device)
            out = model(torch.cat([x_test.repeat(len(ref_tensors),1,1,1), ref_tensors], dim=1))
            p = torch.softmax(out, dim=1)[:,0]  # prob(test older)
            probs.extend(p.detach().cpu().tolist())
        return np.mean(probs)

    # --- loop over test images ---
    for ti, (f_test, true_age) in enumerate(tqdm(test_ds.samples, desc="Binary search inference")):
        x_test = test_ds._get_tensor(f_test).unsqueeze(0).to(device)

        # ---- Coarse binary search ----
        low, high = 0, len(ages)-1
        idx = np.searchsorted(ages, start_age)
        idx = np.clip(idx, 0, high)
        for _ in range(max_depth):
            p = mean_prob(x_test, idx)
            if p > 0.5 and idx < high:
                low = idx + 1
            elif p <= 0.5 and idx > low:
                high = idx - 1
            idx = (low + high) // 2
            if high - low <= 2:
                break

        # ---- Local dense refinement ----
        center = ages[idx]
        local_range = [a for a in ages if abs(a - center) <= local_window]
        votes = []
        for a in local_range:
            refs = age_to_imgs[a]
            probs = []
            for k in range(0, len(refs), batch_size):
                batch_refs = refs[k:k+batch_size]
                ref_tensors = torch.stack([train_ds._get_tensor(f) for f in batch_refs]).to(device)
                out = model(torch.cat([x_test.repeat(len(ref_tensors),1,1,1), ref_tensors], dim=1))
                p_older = torch.sigmoid(out)
                # p_older = torch.softmax(out, dim=1)[:,0]
                probs.extend(p_older.detach().cpu().tolist())
            votes.append(np.mean(probs))

        votes = np.array(votes)
        p = np.clip(votes, 1e-6, 1 - 1e-6)
        entropies = -(p*np.log(p)+(1-p)*np.log(1-p))
        pred_age = local_range[np.argmax(entropies)]

        pred_ages.append(pred_age)
        true_ages.append(true_age)

        if ti in example_idxs:
            examples[ti] = dict(
                path=f_test,
                true_age=true_age,
                pred_age=pred_age,
                local_range=local_range,
                votes=votes,
                entropies=entropies
            )

    # --- MAE computation ---
    pred_ages = np.array(pred_ages)
    true_ages = np.array(true_ages)
    mae = np.mean(np.abs(pred_ages - true_ages))
    print(f"\n✅ Binary-search MAE = {mae:.2f} years over {len(test_ds.samples)} test faces")

    # --- Scatter plot overall ---
    plt.figure(figsize=(6,6))
    plt.scatter(true_ages, pred_ages, alpha=0.6)
    plt.plot([true_ages.min(), true_ages.max()],
             [true_ages.min(), true_ages.max()], 'r--', label='Ideal')
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Binary + Local Search | MAE = {mae:.2f} yrs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Example plots ---
    for ti, ex in examples.items():
        local_range, votes, entropies = ex["local_range"], ex["votes"], ex["entropies"]
        true_age, pred_age = ex["true_age"], ex["pred_age"]
        img_path = ex["path"]

        fig, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(local_range, votes, 'b-', label="P(test older)")
        ax1.axvline(pred_age, color='green', linestyle=':', label=f"Pred = {pred_age:.1f}")
        ax1.axvline(true_age, color='black', linestyle='--', label=f"True = {true_age}")
        ax1.set_xlabel("Reference age (years)")
        ax1.set_ylabel("P(test older)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(local_range, entropies, 'r--', label="Entropy")
        ax2.set_ylabel("Entropy", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(f"Example #{ti} — True {true_age}, Pred {pred_age:.1f}")
        fig.tight_layout()
        plt.show()

    return pred_ages, true_ages, mae

@torch.no_grad()
def infer_all_by_binary_search_with_filter_tta(model, train_ds, test_ds, device,
                                               start_age=60, max_depth=6,
                                               local_window=5, batch_size=256,
                                               n_examples=5,
                                               use_filter=True,
                                               drop_frac=0.075,
                                               use_aug8x=True):
    """
    Binary + local dense age inference with optional:
      • self-consistency filtering (drop_frac worst refs)
      • 8× test-time augmentation
    """

    model.eval()

    # ------------------------------------------------------------
    # 1. Optional self-consistency filtering
    # ------------------------------------------------------------
    if use_filter:
        print(f"🧩 Applying self-consistency filtering (drop_frac={drop_frac})...")
        train_ds = filter_reference_pool(model, train_ds, device,
                                         drop_frac=drop_frac,
                                         batch_size=batch_size)

    # ------------------------------------------------------------
    # 2. Build reference lookup by age
    # ------------------------------------------------------------
    age_to_imgs = {}
    for f, age in train_ds.samples:
        age_to_imgs.setdefault(age, []).append(f)
    ages = sorted(age_to_imgs.keys())

    pred_ages, true_ages = [], []
    example_idxs = np.linspace(0, len(test_ds.samples)-1, num=n_examples, dtype=int)
    examples = {}

    # ------------------------------------------------------------
    # 3. Helper: mean probability (with optional 8× TTA)
    # ------------------------------------------------------------
    def mean_prob(x_test, refs):
        probs = []
        for k in range(0, len(refs), batch_size):
            batch_refs = refs[k:k+batch_size]
            for f in batch_refs:
                x_ref = train_ds._get_tensor(f).to(device)

                if use_aug8x:
                    aug_pairs = get_augmented_pairs(x_test.squeeze(0), x_ref)
                    x_aug = torch.stack([p[0] for p in aug_pairs]).to(device)
                    out = model(x_aug)
                    p_older = torch.sigmoid(out).cpu().numpy()
                    #p_older = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
                    for p_val, (_, swapped) in zip(p_older, aug_pairs):
                        probs.append(1 - p_val if swapped else p_val)
                else:
                    out = model(torch.cat([x_test, x_ref.unsqueeze(0)], dim=1))
                    p_older = torch.sigmoid(out)
                    #p_older = torch.softmax(out, dim=1)[:, 0]
                    probs.append(p_older.item()).cpu().numpy()
        return np.mean(probs)

    # ------------------------------------------------------------
    # 4. Iterate over test images
    # ------------------------------------------------------------
    for ti, (f_test, true_age) in enumerate(tqdm(test_ds.samples, desc="Binary search inference")):
        x_test = test_ds._get_tensor(f_test).unsqueeze(0).to(device)

        # ---- Coarse binary search ----
        low, high = 0, len(ages)-1
        idx = np.searchsorted(ages, start_age)
        idx = np.clip(idx, 0, high)
        for _ in range(max_depth):
            p = mean_prob(x_test, age_to_imgs[ages[idx]])
            if p > 0.5 and idx < high:
                low = idx + 1
            elif p <= 0.5 and idx > low:
                high = idx - 1
            idx = (low + high) // 2
            if high - low <= 2:
                break

        # ---- Local dense refinement ----
        center = ages[idx]
        local_range = [a for a in ages if abs(a - center) <= local_window]
        votes = []
        for a in local_range:
            refs = age_to_imgs[a]
            votes.append(mean_prob(x_test, refs))

        votes = np.array(votes)
        p = np.clip(votes, 1e-6, 1 - 1e-6)
        entropies = -(p*np.log(p)+(1-p)*np.log(1-p))
        pred_age = local_range[np.argmax(entropies)]

        pred_ages.append(pred_age)
        true_ages.append(true_age)

        if ti in example_idxs:
            examples[ti] = dict(
                path=f_test,
                true_age=true_age,
                pred_age=pred_age,
                local_range=local_range,
                votes=votes,
                entropies=entropies
            )

    # ------------------------------------------------------------
    # 5. Compute and visualize results
    # ------------------------------------------------------------
    pred_ages = np.array(pred_ages)
    true_ages = np.array(true_ages)
    mae = np.mean(np.abs(pred_ages - true_ages))
    print(f"\n✅ Binary-search MAE = {mae:.2f} years ({len(test_ds.samples)} test faces)")

    plt.figure(figsize=(6,6))
    plt.scatter(true_ages, pred_ages, alpha=0.6)
    plt.plot([true_ages.min(), true_ages.max()],
             [true_ages.min(), true_ages.max()], 'r--', label='Ideal')
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Binary + Local + Filter + 8× TTA | MAE = {mae:.2f} yrs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for ti, ex in examples.items():
        local_range, votes, entropies = ex["local_range"], ex["votes"], ex["entropies"]
        true_age, pred_age = ex["true_age"], ex["pred_age"]
        img_path = ex["path"]

        fig, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(local_range, votes, 'b-', label="P(test older)")
        ax1.axvline(pred_age, color='green', linestyle=':', label=f"Pred = {pred_age:.1f}")
        ax1.axvline(true_age, color='black', linestyle='--', label=f"True = {true_age}")
        ax1.set_xlabel("Reference age (years)")
        ax1.set_ylabel("P(test older)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(local_range, entropies, 'r--', label="Entropy")
        ax2.set_ylabel("Entropy", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title(f"Example #{ti} — True {true_age}, Pred {pred_age:.1f}")
        fig.tight_layout()
        plt.show()

    return pred_ages, true_ages, mae

if __name__ == "__main__":
    #main()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ds = PairwiseUTKFace("train", resize=192, max_images=4000, test_images=500)
    test_ds  = PairwiseUTKFace("test",  resize=192, max_images=4000, test_images=500)
    model = AgeRanker(base_ch=32).to(device)
    model.load_state_dict(torch.load("age_ranker_best_resnet10_4ktrain_500test_soft_logits.pth", map_location=device))

    import copy
    ref_ds = copy.deepcopy(train_ds)
    ref_ds.samples += test_ds.samples

    pred_ages, true_ages, mae = infer_all_by_binary_search_with_filter_tta(
        model, ref_ds, test_ds, device,
        start_age=60,
        max_depth=6,
        local_window=5,
        batch_size=256,
        n_examples=20,
        use_filter=True,
        drop_frac=0.075,
        use_aug8x=True
    )

    '''
    # 🔧 toggles
    USE_FILTER = True
    USE_AUG8X = True

    pred_ages, true_ages, mae = infer_all_by_entropy_with_images(
        model, train_ds, test_ds, device,
        smooth_sigma=1.0,
        batch_size=256,
        n_examples=50,
        use_filter=USE_FILTER,
        use_aug8x=USE_AUG8X,
        drop_frac=0.075
    )
    '''
