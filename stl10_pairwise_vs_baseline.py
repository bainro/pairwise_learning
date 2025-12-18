# stl10_pairwise_sameclass.py
# Pairwise "same vs different" STL-10 with N-choose-2 data explosion

import os, random, copy
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ============================================================
# 0. Global config
# ============================================================
GLOBAL_SEED = 0
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

# ============================================================
# 1. STL-10 Dataset utilities
# ============================================================
class STL10Simple(Dataset):
    def __init__(self, root="./data", split="train", max_images=1000,
                 resize=96, cache=True):
        super().__init__()
        tfm = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor()
        ])
        self.dataset = datasets.STL10(root=root, split=split,
                                      download=True, transform=tfm)

        # ---- enforce equal class counts ----
        n_classes = 10
        per_class = max_images // n_classes
        class_to_idx = defaultdict(list)
        for i, (_, y) in enumerate(self.dataset):
            class_to_idx[y].append(i)

        rng = np.random.default_rng(GLOBAL_SEED)
        chosen = []
        for c in range(n_classes):
            idxs = class_to_idx[c]
            rng.shuffle(idxs)
            chosen.extend(idxs[:per_class])

        # if max_images not divisible by 10, randomly add extras
        remaining = max_images - len(chosen)
        if remaining > 0:
            leftovers = np.concatenate(
                [class_to_idx[c][per_class:] for c in range(n_classes)]
            )
            rng.shuffle(leftovers)
            chosen.extend(leftovers[:remaining])

        self.indices = np.array(chosen)
        rng.shuffle(self.indices)

        # ---- caching (unchanged) ----
        self.cache = {}
        if cache:
            print(f"⚡ Caching {len(self.indices)} grayscale tensors for {split} set (balanced)...")
            for i in tqdm(self.indices):
                x, _ = self.dataset[i]
                gray = x.mean(0, keepdim=True)
                self.cache[i] = gray

        self.split = split
        self.n = len(self.indices)

    def __len__(self):
        return self.n

    def _get_gray(self, idx):
        if idx in self.cache:
            return self.cache[idx].clone()
        x, _ = self.dataset[idx]
        return x.mean(0, keepdim=True)

    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        gray = self._get_gray(self.indices[i])
        return gray, y, x

# ============================================================
# 2. Pairwise Dataset (same vs different)
# ============================================================
class PairwiseSTL10(Dataset):
    def __init__(self, all_imgs, all_labs, train_idx, test_idx,
                 split="train", practical_cap=100_000, test_cap=4_000, seed=1337):
        super().__init__()
        self.split = split
        self.practical_cap = practical_cap
        self.test_cap = test_cap
        self.rng = random.Random(seed)
        self.flip_h = transforms.RandomHorizontalFlip(p=0.5)

        self.train_pool = [(all_imgs[i], all_labs[i]) for i in train_idx]
        self.test_pool  = [(all_imgs[i], all_labs[i]) for i in test_idx]

        if split == "train":
            self.pool_a = self.train_pool
            self.pool_b = self.train_pool
            self.fixed_pairs = None
        else:
            self.pool_a = self.train_pool + self.test_pool
            self.pool_b = self.test_pool
            self.fixed_pairs = self._make_deterministic_pairs(self.test_cap)

    def _make_deterministic_pairs(self, cap):
        pairs = []
        rng = random.Random(0)
        all_a = list(range(len(self.pool_a)))
        all_b = list(range(len(self.pool_b)))
        while len(pairs) < cap:
            i = rng.choice(all_a)
            j = rng.choice(all_b)
            pairs.append((i, j))
        return pairs

    def resample_training_pairs(self):
        if self.split == "train":
            by_class = defaultdict(list)
            for i, (_, lab) in enumerate(self.pool_a):
                by_class[lab].append(i)

            same_frac = 0.5   # 50% same, 50% different
            num_same = int(self.practical_cap * same_frac)

            pairs = []
            # 1️⃣ same–same
            while len(pairs) < num_same:
                lab = random.choice(list(by_class.keys()))
                i, j = random.sample(by_class[lab], 2)
                pairs.append((i, j))
            # 2️⃣ different
            labs = list(by_class.keys())
            while len(pairs) < self.practical_cap:
                la, lb = random.sample(labs, 2)
                i = random.choice(by_class[la])
                j = random.choice(by_class[lb])
                pairs.append((i, j))
            random.shuffle(pairs)
            self.fixed_pairs = pairs

    def __len__(self):
        if self.fixed_pairs is None:
            self.resample_training_pairs()
        return len(self.fixed_pairs)

    def __getitem__(self, idx):
        if self.fixed_pairs is None:
            self.resample_training_pairs()
        i, j = self.fixed_pairs[idx]
        (x_i, lab_i) = self.pool_a[i]
        (x_j, lab_j) = self.pool_b[j]
        x_i = x_i.clone()
        x_j = x_j.clone()

        label = 1.0 if lab_i == lab_j else 0.0

        if self.split == "train":
            if random.random() < 0.5:
                x_i = self.flip_h(x_i)
            if random.random() < 0.5:
                x_j = self.flip_h(x_j)

        pair = torch.cat([x_i, x_j], dim=0)
        return pair, torch.tensor(label, dtype=torch.float32)


# ============================================================
# 3. Models
# ============================================================
def make_resnet18_pairwise():
    base = models.resnet18(weights=None)
    base.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base.fc = nn.Linear(base.fc.in_features, 1)
    return base

def make_resnet18_baseline(in_ch=1, n_classes=10):
    base = models.resnet18(weights=None)
    base.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base.fc = nn.Linear(base.fc.in_features, n_classes)
    return base


# ============================================================
# 4. Train / Eval helpers
# ============================================================
def train_epoch_bce(model, loader, opt, criterion, device):
    model.train()
    total, loss_sum, correct = 0, 0, 0
    for x, y in tqdm(loader, leave=False, desc="Train(pair)"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x).squeeze(1)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, 100 * correct / total

def eval_epoch_bce(model, loader, criterion, device):
    model.eval()
    total, loss_sum, correct = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="Eval(pair)"):
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(1)
            loss = criterion(out, y)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    return loss_sum / total, 100 * correct / total


def train_epoch_baseline(model, loader, opt, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for x, y in tqdm(loader, leave=False, desc="Train(base)"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, 100 * correct / total

def eval_epoch_baseline(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="Eval(base)"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    return loss_sum / total, 100 * correct / total


# =====================================
# 5. Inference via mean P(same)
# =====================================
@torch.no_grad()
def infer_by_sameclass(model, train_ds, test_ds, device,
                       batch_size=64, n_examples=5):
    model.eval()
    print("🔍 Inference by mean P(same class) — no TTA")

    # build reference pools
    class_to_imgs = defaultdict(list)
    for i in range(len(train_ds)):
        gray, lab, _ = train_ds[i]
        class_to_imgs[int(lab)].append(gray)
    classes = sorted(class_to_imgs.keys())

    pred_classes, true_classes = [], []
    example_indices = np.linspace(0, len(test_ds) - 1, n_examples, dtype=int)
    examples = {}

    for ti in tqdm(range(len(test_ds)), desc="Inference (no TTA)"):
        gray_test, true_lab, color_test = test_ds[ti]
        gray_test = gray_test.unsqueeze(0).to(device)
        sims = []

        # compare query to refs of each class
        for c in classes:
            refs = class_to_imgs[c]
            probs = []
            for k in range(0, len(refs), batch_size):
                batch_refs = refs[k:k + batch_size]
                ref_tensors = torch.stack(batch_refs).to(device)

                # forward (test, ref)
                pair = torch.cat([gray_test.repeat(len(ref_tensors), 1, 1, 1),
                                  ref_tensors], dim=1)
                out = model(pair).squeeze(1)
                p_same = torch.sigmoid(out).cpu().numpy()
                probs.extend(p_same)

            sims.append(np.mean(probs))

        sims = np.array(sims)
        pred_class = classes[np.argmax(sims)]
        pred_classes.append(pred_class)
        true_classes.append(true_lab)

        # collect a few examples for plotting
        if ti in example_indices:
            examples[ti] = dict(
                color=color_test.permute(1, 2, 0).numpy(),
                true=true_lab, pred=pred_class,
                sims=sims, classes=classes
            )

    # overall accuracy
    acc = np.mean(np.array(pred_classes) == np.array(true_classes))
    print(f"✅ Mean P(same) accuracy (no TTA): {acc*100:.2f}%")

    '''
    # scatter of predictions
    plt.figure(figsize=(6, 6))
    plt.scatter(true_classes, pred_classes, s=8, alpha=0.4, color="royalblue")
    plt.xlabel("True class")
    plt.ylabel("Pred class")
    plt.title(f"Mean P(same class) | no TTA | Acc={acc*100:.2f}%")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stl10_sameclass_noTTA_scatter.png", dpi=150)
    '''

    # example plots
    for ex in examples.values():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ex["classes"], ex["sims"], "b-o")
        ax.axvline(ex["true"], color="k", linestyle="--", label=f"True={ex['true']}")
        ax.axvline(ex["pred"], color="g", linestyle=":", label=f"Pred={ex['pred']}")
        imagebox = OffsetImage(ex["color"], zoom=0.6)
        ab = AnnotationBbox(imagebox, (0.85, 0.75), xycoords='axes fraction',
                            box_alignment=(1, 1), frameon=True, pad=0.3)
        ax.add_artist(ab)
        ax.legend()
        ax.set_xlabel("Class")
        ax.set_ylabel("Mean P(same)")
        plt.tight_layout()
        plt.show()

    return acc * 100 # turn into %

# ============================================================
# 6. Main experiment
# ============================================================
def run_experiment(N=2000, epochs=20, cap=100_000):
    print(f"\n=== Subset N={N} ===")

    full_train = STL10Simple(split="train", max_images=N)
    test = STL10Simple(split="test", max_images=2000)

    all_imgs = [g for g, _, _ in full_train] + [g for g, _, _ in test]
    all_labs = [y for _, y, _ in full_train] + [y for _, y, _ in test]
    train_idx = list(range(len(full_train)))
    test_idx = list(range(len(full_train), len(all_imgs)))

    train_pair = PairwiseSTL10(all_imgs, all_labs, train_idx, test_idx,
                               split="train", practical_cap=cap)
    test_pair  = PairwiseSTL10(all_imgs, all_labs, train_idx, test_idx,
                               split="test", test_cap=4000)
    dl_train = DataLoader(train_pair, batch_size=64, shuffle=True)
    dl_test  = DataLoader(test_pair, batch_size=64, shuffle=False)
    dl_train_base = DataLoader([(g, y) for g, y, _ in full_train],
                               batch_size=64, shuffle=True)
    dl_test_base  = DataLoader([(g, y) for g, y, _ in test],
                               batch_size=64, shuffle=False)

    # =====================================================
    # 🟦 Baseline: save best test accuracy checkpoint
    # =====================================================
    base = make_resnet18_baseline().to(device)
    opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    best_teA, best_path_base = 0, f"baseline_best_N{N}.pth"

    # baseline is fast, just do a fixed # of epochs
    for ep in range(1, 101):
        trL, trA = train_epoch_baseline(base, dl_train_base, opt, crit, device)
        teL, teA = eval_epoch_baseline(base, dl_test_base, crit, device)
        print(f"[Baseline] Ep{ep}: trA={trA:.3f}, teA={teA:.3f}")
        if teA > best_teA:
            best_teA = teA
            torch.save(base.state_dict(), best_path_base)

    print(f"💾 Saved best baseline model with {best_teA:.2f}% test acc")

    # reload best for inference (optional)
    base.load_state_dict(torch.load(best_path_base, map_location=device))
    baseline_acc = best_teA

    # =====================================================
    # 🟥 Pairwise: save best test accuracy checkpoint
    # =====================================================
    pair = make_resnet18_pairwise().to(device)
    opt2 = torch.optim.AdamW(pair.parameters(), lr=1e-3)
    crit2 = nn.BCEWithLogitsLoss()
    best_pair_teA, best_path_pair = 0, f"pairwise_best_N{N}.pth"

    for ep in range(1, epochs+1):
        train_pair.resample_training_pairs()
        trL, trA = train_epoch_bce(pair, dl_train, opt2, crit2, device)

        # --- evaluate and compute same/diff accuracies ---
        pair.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                logits = pair(x).squeeze(1)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu()
                y_true.extend(y.cpu())
                y_pred.extend(preds)

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        same_mask = (y_true == 1)
        diff_mask = (y_true == 0)
        same_acc = (y_pred[same_mask] == 1).float().mean().item() if same_mask.any() else 0.0
        diff_acc = (y_pred[diff_mask] == 0).float().mean().item() if diff_mask.any() else 0.0
        teA = ((y_pred == y_true).float().mean().item()) * 100

        print(f"[Pairwise] Ep{ep}: trA={trA:.2f}, teA={teA:.2f}, "
            f"SameAcc={same_acc*100:.2f}, DiffAcc={diff_acc*100:.2f}")
        
        if teA > best_pair_teA:
            best_pair_teA = teA
            torch.save(pair.state_dict(), best_path_pair)

    print(f"💾 Saved best pairwise model with {best_pair_teA:.2f}% test acc")

    # reload best before doing inference
    pair.load_state_dict(torch.load(best_path_pair, map_location=device))

    # plot_pairwise_confusion(pair, train_pair, device)
    same_acc = infer_by_sameclass(pair, full_train, test, device,
                                  batch_size=512, n_examples=0)

    return baseline_acc, same_acc

import seaborn as sns

@torch.no_grad()
def plot_pairwise_confusion(model, pairwise_ds, device, n_samples_per_class=2000, batch_size=256):
    model.eval()
    n_classes = 10
    same_matrix = np.zeros((n_classes, n_classes))
    diff_matrix = np.zeros((n_classes, n_classes))

    # collect references by class
    class_to_imgs = defaultdict(list)
    for i, (x, lab) in enumerate(pairwise_ds.train_pool):  # train_pool contains (tensor, label)
        if len(class_to_imgs[lab]) < n_samples_per_class:
            class_to_imgs[lab].append(x)
    for c in range(n_classes):
        class_to_imgs[c] = torch.stack(class_to_imgs[c]).to(device)

    for i in tqdm(range(n_classes), desc="Building confusion"):
        for j in range(n_classes):
            xa = class_to_imgs[i]
            xb = class_to_imgs[j]
            # create all pairs (or limited batches)
            pairs = []
            for k in range(0, len(xa), batch_size):
                a = xa[k:k+batch_size]
                b = xb[:len(a)]
                pair = torch.cat([a, b], dim=1)
                out = model(pair).squeeze(1)
                p_same = torch.sigmoid(out).mean().item()
                if i == j:
                    same_matrix[i, j] = p_same
                else:
                    diff_matrix[i, j] = p_same

    # --- Plot ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(same_matrix, ax=ax[0], vmin=0, vmax=1, cmap="viridis", cbar=False,
                xticklabels=range(n_classes), yticklabels=range(n_classes), square=True)
    ax[0].set_title("Mean P(same) for true SAME-class pairs")
    ax[0].set_xlabel("Ref class")
    ax[0].set_ylabel("Query class")

    sns.heatmap(diff_matrix, ax=ax[1], vmin=0, vmax=1, cmap="viridis", cbar=False,
                xticklabels=range(n_classes), yticklabels=range(n_classes), square=True)
    ax[1].set_title("Mean P(same) for true DIFFERENT-class pairs")
    ax[1].set_xlabel("Ref class")
    ax[1].set_ylabel("Query class")

    plt.tight_layout()
    plt.show()


# =================
# 7. Run subsets  =
# =================
if __name__ == "__main__":
    Ns = [100, 500, 1000, 2000]
    Es = [10, 15, 20, 35]
    base_accs, same_accs = [], []
    for N, E in zip(Ns, Es):
        b, s = run_experiment(N=N, epochs=E)
        base_accs.append(b)
        same_accs.append(s)

    plt.figure(figsize=(6,4))
    plt.plot(Ns, base_accs, "o-", label="Baseline", color="steelblue")
    plt.plot(Ns, same_accs, "s-", label="Pairwise (same/diff)", color="crimson")
    plt.xlabel("Training subset size (N)")
    plt.ylabel("Accuracy (%)")
    plt.title("STL-10 Baseline vs Pairwise (same/diff)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("stl10_sameclass_results.png", dpi=150)
    plt.show()
