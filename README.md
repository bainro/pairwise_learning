# Pairwise Learning for Computer Vision and MRI

Research code accompanying **Pairwise Learning Increases Data Efficiency in Deep Neural Networks** (Bain, Rashidi, and Chrastil, 2026, submitted to *Machine Learning with Applications*).

Instead of predicting an absolute value from one image, a pairwise model learns a relative comparison between two images. Comparisons can then be combined to estimate continuous values. The repository also includes image classification and structural MRI experiments.

## Scripts

| Script | Purpose |
| --- | --- |
| `age_ranker.py` | UTKFace age prediction with pairwise CNN training and entropy-based inference. |
| `age_ranker_BST.py` | UTKFace inference using age-bin binary search and local refinement. |
| `stl10_pairwise_vs_baseline.py` | Compare pairwise same-class supervision with a standard STL-10 classifier at several label budgets. |
| `T1_pairwise_regression.py` | Slice-based CNN features and pairwise ranking on structural brain MRI. |
| `T1_mse_regression.py` | Pointwise MRI regression baseline for comparison. |

## Getting started

Use Python 3 with compatible PyTorch and torchvision versions. The main dependencies are NumPy, SciPy, pandas, Pillow, matplotlib, tqdm, and `kagglehub`. The BST and STL-10 experiments also use seaborn. A CUDA-capable GPU is recommended for training.

- **UTKFace:** The age scripts retrieve images with `kagglehub`. Training functions are defined in the scripts, while the current `__main__` sections may expect locally saved model checkpoints. Adjust the paths and choose the desired training or inference entry point before running.
- **STL-10:** Run `python stl10_pairwise_vs_baseline.py` to download STL-10 through torchvision and compare the two training approaches. The default script runs several dataset sizes.
- **Structural MRI:** Provide your own appropriately prepared MRI data and labels. The MRI scripts reference a local `./indivs_192/` dataset and saved model weights. Update those paths for your environment.

`requirements.txt` is an environment snapshot and includes machine-specific paths. It is not a portable installation lockfile.

## Research notes

The scripts are research examples with experiment-specific settings, data paths, and checkpoints. Review dataset partitions and reference selection when evaluating on new data.

**Paper:** [Read the manuscript (PDF)](https://rkbain.com/pairwise.pdf).
