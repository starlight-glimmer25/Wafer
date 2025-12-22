# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import time

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, silhouette_score
from scipy.stats import entropy
import umap


# ============================================================
# 0) Configuration
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PCA_DIM = 50
N_GMM = 10  # Number of clusters (K)

# ---- Sampling Config ----
N_SHOW = 40          # Number of samples in grid (8x5)
EXTREME_PERCENT = 0.01
PCA_PC_TO_PLOT = 5   # PC1 to PC5
MIN_POSTERIOR = 0.60

# ---- Paths (Update these for your CRC environment) ----
# Note: Ensure test paths are also defined for benchmarking
TRAIN_EMBED_PATH = "/users/ysong25/wafer_project/embeddings_train/train_embeddings_dino_vits14.npy"
TRAIN_DATA_PATH  = "/users/ysong25/wafer_project/data/train_unsupervised.pkl"

TEST_EMBED_PATH  = "/users/ysong25/wafer_project/embeddings_test/test_embeddings_dino_vits14.npy"
TEST_DATA_PATH   = "/users/ysong25/wafer_project/data/test_unsupervised.pkl"

SAVE_DIR         = "/users/ysong25/wafer_project/gmm_pipeline_full_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Helpers
# ============================================================
def safe_load_embeddings(path: str) -> np.ndarray:
    emb_raw = np.load(path, allow_pickle=True)
    if isinstance(emb_raw, np.ndarray) and emb_raw.dtype == object:
        emb = np.vstack(emb_raw)
    else:
        emb = emb_raw
    return emb.astype(np.float32)

def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)

def plot_wafer_grid(df: pd.DataFrame, idxs: np.ndarray, title: str, outpath: str, n_show: int = 40):
    """Generates an 8x5 grid of wafers with forced resizing to 224x224."""
    if len(idxs) == 0: return
    n = min(n_show, len(idxs))
    chosen = np.random.choice(idxs, n, replace=False)

    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        idx = chosen[i]
        
        wafer = np.array(df.iloc[int(idx)]["waferMap"]).astype(np.uint8)

        axes[i].imshow(wafer, cmap="gray")
        axes[i].axis("off")

    for j in range(n, len(axes)): axes[j].axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_scatter(X2: np.ndarray, c: np.ndarray, title: str, outpath: str, cmap: str = "tab10", s: int = 1):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=c, s=s, cmap=cmap, alpha=0.6)
    plt.colorbar(sc)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ============================================================
# 1) Load & Prepare Training Data
# ============================================================
print("Step 1: Loading Training data...")
train_emb = safe_load_embeddings(TRAIN_EMBED_PATH)
train_df = pd.read_pickle(TRAIN_DATA_PATH)
X_train = l2_normalize(train_emb)

# ============================================================
# 2) PCA Dimensionality Reduction
# ============================================================
print("Step 2: Running PCA...")
pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)

# ============================================================
# 3) GMM (Probabilistic Modeling)
# ============================================================
print(f"Step 3: Fitting GMM (K={N_GMM})...")
gmm = GaussianMixture(n_components=N_GMM, covariance_type="full", random_state=RANDOM_STATE, n_init=3)
gmm.fit(X_train_pca)
joblib.dump(gmm, os.path.join(SAVE_DIR, "gmm_model.pkl"))

# ============================================================
# 4) UMAP Visualization
# ============================================================
print("Step 4: Generating UMAP Visualization...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
X_umap = reducer.fit_transform(X_train_pca)
labels = gmm.predict(X_train_pca)
save_scatter(X_umap, labels, "UMAP: GMM Clusters", os.path.join(SAVE_DIR, "umap_clusters.png"))

# ============================================================
# 5) FINAL BENCHMARK (The Evaluation "Tricks")
# ============================================================
print("\nStep 5: Starting Performance Benchmark (Training vs. Testing)...")

# Load Test data for comparison
test_emb = safe_load_embeddings(TEST_EMBED_PATH)
test_df = pd.read_pickle(TEST_DATA_PATH)
X_test_pca = pca.transform(l2_normalize(test_emb))

eval_sets = [('Train', X_train_pca, train_df), ('Test', X_test_pca, test_df)]
performance_results = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for set_name, data_pca, data_df in eval_sets:
    print(f"Evaluating {set_name} Set...")
    
    # Anomaly Scores: Negative Log-Likelihood
    # Higher scores indicate data points far from GMM cluster centers (Potential Defects)
    scores = -gmm.score_samples(data_pca)
    
    # Ground Truth: 'none' labels -> 0 (Normal), All other failure types -> 1 (Defect)
    y_true = (data_df['failureType_flat'] != 'none').astype(int)
    
    # Metric 1: ROC-AUC (Measures ranking ability against human labels)
    roc_auc = roc_auc_score(y_true, scores)
    
    # Metric 2: PR-AUC (Measures defect detection precision in imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # Metric 3: Silhouette Score (Measures mathematical cluster separation)
    # Subsampled at 5000 for computational efficiency
    sample_idx = np.random.choice(len(data_pca), min(5000, len(data_pca)), replace=False)
    s_score = silhouette_score(data_pca[sample_idx], gmm.predict(data_pca[sample_idx]))
    
    performance_results.append({
        "Dataset": set_name,
        "ROC-AUC": f"{roc_auc:.4f}",
        "PR-AUC": f"{pr_auc:.4f}",
        "Silhouette": f"{s_score:.4f}"
    })

    # Plot Curves for the Test Set
    if set_name == 'Test':
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax1.plot(fpr, tpr, color='darkorange', label=f'DINOv2 (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='navy')
        ax1.set_title('Test Set: ROC Curve')
        ax1.legend()
        
        ax2.plot(recall, precision, color='forestgreen', label=f'PR-AUC = {pr_auc:.3f}')
        ax2.set_title('Test Set: Precision-Recall Curve')
        ax2.legend()

# Save Benchmark Graphics
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'final_performance_curves.png'))

# Final Report Table
final_report = pd.DataFrame(performance_results)
print("\n" + "="*60)
print("             FINAL DINOv2+GMM BENCHMARK REPORT")
print("="*60)
print(final_report.to_string(index=False))
print("="*60)

print(f"\nAll Done! Pipeline results saved in: {SAVE_DIR}")
