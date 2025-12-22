# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
import umap
from PIL import Image

# ============================================================
# 0) Configuration
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PCA_DIM = 50
N_GMM = 10  # Number of clusters (K)

# ---- Sampling Config ----
N_SHOW = 40          # Number of samples to show in each grid (8x5)
EXTREME_PERCENT = 0.01
CENTRAL_BAND = 0.05
PCA_PC_TO_PLOT = 5   # PC1 to PC5

CORE_PERCENT = 0.05
BULK_BAND = (0.45, 0.55)
BOUNDARY_PERCENT = 0.05
MIN_POSTERIOR = 0.60

# ---- Paths ----
EMBED_PATH = "/users/ysong25/wafer_project/embeddings_train/train_embeddings_dino_vits14.npy"
DATA_PATH  = "/users/ysong25/wafer_project/data/train_unsupervised.pkl"
SAVE_DIR   = "/users/ysong25/wafer_project/gmm_pipeline_full_v2"

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
    """Generates an 8x5 grid of wafer maps."""
    if len(idxs) == 0:
        return
    n = min(n_show, len(idxs))
    chosen = np.random.choice(idxs, n, replace=False)

    # 8 rows x 5 columns = 40 images
    cols = 5
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        idx = chosen[i]
        wafer = np.array(df.iloc[int(idx)]["waferMap"]).astype(np.uint8)
        axes[i].imshow(wafer, cmap="gray")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outpath, dpi=200) # Slightly lower DPI to save time and disk space
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
# 1) Load & Normalize
# ============================================================
print("Step 1: Loading data...")
emb = safe_load_embeddings(EMBED_PATH)
df = pd.read_pickle(DATA_PATH)
X = l2_normalize(emb)

# Save Norm Distribution for sanity check
norms_before = np.linalg.norm(emb, axis=1)
plt.hist(norms_before, bins=100)
plt.title("L2 Norms Before Normalization (Bell Curve Check)")
plt.savefig(os.path.join(SAVE_DIR, "norms_distribution.png"))
plt.close()

# ============================================================
# 2) PCA & Interpretability
# ============================================================
print("Step 2: Running PCA...")
pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

pca_diag_dir = os.path.join(SAVE_DIR, "pca_diagnostics")
os.makedirs(pca_diag_dir, exist_ok=True)

print("Sampling PCA diagnostic grids (40 samples per PC)...")
for pc_idx in tqdm(range(PCA_PC_TO_PLOT)):
    scores = X_pca[:, pc_idx]
    order = np.argsort(scores)
    
    # Low / Mid / High sampling
    k_ext = max(1, int(len(scores) * EXTREME_PERCENT))
    low_pool = order[:k_ext]
    high_pool = order[-k_ext:]
    
    plot_wafer_grid(df, low_pool, f"PC{pc_idx+1} Low Extreme", os.path.join(pca_diag_dir, f"pc{pc_idx+1}_low.png"), n_show=N_SHOW)
    plot_wafer_grid(df, high_pool, f"PC{pc_idx+1} High Extreme", os.path.join(pca_diag_dir, f"pc{pc_idx+1}_high.png"), n_show=N_SHOW)

# ============================================================
# 3) GMM (Probabilistic Modeling)
# ============================================================
print(f"Step 3: Fitting GMM (K={N_GMM})...")
gmm = GaussianMixture(n_components=N_GMM, covariance_type="full", random_state=RANDOM_STATE, n_init=3)
gmm.fit(X_pca)

probs = gmm.predict_proba(X_pca)
labels = np.argmax(probs, axis=1)
# Entropy calculation for transition states
ent = entropy(probs.T) / np.log(N_GMM) 
# Log-likelihood for core/boundary sampling
logp_cond = gmm._estimate_log_prob(X_pca) 

# ============================================================
# 4) UMAP (Visualization)
# ============================================================
print("Step 4: Running UMAP (This may take 15-20 mins for 80k samples)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
X_umap = reducer.fit_transform(X_pca)

save_scatter(X_umap, labels, "UMAP: GMM Clusters", os.path.join(SAVE_DIR, "umap_clusters.png"), cmap="tab10")
save_scatter(X_umap, ent, "UMAP: Transition Entropy (Bright=Mixed)", os.path.join(SAVE_DIR, "umap_entropy.png"), cmap="magma")

# ============================================================
# 5) Component Dissection (Core / Bulk / Boundary)
# ============================================================
print("Step 5: Sampling Core/Bulk/Boundary for each cluster...")
gmm_rep_dir = os.path.join(SAVE_DIR, "gmm_dissection")
os.makedirs(gmm_rep_dir, exist_ok=True)

for k in tqdm(range(N_GMM)):
    # Filter by posterior confidence
    cand = np.where(probs[:, k] >= MIN_POSTERIOR)[0]
    if len(cand) < N_SHOW: cand = np.where(labels == k)[0] # Fallback
    
    if len(cand) == 0: continue
    
    # Sort candidates by conditional log-likelihood p(x|k)
    s = logp_cond[cand, k]
    order = np.argsort(s)
    
    core_idx = cand[order[-N_SHOW:]]     # Top density
    boundary_idx = cand[order[:N_SHOW]]  # Low density
    
    comp_dir = os.path.join(gmm_rep_dir, f"cluster_{k}")
    os.makedirs(comp_dir, exist_ok=True)
    
    plot_wafer_grid(df, core_idx, f"Cluster {k}: CORE (Textbook Samples)", os.path.join(comp_dir, "core.png"), n_show=N_SHOW)
    plot_wafer_grid(df, boundary_idx, f"Cluster {k}: BOUNDARY (Transition/Outliers)", os.path.join(comp_dir, "boundary.png"), n_show=N_SHOW)

print(f"\nAll Done! Please check results in: {SAVE_DIR}")
