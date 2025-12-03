# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import umap
import hdbscan
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ==============================
# Paths 
# ==============================

EMBED_PATH = "/users/ysong25/wafer_project/embeddings_train/train_embeddings_dino_vits14.npy"
DATA_PATH = "/users/ysong25/wafer_project/data/train_unsupervised.pkl"
SAVE_DIR = "/users/ysong25/wafer_project/cluster_results_train_l2"

os.makedirs(SAVE_DIR, exist_ok=True)

# Fix random seed to ensure consistent sampling / initialization
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==============================
# Load data
# ==============================

print("Loading embeddings...")
emb_raw = np.load(EMBED_PATH, allow_pickle=True)

# ---- Key fix: stack object array into a real 2D float array ----
if isinstance(emb_raw, np.ndarray) and emb_raw.dtype == object:
    embeddings = np.vstack(emb_raw)
else:
    embeddings = emb_raw

# Force float32 to avoid dtype issues in linalg
embeddings = embeddings.astype(np.float32)

print("Embedding shape:", embeddings.shape)

print("Loading dataframe...")
df = pd.read_pickle(DATA_PATH)
print("Dataframe shape:", df.shape)

if len(df) != embeddings.shape[0]:
    print("Warning: number of rows in df and embeddings differ")

# ==============================
# Norm diagnostics (before L2)
# ==============================

print("Computing norms before L2...")
norms_before = np.linalg.norm(embeddings, axis=1)

print(
    f"Before L2 norms: mean={np.mean(norms_before):.6f}, "
    f"std={np.std(norms_before):.6f}, "
    f"min={np.min(norms_before):.6f}, "
    f"max={np.max(norms_before):.6f}"
)

plt.figure(figsize=(8, 5))
plt.hist(norms_before, bins=50)
plt.xlabel("L2 norm (before normalization)")
plt.ylabel("Count")
plt.title("Distribution of L2 norms (before L2 normalization)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "norms_before_l2_hist.png"), dpi=300)
plt.close()

# ==============================
# L2 normalization
# ==============================

print("Applying L2 normalization...")
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

# Prevent division by zero: replace 0 with 1
norms_safe = np.where(norms == 0, 1.0, norms)
X = embeddings / norms_safe

norms_after = np.linalg.norm(X, axis=1)
print(
    f"After L2 norms: mean={np.mean(norms_after):.6f}, "
    f"std={np.std(norms_after):.6f}, "
    f"min={np.min(norms_after):.6f}, "
    f"max={np.max(norms_after):.6f}"
)

# Save original norms (in case you want to analyze later)
np.save(os.path.join(SAVE_DIR, "l2_norms_before.npy"), norms_before)
np.save(os.path.join(SAVE_DIR, "l2_norms_after.npy"), norms_after)


# ==============================
# 2D PCA: no-L2 vs L2 comparison
# ==============================

print("Running 2D PCA comparison (no-L2 vs L2)...")

pca_2_nol2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2_nol2 = pca_2_nol2.fit_transform(embeddings)

pca_2_l2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2_l2 = pca_2_l2.fit_transform(X)

# Save 2D PCA results
np.save(os.path.join(SAVE_DIR, "pca2_nol2.npy"), X_pca2_nol2)
np.save(os.path.join(SAVE_DIR, "pca2_l2.npy"), X_pca2_l2)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca2_nol2[:, 0], X_pca2_nol2[:, 1], s=1)
plt.title("PCA 2D on original embeddings (no L2)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.scatter(X_pca2_l2[:, 0], X_pca2_l2[:, 1], s=1)
plt.title("PCA 2D on L2-normalized embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pca_2d_nol2_vs_l2.png"), dpi=300)
plt.close()

# ==============================
# PCA 50D on L2 features
# ==============================

print("Running PCA (50D) on L2-normalized features...")
pca = PCA(n_components=50, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)
print(f"PCA 50D cumulative explained variance: {cum_explained[-1]:.4f}")

np.save(os.path.join(SAVE_DIR, "pca_variance_ratio.npy"), explained)

# Plot PCA variance curve
plt.figure(figsize=(8, 5))
plt.bar(range(1, 51), explained, alpha=0.7, label="per-component")
plt.plot(range(1, 51), cum_explained, marker="o", label="cumulative")
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("PCA explained variance (50 components)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pca_variance_curve.png"), dpi=300)
plt.close()

# ==============================
# UMAP (2D on PCA space)
# ==============================

print("Running UMAP (2D on PCA space)...")
reducer = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric="euclidean",
    random_state=RANDOM_STATE
)
X_umap = reducer.fit_transform(X_pca)
np.save(os.path.join(SAVE_DIR, "umap_2d.npy"), X_umap)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1)
plt.title("UMAP 2D projection (train, no labels)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "umap_train.png"), dpi=300)
plt.close()

# ==============================
# t-SNE (2D on PCA space, subsampled)
# ==============================

print("Running t-SNE (2D, subsampled)...")
n_samples = X_pca.shape[0]
TSNE_MAX_POINTS = 50000  # subsample for t-SNE

if n_samples > TSNE_MAX_POINTS:
    idx_tsne = np.random.choice(n_samples, TSNE_MAX_POINTS, replace=False)
else:
    idx_tsne = np.arange(n_samples)

X_pca_tsne = X_pca[idx_tsne]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    random_state=RANDOM_STATE,
    init="random"
)
X_tsne = tsne.fit_transform(X_pca_tsne)
np.save(os.path.join(SAVE_DIR, "tsne_indices.npy"), idx_tsne)
np.save(os.path.join(SAVE_DIR, "tsne_2d.npy"), X_tsne)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=1)
plt.title("t-SNE 2D projection (train, no labels)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "tsne_train.png"), dpi=300)
plt.close()

# ==============================
# HDBSCAN clustering (on PCA 50D)
# ==============================

print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=10
)
labels_hdb = clusterer.fit_predict(X_pca)
np.save(os.path.join(SAVE_DIR, "labels_hdbscan.npy"), labels_hdb)

unique_hdb = np.unique(labels_hdb)
print("HDBSCAN clusters (including noise):", unique_hdb)

# ==============================
# KMeans (single K = 10 for downstream visualization)
# ==============================

print("Running KMeans (K=10)...")
k_fixed = 10
kmeans = KMeans(n_clusters=k_fixed, random_state=RANDOM_STATE, n_init=10)
labels_km = kmeans.fit_predict(X_pca)
np.save(os.path.join(SAVE_DIR, "labels_kmeans.npy"), labels_km)

unique_km = np.unique(labels_km)
print("KMeans clusters:", unique_km)

# ==============================
# Grid search over K (5–20) for diagnostics
# ==============================

print("Scanning K from 5 to 20 for KMeans metrics...")
k_values = list(range(5, 21))
sil_scores = []
ch_scores = []

for k in k_values:
    print(f"  K = {k} ...")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(X_pca)

    # Silhouette calculation
    if len(np.unique(labels_k)) > 1:
        sil = silhouette_score(X_pca, labels_k)
        ch = calinski_harabasz_score(X_pca, labels_k)
    else:
        sil = np.nan
        ch = np.nan

    sil_scores.append(sil)
    ch_scores.append(ch)

# Save K scan results
k_scan_df = pd.DataFrame({
    "k": k_values,
    "silhouette": sil_scores,
    "calinski_harabasz": ch_scores
})
k_scan_df.to_csv(os.path.join(SAVE_DIR, "kmeans_k_scan_5_20.csv"), index=False)

# Plot two metrics vs K
plt.figure(figsize=(8, 6))
plt.plot(k_values, sil_scores, marker="o")
plt.xlabel("Number of clusters K")
plt.ylabel("Silhouette score")
plt.title("KMeans silhouette score vs K")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "kmeans_silhouette_vs_k.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(k_values, ch_scores, marker="o")
plt.xlabel("Number of clusters K")
plt.ylabel("Calinski-Harabasz score")
plt.title("KMeans Calinski-Harabasz score vs K")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "kmeans_calinski_vs_k.png"), dpi=300)
plt.close()

# ==============================
# Clustering metrics for HDBSCAN & fixed-K KMeans
# ==============================

print("Computing clustering metrics for HDBSCAN & K=10 KMeans...")

# HDBSCAN metrics (exclude noise)
valid_mask_hdb = labels_hdb != -1
if np.sum(valid_mask_hdb) > 2 and len(np.unique(labels_hdb[valid_mask_hdb])) > 1:
    sil_hdb = silhouette_score(X_pca[valid_mask_hdb], labels_hdb[valid_mask_hdb])
    ch_hdb = calinski_harabasz_score(X_pca[valid_mask_hdb], labels_hdb[valid_mask_hdb])
    n_hdb = len(np.unique(labels_hdb[valid_mask_hdb]))
    print(f"HDBSCAN: silhouette={sil_hdb:.4f}, CH={ch_hdb:.4f}, n_clusters={n_hdb}")
else:
    sil_hdb = np.nan
    ch_hdb = np.nan
    n_hdb = len(np.unique(labels_hdb[valid_mask_hdb])) if np.sum(valid_mask_hdb) > 0 else 0
    print("HDBSCAN: not enough valid points for metrics")

# KMeans metrics
if len(np.unique(labels_km)) > 1:
    sil_km = silhouette_score(X_pca, labels_km)
    ch_km = calinski_harabasz_score(X_pca, labels_km)
    n_km = len(np.unique(labels_km))
    print(f"KMeans (K={k_fixed}): silhouette={sil_km:.4f}, CH={ch_km:.4f}, n_clusters={n_km}")
else:
    sil_km = np.nan
    ch_km = np.nan
    n_km = len(np.unique(labels_km))
    print("KMeans: not enough clusters for metrics")

metrics_df = pd.DataFrame({
    "method": ["HDBSCAN", f"KMeans_K_{k_fixed}"],
    "silhouette": [sil_hdb, sil_km],
    "calinski_harabasz": [ch_hdb, ch_km],
    "n_clusters": [n_hdb, n_km]
})
metrics_df.to_csv(os.path.join(SAVE_DIR, "clustering_metrics_summary.csv"), index=False)

# ==============================
# UMAP with labels
# ==============================

print("Plotting UMAP with labels...")

# UMAP + HDBSCAN
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_umap[:, 0], X_umap[:, 1],
    c=labels_hdb,
    s=1,
    cmap="tab20"
)
plt.colorbar(scatter)
plt.title("UMAP + HDBSCAN labels (train)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "umap_hdbscan.png"), dpi=300)
plt.close()

# UMAP + KMeans
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_umap[:, 0], X_umap[:, 1],
    c=labels_km,
    s=1,
    cmap="tab10"
)
plt.colorbar(scatter)
plt.title(f"UMAP + KMeans (K={k_fixed}) labels (train)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "umap_kmeans.png"), dpi=300)
plt.close()

# ==============================
# t-SNE with labels (on subsample)
# ==============================

print("Plotting t-SNE with labels (subsample)...")

labels_hdb_tsne = labels_hdb[idx_tsne]
labels_km_tsne = labels_km[idx_tsne]

# t-SNE + HDBSCAN
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=labels_hdb_tsne,
    s=1,
    cmap="tab20"
)
plt.colorbar(scatter)
plt.title("t-SNE + HDBSCAN labels (train, subsampled)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "tsne_hdbscan.png"), dpi=300)
plt.close()

# t-SNE + KMeans
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=labels_km_tsne,
    s=1,
    cmap="tab10"
)
plt.colorbar(scatter)
plt.title(f"t-SNE + KMeans (K={k_fixed}) labels (train, subsampled)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "tsne_kmeans.png"), dpi=300)
plt.close()

# ==============================
# Sample images per cluster (HDBSCAN)
# ==============================

print("Sampling up to 40 images for each HDBSCAN cluster...")

hdb_dir = os.path.join(SAVE_DIR, "cluster_samples_hdbscan")
os.makedirs(hdb_dir, exist_ok=True)

for label in unique_hdb:
    if label == -1:
        continue

    idxs = np.where(labels_hdb == label)[0]
    if len(idxs) == 0:
        continue

    chosen = np.random.choice(idxs, min(40, len(idxs)), replace=False)

    fig, axes = plt.subplots(8, 5, figsize=(12, 18))
    axes = axes.flatten()

    for i, idx in enumerate(chosen):
        wafer = np.array(df.iloc[idx]["waferMap"]).astype(np.uint8)
        img = Image.fromarray(wafer)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")

    for j in range(len(chosen), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"HDBSCAN cluster {label} (up to 40 samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(hdb_dir, f"hdbscan_cluster_{label}.png"), dpi=300)
    plt.close()

# ==============================
# Sample images per cluster (KMeans)
# ==============================

print("Sampling up to 40 images for each KMeans cluster...")

km_dir = os.path.join(SAVE_DIR, "cluster_samples_kmeans")
os.makedirs(km_dir, exist_ok=True)

for label in unique_km:
    idxs = np.where(labels_km == label)[0]
    if len(idxs) == 0:
        continue

    chosen = np.random.choice(idxs, min(40, len(idxs)), replace=False)

    fig, axes = plt.subplots(8, 5, figsize=(12, 18))
    axes = axes.flatten()

    for i, idx in enumerate(chosen):
        wafer = np.array(df.iloc[idx]["waferMap"]).astype(np.uint8)
        img = Image.fromarray(wafer)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")

    for j in range(len(chosen), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"KMeans cluster {label} (up to 40 samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(km_dir, f"kmeans_cluster_{label}.png"), dpi=300)
    plt.close()

print("All done.")


