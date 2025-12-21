## Part 1: Problem Overview
1.1 Introduction
This project focuses on developing an unsupervised anomaly detection system for semiconductor wafer maps. In the semiconductor industry, even minor undetected defects on wafers can result in severe yield loss, especially when the issue is only discovered after dozens or even hundreds of wafers have already been manufactured and discarded. 

While existing research, such as supervised CNN-based classification models (e.g., by Stanford teams), have shown strong performance in controlled environments, they often fail in real-world industrial use cases when confronted with novel, 'zero-shot' defect types not present in the training data.. This is primarily because real-world manufacturing defects are frequently novel or previously unseen, making supervised methods less effective due to the lack of labeled examples for rare anomalies.

Our goal is to design a system that can leverage patterns from known “good” wafers to identify previously unseen or subtle anomalies early in the process. Ideally, such a system could help prevent large-scale quality loss by detecting potential yield-impacting anomalies as soon as they emerge.

This project is directly inspired by my firsthand experience with yield loss challenges during my tenure as a Manufacturing Quality Engineer at TSMC. It aims to bridge the gap between academic research and industrial application by developing a practical solution for early detection of novel defects.It not only fulfills the course requirement but also provides a valuable opportunity to build a research-grade project that may enhance my resume and career prospects. If the work yields meaningful or novel results, I would also be interested in exploring publication opportunities in relevant industrial or computer vision venues.

1.2 Proposed High-Level Solution 

This project aims to build an unsupervised anomaly detection system for semiconductor wafer images. Unlike supervised learning, which focuses on classifying known defect types, our goal is to rely solely on normal samples, train the model to learn the structural patterns of normal wafers, and then identify images that deviate from these patterns as potential anomalies. This approach is tailored to the real-world industrial environment where unknown defects are frequent but labels are scarce.

While many different types of defects exist in the actual wafer production process, they often share certain common characteristics: disruptions to the symmetry, repetitiveness, or spatial structure of the wafer image; localized brightness abrupt changes, texture breaks, and discontinuous patterns. While these characteristics are visually perceptible, they are often difficult to formalize using handcrafted rules. This is where machine learning, particularly representation learning methods, lies in their value.

We have not yet finalized the specific algorithms we will use, but our initial focus is on unsupervised/semi-supervised learning to ensure that they are as close to real-world industrial applications as possible. In terms of method selection, we may employ a denoising autoencoder, a structure more sensitive to anomalies. Through an "input-reconstruction" mechanism, it can improve the model's ability to detect subtle structural deviations while learning from normal wafers. We may also explore strategies such as principal component analysis (PCA) or clustering to construct a representation space for normal samples and identify anomalies based on distance or density.

Notably, we require a model mechanism that can "see through phenomena to the essence" to address the more severe risk distribution characteristics of industrial production: the cost of false negatives (missed detections) far outweighs the cost of false positives (false alarms). Therefore, our primary objective is to maximize recall (minimize false negatives), even at the cost of a higher false positive rate. The operational threshold will be tuned via ROC curve analysis to find an acceptable balance for a hypothetical production line. During the model design and training phases, we prioritize maximizing recall to ensure that no potential defects are missed. Furthermore, to demonstrate the balance of the model's overall performance, we will also report precision and false positive rates in our experiments, and plot receiver operating characteristic (ROC) curves to provide a more comprehensive evaluation metric reference.

The output of this system may include: (1) an anomaly score for each image; (2) a residual map highlighting abnormal areas; or (3) a binary classification judgment (normal vs. abnormal) based on a score threshold. We will further test different output methods and threshold selection strategies in subsequent experiments.

In addition, we hope that the model has a certain degree of invariance: for example, it is insensitive to slight brightness changes, geometric rotations, or small translations of the image. This helps to reduce false positives and allows the model to focus on anomalies that are truly structurally destructive rather than triggering alarms due to irrelevant disturbances.

[Current statement] : Since this project is still in the exploratory stage, the specific model structure, feature extraction method, anomaly score calculation mechanism, and final output form are still under investigation and may be dynamically adjusted based on actual results. We will also continue to evaluate the degree of match between existing public datasets (such as WM-811K) and real industrial scenarios, and consider supplementing other data or fine-tuning the objectives when necessary.

------------------------------Part 2: Data Collection and Description--------------------------------------------------
2.1 Data Source
Dataset: WM-811K Wafer Map Dataset
Download Link: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map (Primary source)

Associated Paper:
The WM-811K dataset is prominently featured and analyzed in the following foundational research paper:
Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets. IEEE Transactions on Semiconductor Manufacturing, 28(1), 1–12.

2.2 Dataset Splits
Given the industrial motivation and unsupervised nature of this project, the dataset is split as follows:

Training Set: 60% (103,770 samples) - Used for feature extraction and clustering algorithm development.

Validation Set: 20% (34,590 samples) - Used for parameter tuning and intermediate evaluation of clustering stability.

Test Set: 20% (34,590 samples) - Reserved for final evaluation and not used during development phases.

The stratified split ensures that each subset maintains the same proportion of "none" class samples (approximately 80%) and defect category distributions as the original dataset, providing representative samples for each phase of the project.
Important Differences Between Training and Validation Subsets

From the project perspective, key differences include:

Purpose: Training set drives unsupervised pattern discovery; validation set tests generalization of discovered patterns and supports hyperparameter tuning.

Defect Representation: Validation set contains a controlled proportion of defects to evaluate anomaly detection performance, while training set is dominated by normal samples to learn normal pattern distribution.

Evaluation Focus: Training evaluation focuses on clustering quality; validation evaluation tests stability with unseen data.

2.3 Data Statistics
Total Samples: 172,950 wafer maps (correcting the initial 800,000 estimate - the WM-811K contains approximately 172,950 samples)

Distinct Defect Categories: 9 + "none" class (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, None)

2.4 Class Distribution:

None (normal): ~80% (138,360 samples)

Defect categories: 300-5,000 samples each (highly imbalanced)

Sample Characterization
Resolution: 256×256 pixels (after preprocessing; original images may vary but are normalized to this resolution)

Data Type: Grayscale images, often binarized (0 = normal, 1 = defect) in practice

Sensor: Optical inspection tools used in semiconductor manufacturing lines

Illumination Wavelength: Not explicitly specified in dataset documentation, but typical wafer inspection systems use visible to ultraviolet light

Ambient Conditions: Cleanroom environment (temperature/humidity controlled) as typical in semiconductor fabrication

Manufacturing Context: Real-world production data with inherent noise and variability

2.5 Data Quality and Challenges
Class Imbalance: Severe imbalance with normal samples dominating (~80%), reflecting real-world distributions but increasing risk of bias.

Label Uncertainty: The "none" class may contain unlabeled or subtle defects, requiring robust anomaly detection rather than supervised classification.

Transitional States: Many samples exhibit mixed or ambiguous patterns, existing between clear defect categories.

Subtle Defect Cues: Some defect categories exhibit very subtle visual cues, requiring highly sensitive structural representations.

2.6 Additional Data Exploration
In addition to using WM-811K, I have explored access to authentic wafer data through industry contacts and university research labs. If granted, such datasets would provide realistic deployment validation. However, for this project phase, WM-811K serves as the primary dataset.

2.7 Data Acquisition Confirmation
I have downloaded the complete WM-811K dataset and preprocessed it for our pipeline. All data is stored locally and ready for development.
---------------------------------------Part3---First Update----------------------------------------------
Part 3: Data Preprocessing, Segmentation and Feature Extraction
3.1. Data Preprocessing
This project uses an open wafer map defect detection dataset, where each sample is a 2-D matrix and each pixel value corresponds to a chip region’s yield status.
Preprocessing steps include:
	Reading wafer maps from the DataFrame and converting them to NumPy arrays.
	Normalizing pixel values to the range 0,255to work with OpenCV operations.
	Removing empty or invalid entries to ensure valid image data.
These operations stabilize segmentation and standardize inputs for feature extraction.

3.2. Image Segmentation
Two segmentation strategies were tested:
(a) Otsu Adaptive Thresholding
Otsu’s method computes a global threshold automatically, but performed poorly because the wafer images are brightness-uniform — it often classified the entire wafer as foreground, losing defect information.
(b) Manual Fixed Thresholding
A fixed threshold was applied and compared across values:
	100 → wafer region disappeared (blank image).
	150 → wafer shape retained but fine defects lost.
	128 → best contrast and preserved circular structure.
Thus, threshold = 128 was selected as the standard setting — efficient, interpretable, and CPU-friendly.

3.3 Feature Extraction
We extracted twelve classical, interpretable features from each segmented wafer map. These features are designed to capture both spatial and structural properties of wafer defects in a way that does not rely on training or labeling, making them ideal for unsupervised or semi-supervised anomaly detection.

(1) Defect ratio — the proportion of defective pixels relative to the total wafer area, representing the overall defect intensity.
(2) Number of regions — the count of connected defect components, indicating how fragmented or clustered the defects are.
(3) Average region area — the typical size of connected defect regions, useful for distinguishing local point defects from large continuous failures.
(4) Maximum region area — captures extreme cases such as near-full or large ring-type defects.
(5) Centroid_x — the normalized horizontal coordinate of the overall defect center, showing whether the defects are left/right biased.
(6) Centroid_y — the normalized vertical coordinate of the defect center, showing top/bottom positional bias.
(7) Radial mean — the mean normalized distance of defect pixels from the wafer center, distinguishing center-type versus edge-type patterns.
(8) Radial standard deviation — measures how uniformly defects are distributed across the radius; low values imply concentrated defects, while high values suggest scattered patterns.
(9) Edge ratio — the fraction of defect pixels located near the wafer’s periphery, sensitive to edge-ring or edge-loc defects.
(10) Left–right symmetry — the IoU-based similarity between the mask and its horizontal flip, indicating structural symmetry across the wafer’s vertical axis.
(11) Top–bottom symmetry — the IoU-based similarity between the mask and its vertical flip, capturing horizontal symmetry or imbalance.
(12) Edge density — the proportion of strong edges detected using the Canny operator, highly responsive to linear or scratch-like defects.

Justification:
These twelve features were selected to achieve a balance between physical interpretability and computational simplicity.
They represent multiple perspectives of wafer quality — intensity, geometry, location, and symmetry — all of which are crucial in real semiconductor manufacturing defect analysis.
Compared to deep-learning-based embeddings (e.g., DINOv3 or Autoencoder features), these hand-crafted features have the advantages of being fully explainable, requiring no GPU resources, and being feasible for quick iteration in the early phase of the project.
In future phases, these classical descriptors can serve as a baseline for validating the improvements achieved by deep visual representations.

【Example Photo】<img width="1858" height="926" alt="image" src="https://github.com/user-attachments/assets/0b6eeb77-870f-4a1d-963c-d446380db4d6" />


3.4. Methodology Justification
We adopted a traditional segmentation + feature pipeline because:
	The goal is to build an interpretable, lightweight baseline before introducing deep models.
	Classical methods are training-free, GPU-independent, and fast on CPUs.
	Extracted features have direct physical meaning aligned with wafer defect morphology.

3.5. Future Work and Optimization Directions
(1) Integration of Gabor Wavelet Features
Gabor filters are frequency- and orientation-selective kernels capable of capturing fine texture and directional patterns such as scratches or edge-rings. By building a multi-scale, multi-orientation Gabor filter bank and extracting energy-based statistics, we can add richer local descriptors to the current global statistical features. This extension would serve as a CPU-friendly short-term upgrade that enhances sensitivity to low-contrast directional defects.
(2) High-Level Representation via DINOv3
We also plan to apply DINOv3, a self-supervised vision transformer, to extract semantic embeddings from wafer images. Current hardware (Intel Arc GPU without CUDA) prevents local execution, but future GPU access on CRC clusters will allow testing. Comparing DINOv3 embeddings with classical and Gabor features will bridge interpretability and representation power.
Together, these two paths outline a clear roadmap for enhancing this baseline pipeline from statistical to hybrid feature learning.

3.6. Summary
The current stage completes a working pipeline from data preprocessing to segmentation and feature extraction. Manual threshold = 128 preserves wafer geometry while isolating defects, and the extracted features are well-suited for subsequent analysis.
Future tasks will include GPU-based DINOv3 testing, Gabor feature integration, and clustering/classification experiments for automatic defect categorization.
Additional Discussion — Pipeline Limitations and Rationale

This stage of the project primarily focuses on establishing a fully functional end-to-end pipeline, rather than optimizing ultimate detection performance.
The manual-threshold segmentation (fixed at 128) successfully isolates bright defect pixels while preserving the circular wafer geometry, serving as a transparent and interpretable baseline.
However, several inherent limitations remain:
Low-contrast defects — When defect brightness is close to the wafer background, they are often missed because a single global threshold lacks sensitivity.
Structural patterns — Continuous defects such as rings or scratches may be fragmented, as global thresholding is not geometry-aware.
Noise and minor artifacts — Isolated bright pixels can be incorrectly detected as defects without post-processing or morphological filtering.
Limited generalization — The method performs reliably only under uniform lighting and clearly defined contrast conditions, which may not hold for real-world manufacturing data.
Despite these shortcomings, this baseline is valuable because it verifies the data-processing flow, enables quantitative feature extraction, and provides a benchmark for future enhancement. Subsequent stages will incorporate more advanced techniques—such as Gabor-wavelet texture descriptors and self-supervised DINOv3 embeddings—to overcome the above weaknesses and achieve robust, semantically meaningful defect detection.

***Additional Discussion — Pipeline Limitations and Rationale

This stage of the project primarily focuses on establishing a fully functional end-to-end pipeline, rather than optimizing ultimate detection performance.
The manual-threshold segmentation (fixed at 128) successfully isolates bright defect pixels while preserving the circular wafer geometry, serving as a transparent and interpretable baseline.
However, several inherent limitations remain:

Low-contrast defects — When defect brightness is close to the wafer background, they are often missed because a single global threshold lacks sensitivity.

Uneven illumination — Bright or dark regions may lead to false positives or false negatives, since the current method does not adapt to local intensity variation.

Structural patterns — Continuous defects such as rings or scratches may be fragmented, as global thresholding is not geometry-aware.

Noise and minor artifacts — Isolated bright pixels can be incorrectly detected as defects without post-processing or morphological filtering.

Limited generalization — The method performs reliably only under uniform lighting and clearly defined contrast conditions, which may not hold for real-world manufacturing data.

Despite these shortcomings, this baseline is valuable because it verifies the data-processing flow, enables quantitative feature extraction, and provides a benchmark for future enhancement.
Subsequent stages will incorporate more advanced techniques—such as Gabor-wavelet texture descriptors and self-supervised DINOv3 embeddings—to overcome the above weaknesses and achieve robust, semantically meaningful defect detection.

@@@ Run Instructions @@@
1. Environment Setup

Before running the project, please make sure the following dependencies are installed:

pip install numpy pandas matplotlib opencv-python scikit-learn joblib


If using Jupyter Notebook (recommended), execute the cells sequentially to visualize intermediate results.

2. Data Preparation

This project uses the WM-811K (LSWMD) wafer defect dataset.

The original .pki file is in a legacy format; conversion and compatibility are already handled automatically at the beginning of the notebook.

Please place the downloaded file LSWMD.pki inside the archive/ folder under the project root directory.

Dataset download link (Kaggle official):
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

3. Run the Main Notebook

Open and execute the following notebook:

jupyter notebook wafer-part-3-update.ipynb


Alternatively, open it directly in VSCode or JupyterLab and run all cells in order.

The notebook automatically performs the following steps:

Loads and converts the legacy .pki data file.

Performs data preprocessing and cleaning (removing invalid samples, normalizing pixel values).

Conducts wafer image segmentation using a manual threshold of 128.

Extracts 12 classical, interpretable features (e.g., defect ratio, centroid, symmetry, edge density).

Outputs a structured features_df DataFrame for later clustering or classification analysis.

4. Output Results

features_df.csv (or DataFrame): contains 12 extracted features per wafer sample.

Visualization outputs (matplotlib plots): original wafer and defect overlay in soft pink/green colors.

--------------------Dec.3 rd Updated-Part 4: Final Report (Unsupervised Clustering Pipeline with Dino)---------------------------
4.1 Overview and Motivation
Our goal in this project is to explore wafer map anomaly patterns using an unsupervised pipeline. Since the dataset contains many uncertain or transitional states, and since a large portion of samples do not have reliable labels, a direct supervised classifier would not be appropriate. For the same reason, methods like (one-class) SVM are also not suitable here. In this dataset, it is extremely hard to define a consistent and 100% reliable , "normal" region or decide which wafers should be treated as clean references. Many samples fall into gray areas, and any decision about what counts as "normal" or "abnormal" would be very subjective. Instead, we focus on learning meaningful representations and grouping wafers by visual similarity.
To do this, we build a pipeline based on DINO ViT (S-14) embeddings, dimensionality reduction, and clustering. The final output is a set of clusters that reflect structural patterns across the dataset.

4.2 Choice of Representation and Clustering Method
Why we chose DINO ViT
We selected the DINO ViT-S/14 (384-dimensional) vision transformer for feature extraction. There are a few reasons for this choice:
1.	Self-attention structure – ViT models are good at capturing global relationships across the wafer map, which is important for understanding holistic defect shapes and spatial patterns.
2.	Lower dimensional backbone is enough – DINO is trained on natural images, which have far more visual variability than wafer maps. Our dataset is significantly more structured and homogeneous, so a smaller embedding size (384) is not only sufficient but also helps avoid unnecessary model complexity and reduces the risk of overfitting. Stable and expressive embeddings – Even without fine-tuning, DINO produces embeddings that already separate wafers with different high-level patterns.
Overall, DINO provides a strong and consistent representation, especially compared to our earlier autoencoder baseline.
Why PCA + UMAP
Wafer embeddings live in a high-dimensional space, but from our earlier experiments (handcrafted features and autoencoder latent space), we already had some intuition that the actual data structure is low-dimensional. PCA confirmed this: only the first ~10–15 principal components explain most of the variance (80%).
Therefore, after L2 normalization, we apply:
(1)	PCA – reduces noise and keeps global structure
(2)	UMAP – captures local neighborhood relations and gives an interpretable 2D/3D manifold
This combination gives us a clearer view of the dataset and helps clustering algorithms behave more reliably.
Why K-Means and HDBSCAN
We experimented with two clustering approaches:
HDBSCAN
HDBSCAN naturally pairs well with UMAP, and it automatically discovers cluster shapes without requiring a fixed k.
However, in practice, the performance was inconsistent, possibly due to parameter sensitivity or instability in noisy regions of the dataset.
K-Means
Our initial plan was to use Spherical K-Means, because cosine similarity matches the geometry of DINO embeddings. Unfortunately, the Python spherical K-Means package is no longer maintained, several versions are incompatible with newer environments, and CRC's environment made multiple installation risky. (We will continue to refine and improve this aspect moving forward.) So we switched to the standard K-Means implementation. To partially compensate, we apply L2 normalization before clustering, so Euclidean distance becomes aligned with cosine similarity.
4.3 Evaluation on Training and Validation Sets
Since this is an unsupervised task, we evaluate cluster quality using internal metrics instead of accuracy.
Quantitative Results:
HDBSCAN achieved a silhouette score of 0.275 and a Calinski–Harabasz index of 1022.808, but it only found 2 clusters (excluding noise), which is too coarse for our goal of discovering nuanced defect patterns. After further analysis, we believe this behavior likely stems from the choice of parameters as well as the presence of substantial continuous or transitional states within the dataset. These issues will be examined in detail and addressed in future algorithmic improvements.
K-Means with K=10 achieved a silhouette score of 0.116 and a Calinski–Harabasz index of 8263.433, with 10 clusters. While the silhouette score is lower than HDBSCAN's, the Calinski–Harabasz index is much higher, indicating better between-cluster separation.
We performed a grid search over K values from 5 to 20 for K-Means. The results show that the optimal K in terms of silhouette score is 5 (0.1326), but we chose K=10 to capture more granular patterns, as the Calinski–Harabasz index remains high. The complete K scan results are:
K=5: silhouette=0.1326, Calinski–Harabasz=12226.69
K=6: silhouette=0.1237, Calinski–Harabasz=11243.54
K=7: silhouette=0.1135, Calinski–Harabasz=10233.53
K=8: silhouette=0.1151, Calinski–Harabasz=9459.75
K=9: silhouette=0.1191, Calinski–Harabasz=8791.81
K=10: silhouette=0.1163, Calinski–Harabasz=8263.43
K=11: silhouette=0.1178, Calinski–Harabasz=7835.41
K=12: silhouette=0.1138, Calinski–Harabasz=7415.77
K=13: silhouette=0.1127, Calinski–Harabasz=7024.21
K=14: silhouette=0.1033, Calinski–Harabasz=6667.07
K=15: silhouette=0.1079, Calinski–Harabasz=6371.15
K=16: silhouette=0.1074, Calinski–Harabasz=6125.10
K=17: silhouette=0.1092, Calinski–Harabasz=5892.29
K=18: silhouette=0.1007, Calinski–Harabasz=5695.39
K=19: silhouette=0.0968, Calinski–Harabasz=5494.37
K=20: silhouette=0.0988, Calinski–Harabasz=5333.73
We also compared the cluster structure between the training portion and a separate validation subset. The validation manifold shows similar overall shapes but with more mixed or noisy regions, which is expected given the transitional states present in the dataset.

4.4 Observations and Commentary
Data characteristics
The wafer dataset is challenging for several reasons:
1.	Long-tail distribution – some defect types are extremely rare, while the majority of samples fall into ambiguous or unlabeled categories.
2.	Many transitional or hybrid states – lots of wafers show partial defects or noisy patterns that do not cleanly belong to a single category.
3.	Real-world messy data – this dataset reflects actual manufacturing behavior, so patterns are not cleanly separated.
Cluster behavior
1.	K-Means tends to draw arbitrary boundaries in unclear regions. Some clusters represent true patterns, while others split what should be a single "mixed-type" region.
2.	Embedding quality is good, but difficult samples still create overlapping areas in the manifold.
3.	Due to environment constraints, we were not able to generate attention visualizations from DINO, which might have helped with interpretability. If time allows, we plan to add these later.
The KMeans clustering (K=10) reveals ten groups that, while not perfectly separable, still exhibit somewhat observable visual tendencies. These tendencies mostly reflect differences in defect density, spatial uniformity, and subtle directional or radial structure rather than strongly defined anomaly types. This is consistent with the nature of the WM-811K "none" class, where many wafers exist in mixed states and do not present clean textbook failure patterns.
Cluster 0 consists of wafers with low-density, evenly scattered defects. They show no directional or radial bias and likely represent baseline noise.
Cluster 1 shows slightly stronger activity near one side of the edge, resembling a very weak partial-edge signature.
Cluster 2 maintains random scatter but with a noticeably higher density than Cluster 0, forming a "medium-density random" category.
Cluster 3 continues this trend with even heavier global scatter, forming a dense, snow-like texture across the wafer.
Cluster 4 exhibits a soft degree of center concentration—still scattered, but with a mild center-weighted tendency.
Cluster 5 contains wafers with medium-to-high density and occasional local patches of elevated activity. Although not forming a coherent pattern, these patches introduce mild regional structure.
Cluster 6 is visually smoother and directionally more uniform, representing a transitional group between low-density and moderate-density random wafers.
Cluster 7 includes wafers with slight anisotropy or oval-like spatial distortion. Some samples show weak directional stretching, which DINO embeddings are particularly sensitive to.
Cluster 8 contains high-density wafers that sometimes display short bursts of regional overkill. These subtle high-intensity areas differentiate them from the more uniformly dense Cluster 3.
Cluster 9 is characterized by mid-range defect counts combined with a high amount of random black pixel noise. The absence of directional or spatial concentration suggests that this group captures wafers influenced by distributed, low-specificity defect mechanisms rather than systematic structural faults.
Overall, the clusters demonstrate a smooth continuum rather than sharp boundaries. Many clusters primarily differ by defect intensity rather than specific shape, while others reflect more localized or directional tendencies. This distribution supports the interpretation that the dataset contains a large number of mixed states, and that unsupervised clustering on DINO embeddings naturally organizes wafers along gradients of density and subtle structural cues rather than well-defined anomaly categories. The presence of weak but consistent tendencies across clusters indicates that the embedding model successfully captures meaningful spatial information despite the inherently noisy and heterogeneous nature of the data.

4.5 Ideas for Improvement
There are several directions to improve this pipeline:
1.	Use flow-based models (normalizing flows): Because flows learn an invertible mapping between the embedding space and a simple Gaussian density, they can reshape a complex high-dimensional manifold into a more regular and separable form. This reversible transformation effectively straightens tangled structures in the embedding geometry, improving cluster separability. Moreover, since flows model an explicit likelihood, they also provide probabilistic diagnostics for detecting low-density or anomalous regions in the embedding space.(And probably with GMM).
2.	Move back toward cosine-based clustering: If environment constraints are resolved, implementing spherical K-Means or another cosine-aligned method or probability-based algorithms (kernal PCA; Spectral Clustering/Hierarchical Clustering) could improve performance, since direction matters more than magnitude in our dataset.
3.	Feature fusion: Combining DINO embeddings with our earlier handcrafted features—or using fuzzy logic to estimate each defect's probability—could help resolve ambiguous cases with specific domain knowledge as proofs.
One small improvement before final testing
A simple and practical improvement would be: Adjusting UMAP hyperparameters (e.g., lowering n_neighbors) to preserve more local structure and potentially increase cluster separability. This is easy to implement and may yield a noticeable gain.

4.6 Runnable Code and Test Example Output
How to run the code:
The complete pipeline is implemented in a single Python script. To run it, execute the following command in the CRC terminal with submit file:
python wafer_clustering_pipeline.py
The script will load the DINO embeddings, apply L2 normalization, perform PCA and UMAP dimensionality reduction, run both HDBSCAN and K-Means clustering, and generate visualizations and metrics. All results will be saved to the directory specified in the script.
Test example output:
 <img width="2400" height="1800" alt="umap_kmeans" src="https://github.com/user-attachments/assets/1c9c8eb3-a742-4b68-b9b4-357f8a1aca71" />

<img width="2400" height="1800" alt="umap_hdbscan" src="https://github.com/user-attachments/assets/8b96b2ae-e6f4-426b-b87d-7c157a431143" />

<img width="3600" height="5400" alt="kmeans_cluster_0" src="https://github.com/user-attachments/assets/4259f441-7eb1-4832-b170-c0058c6b4938" />
<img width="3600" height="5400" alt="kmeans_cluster_1" src="https://github.com/user-attachments/assets/868a862c-a6ac-46f6-9f77-50a4051a4e58" />
<img width="3600" height="5400" alt="kmeans_cluster_2" src="https://github.com/user-attachments/assets/7e741ed5-67b2-4729-881f-1f9b8c06ea52" />
<img width="3600" height="5400" alt="kmeans_cluster_3" src="https://github.com/user-attachments/assets/88fe0bbb-c33f-461d-a014-4fd25d14f7bc" />
<img width="3600" height="5400" alt="kmeans_cluster_4" src="https://github.com/user-attachments/assets/73ca156c-4fc5-4e11-a1da-424596aaf953" />
<img width="3600" height="5400" alt="kmeans_cluster_5" src="https://github.com/user-attachments/assets/a38690a6-f67b-416e-b7d3-16aeeaf8cbe6" />
<img width="3600" height="5400" alt="kmeans_cluster_6" src="https://github.com/user-attachments/assets/0af0febd-a749-457c-b528-dbc70d6bab8e" />
<img width="3600" height="5400" alt="kmeans_cluster_7" src="https://github.com/user-attachments/assets/fda2580c-2c04-4ef8-afef-a2c52df8898b" />
<img width="3600" height="5400" alt="kmeans_cluster_8" src="https://github.com/user-attachments/assets/d871ea1c-6f5e-4be4-89c1-a0f878f3b145" />
<img width="3600" height="5400" alt="kmeans_cluster_9" src="https://github.com/user-attachments/assets/9fa22a4e-bfca-4a7c-89b2-03f3b9fa8092" />


<img width="2400" height="1800" alt="kmeans_calinski_vs_k" src="https://github.com/user-attachments/assets/ef788818-5989-4336-b696-9293cb0357b8" />
<img width="2400" height="1800" alt="kmeans_silhouette_vs_k" src="https://github.com/user-attachments/assets/aaec5b88-6f2b-4981-9980-99a9f4af11b0" />

----------------------【12.21 Update】 Part 5: Final Evaluation & Comprehensive Benchmark----------------------
5.1. Test Database & Protocol (1 point)
1. Scale & Isolation: The final test set consists of 34,590 samples (matching the size of the validation set). The test set corresponds to a 20% hold-out split of the full dataset. This data was kept strictly sequestered during the feature engineering and parameter tuning phases to ensure the integrity of the "unseen data" evaluation.
   
2. Pipeline Consistency: After the dataset was split into training, validation, and test subsets, we deliberately avoided accessing the test set during model development in order to preserve a strict unseen-data evaluation protocol. As a result, embeddings were initially generated only for the training and validation sets. The test embeddings were generated later using the same DINOv2 (ViT-S/14) model, identical preprocessing steps, and the same embedding extraction code, with the only change being the input file paths. No model parameters were updated during this process. This confirms that the observed performance reflects stable and generalizable representation learning rather than procedural variation or overfitting.

5.2. Technical Evolution: From HDBSCAN to GMM (1 point)
1. The Failure of Density-Based Clustering: We initially explored HDBSCAN due to its theoretical superiority in density modeling. However, the WM-811K dataset presented a "smooth slope" feature distribution rather than the discrete "peaks and valleys" required by HDBSCAN. Even after extensive and repeated hyperparameter tuning on the validation set, HDBSCAN could not yield stable clusters. The dataset's continuous transition states (high-entropy samples) violated the model's strong assumptions about discrete density peaks and valleys.
   
2. The GMM Advantage: Weighing the high computational complexity of Spectral Clustering, and the challenges of Manual Feature Fusion (requiring additional multi-dimensional quantitative features with uncertain redundancy and duplication), we were fortunate to discover and successfully validate a Gaussian Mixture Model (GMM), which delivered surprisingly strong performance. GMM’s ability to perform soft clustering and probabilistic density estimation perfectly aligns with the overlapping nature of wafer defect distributions.These metrics evaluate ranking quality rather than strict class prediction.

5.3. Performance Comparison & Methodology (3 points)
1. Summary & Comparison with Kmeans in Part4 : Compared to the previously tested KMeans approach, which primarily captures defect density and relies on relatively arbitrary and rigid cluster boundaries, the GMM provides a softer partition of the data along with explicit confidence estimates. This probabilistic formulation allows samples near cluster boundaries or in transitional states to be represented more naturally, rather than being forced into hard assignments.
2.Results Analysis:
(1)【PCA】
Given our pipeline's streamlined nature, we conducted an in-depth analysis of the PCA outputs to investigate the spatial features captured by the model. To "open the black box" of the latent space, we further visualize and analyze the dimensionality reduction results:
PC1 — Global Defect Density Axis
PC1 High→ High overall defect density; many defects distributed across the wafer.
PC1 Low→ Low overall defect density; few or near-absent defects.
PC2 — Spatial Dispersion / Structure Axis
PC2 High→ Defects are sparsely and randomly scattered; weak spatial correlation.
PC2 Low→ Defects exhibit spatial clustering or localized structure; strong spatial correlation.
PC3 – Random Noise vs. Structural Signal
PC3 High→ Appears more random and noisy, with no obvious structure in the data points.
PC3 Low→ :Cleaner/more regular, with fewer defects or a more orderly arrangement of defects.
While PC1–PC3 exhibit clear and interpretable defect-related patterns, higher-order components (PC4–PC5) become increasingly mixed and less semantically coherent.
In particular, PC5 shows a noticeable mixture of wafers with central defect patterns and near-normal wafers, suggesting that the underlying structural semantics along this axis are no longer well aligned with a single physical defect mode.
These components are therefore more likely to capture residual variance and heterogeneous noise rather than distinct, meaningful defect structures.
Examples：
<img width="3000" height="4800" alt="pc1_high" src="https://github.com/user-attachments/assets/abaa6288-d0e0-4031-8f6b-4ad30472f55c" />
<img width="3000" height="4800" alt="pc1_low" src="https://github.com/user-attachments/assets/3c5ee4d1-cb74-4e8f-98f7-2973a84e4af9" />

3. GMM &entropy-based UMAP
The Transition Entropy map quantifies the uncertainty of our GMM predictions. Bright regions indicate high-entropy transition states where samples exhibit overlapping characteristics of multiple failure modes. This confirms our hypothesis of a 'continuous slope' in wafer defects, highlighting the model's ability to identify ambiguous samples that may require human secondary inspection.
<img width="3000" height="2400" alt="umap_entropy" src="https://github.com/user-attachments/assets/520fa24b-756e-4ec7-a032-f6d81103032e" />
<img width="3000" height="2400" alt="umap_clusters" src="https://github.com/user-attachments/assets/19bf8eed-63cd-458f-b925-0d8e559a0a18" />

4. Analysis of GMM Clustering Results
Since each of the 10 clusters is further bifurcated into Core and Boundary regions based on probability density, we can categorize the Cluster Cores into three primary "Production Baselines" for a more streamlined and efficient analysis:
（1）	The Golden Standard (Clusters 7, 8)
Morphology: Extremely pristine surfaces with minimal, highly uniform white noise.
Definition: Represents the theoretical maximum quality attainable under current process conditions.
<img width="3000" height="4800" alt="core" src="https://github.com/user-attachments/assets/fe1921c8-1c4e-4551-b264-f2293cdb2454" />
<img width="3000" height="4800" alt="core" src="https://github.com/user-attachments/assets/a489379a-9dbe-4bae-872d-a47733db539e" />

（2）	The Process Buffer (Clusters 0, 1, 2, 4)
Morphology: Medium-density white noise.
Definition: Represents routine process fluctuations. This background noise reflects minor environmental disturbances during production—while not perfect, it remains well within acceptable yield tolerances.
<img width="3000" height="4800" alt="core" src="https://github.com/user-attachments/assets/c4d11097-06a4-4379-99ae-3719c6c7e23d" />
<img width="3000" height="4800" alt="core" src="https://github.com/user-attachments/assets/0f8fa8ef-f5cb-4cb4-b89f-6d98a9712bca" />
<img width="3000" height="4800" alt="core" src="https://github.com/user-attachments/assets/868c53ee-a0f5-4055-9ea5-b311054d0cd8" />
<img width="3000" height="4800" alt="core" src="https://github.com/user-attachments/assets/c592afe7-94cc-4607-9a1b-ee5351ff2136" />
(Cluster Core-4 already has certain minor Loc defect.Depending on the real-world criteria,it could be either minor Loc defect or acceptable noises.)

（3）	The Machine Artifacts (Cluster 5)
Morphology: Distinctive black noise, uniformly distributed across the entire wafer.
Definition: Identified as systematic errors from the sampling hardware. The model successfully isolated this "systemic background noise" (a non-physical defect), demonstrating the power of self-supervised feature extraction in identifying sensor-specific biases.
For boundry types, since they all have various defect type combination, we take 0 for an example, A granular inspection of Boundary 0 reveals a concentrated 'taxonomy of failure,' where standard industrial defects such as Loc (1), Center (2), Scratch (3), and Donut/Edge-Ring (4) are clearly identifiable; this confirms that our GMM effectively pushes all morphologically diverse anomalies to the distribution boundaries, a pattern that consistently repeats across all other clusters. In other boundary types, we also encounter enormous kinds of catastrophic random defects, so we won't elaborate on them here.
<img width="3000" height="4800" alt="core" src="https://github.com/user-attachments/assets/83618254-f364-486b-adfd-e23196ae09a0" />

6. Benchmarking
We unified the evaluation "worldview" by treating anomaly scores (Negative Log-Likelihood for GMM, Reconstruction Error for AE) as a continuous ranking metric to calculate ROC-AUC. This allows heterogeneous methods with different modeling assumptions to be compared under a unified and fair evaluation framework. Specifically, we use Negative Log-Likelihood as the anomaly score for GMM, and reconstruction error for AutoEncoder-based methods.
All scores are treated as continuous rankings rather than hard labels.
Methodology	                      Feature Representation	ROC-AUC             	PR-AUC
Manual + AutoEncoder	          Hand-crafted (14D)	      0.783	               0.953
Manual + Isolation Forest	      Hand-crafted (14D)	      0.802	               0.95
DINOv2 + GMM (Ours)	              Semantic Latent (384D)	  0.933	               0.810
<img width="908" height="698" alt="image" src="https://github.com/user-attachments/assets/1f4ccdab-990c-4af1-b9b8-7b8884210993" />

Analysis: The leap from 0.80 to 0.93 represents a dimensional shift. While the NaN result on the training set (caused by extreme class imbalance where 'none' samples dominate) was a numerical anomaly, the 0.933 ROC-AUC on the independent test set stands as a robust validation of our latent-space modeling.
Interestingly, traditional models show a higher PR-AUC than our ViT-based model. This is actually due to "Feature Blindness": simple models only catch the most obvious defects, so they rarely make mistakes on easy samples, which creates a "fake" high precision.
Our ViT model, however, is "too sensitive" because it sees much more detail. The drop in our PR-AUC shows that the model is finding the "Grey Zone"—those wafers that look "normal" to humans but are already starting to drift toward failure. We aren't failing at precision; we are just catching subtle anomalies that human labels often miss.
Moreover, It is important to note that labels were not used as training data at any stage of the pipeline; therefore, the task remains strictly unsupervised in nature. The use of labels in post-hoc score computation and evaluation serves primarily to quantify the degree of alignment between self-supervised representations (machine knowledge) and human expert annotations (human knowledge). This allows for a more objective comparison across different models and helps verify that the observed performance gains indeed reflect a genuine dimensional or representational improvement rather than incidental effects.

This evaluation strategy is well aligned with industrial practice. By analyzing score distributions against human labels, we can determine an optimal operating threshold and directly address a practical question faced in manufacturing environments: at what anomaly score should the system trigger intervention or halt production?

5.4.Future Outlook: Industrial Preventive Maintenance
If provided with more time or data (e.g., from Nano Lab), our framework could be extended in several high-impact directions:
1.	Confidence Histograms: Visualizing the "uncertainty" of GMM predictions to filter out ambiguous samples.
Beyond assigning each wafer to a cluster, the probabilistic nature of GMM allows us to quantify prediction confidence. By visualizing confidence histograms (e.g., posterior probabilities or negative log-likelihood distributions), ambiguous or high-uncertainty samples can be explicitly identified and filtered, which are particularly valuable for human inspection or downstream decision-making. Such uncertainty-aware analysis would improve the robustness of industrial deployment by preventing overconfident decisions on low-quality inputs.
2.	Cosine-Similarity Optimized PCA(or even more dimentional reduction methods & clustering methods based directly on cosine-similarity): Although L2 normalization partially aligns Euclidean distance with cosine similarity, the two metrics are not strictly equivalent. Since Transformer-based models such as DINO primarily encode information in angular relationships, a more principled extension would involve dimensionality reduction and clustering methods that directly operate on cosine similarity. This could include cosine-aware PCA variants, kernel methods, or clustering algorithms explicitly designed for angular distance. Aligning the distance metric across representation learning, dimensionality reduction, and clustering would improve mathematical consistency and may further stabilize the latent structure.
3.	Defect Drift Tracking (Predictive Intervention): This could be a critical industrial application. Our model’s sensitivity to high-entropy transition states allows for the early detection of "defect drift." Instead of reacting to failed wafers in a post-mortem manner, gradual shifts in the latent distribution can be monitored over time, enabling earlier intervention before contamination effects accumulate into catastrophic economic loss
4.	Extension to flow-based GMM for more flexible density modeling. Standard GMM assumes Gaussian components in latent space, which may still be restrictive for complex industrial data. A natural model extension would be to integrate flow-based models to construct more expressive models, especially considering the generative models we studied in the course, since the dependence of k on prior specification directly leads to the intermixing of several defect categories within various boundaries, making it difficult to further refine specific defect types or potentially increasing the workload for reverse-tracing process defects. invertible density transformations before or within the mixture framework. The Flow-based GMM learns a continuous probability density field. Instead of rigidly assuming there must be ten pre-specified categories and fitting them with a fixed number of Gaussian distributions, it learns the 'flow direction' across the entire feature space to capture more dynamic and intrinsic patterns. In terms of application, this approach may be better equipped to capture more subtle wafer textures. 
5. ViT fine-tuning: Further visualize the ViT's attention to identify key regions, followed by targeted fine-tuning based on these findings to acquire more core, context-specific features."





This project will be completed individually.




