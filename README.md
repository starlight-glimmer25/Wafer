# Semester Project: Wafer Defect Anomaly Detection

Part 1: Problem Definition & Objective

Semiconductor manufacturing is highly sensitive to microscopic process deviations. Even a small defect pattern on a wafer can drastically reduce yield, resulting in significant financial losses. Traditional supervised approaches to wafer defect inspection—often based on convolutional neural networks (CNNs)—are effective at classifying known defect types but suffer from a critical limitation: they cannot detect novel, unseen defect types. These novel anomalies, which may arise from equipment drift, material contamination, or unexpected process conditions, are often the most costly because they are not captured by existing supervised models.

Project Objective
The goal of this project is to design a computer vision–based anomaly detection system that can identify and localize wafer defects without requiring pre-labeled examples of every defect type. Instead of predefining all categories, the system should learn what “normal” wafer maps look like, and then raise alerts whenever deviations are detected. This creates a “sentry” mechanism to support engineers in yield management.

Part 2: Proposed High-Level Solution

The proposed approach will shift the problem from supervised classification (predicting a label from a fixed set) to unsupervised anomaly detection (learning “normality” and detecting deviations).

Core Idea

Train the model only on “normal” (defect-free) wafer maps.

Build a compact representation of what a healthy wafer should look like.

During testing, flag wafer regions that differ significantly from this representation as anomalies.

Expected Benefits

Data Efficiency: Requires only defect-free wafers for training, which are abundant.

Generalization: Can detect both known and unknown defects.

Interpretability: Reconstruction errors or anomaly scores can be visualized as heatmaps, pointing engineers to specific regions of concern.

Key Requirements
The model must be agnostic to global variations such as brightness or contrast changes between scans.
It must be robust to minor process variations that do not constitute true defects (otherwise false positives will overwhelm the system).
It should be sensitive to spatially structured anomalies, which may take forms such as:

Linear scratches

Circular rings or donuts

Localized blobs or clusters

Random point-like defects

Part 3: Required Datasets & Strategy

Primary Dataset
We plan to use the WM-811K Wafer Map Dataset, a large-scale, publicly available dataset from real fabrication lines.

Contains over 800,000 wafer maps.

Includes wafers labeled with 9 known defect patterns (e.g., Center, Donut, Edge-Loc, Scratch) plus a “None” (defect-free) class.

For anomaly detection, the “None” class will be the primary source of training data.

Data Splitting

Training Set: Exclusively “None” (good wafers) for the anomaly detection model.

Validation Set: Mixture of good wafers + a subset of known defect categories, used to tune thresholds and prevent overfitting.

Test Set (Final Evaluation):

Good wafers

Defects from categories seen during validation

Defects from categories withheld entirely during training/validation → these represent novel defects, the true “final exam” for the system.

Possible Extensions

Use synthetic anomaly generation (rotations, noise injection, shape overlays) to stress-test robustness.

Explore transfer learning from models pretrained on industrial anomaly detection datasets (e.g., MVTec-AD).

Part 4: Learning Goals

To implement this project, I will need to acquire and practice knowledge in several areas:

Unsupervised Deep Learning

Autoencoders and Variational Autoencoders (VAEs) for reconstruction-based anomaly detection.

Student–Teacher networks and patch-based modeling (e.g., PaDiM, PatchCore).

Evaluation Metrics for Imbalanced Anomaly Detection

AUROC (Area Under ROC Curve)

AUPRC (Area Under Precision–Recall Curve)

False Positive Rate (FPR) control for deployment relevance.

Interpretability Techniques

Visualization of reconstruction errors as heatmaps.

Gradient-based saliency or attention maps to localize anomalies.

Industrial Constraints

Computational efficiency (wafer inspection must keep up with high-throughput production lines).

Controlling false alarms, since excessive alerts would undermine trust in the system.

Part 5: Expected Contribution

By the end of the semester, the project should deliver:

A baseline unsupervised model capable of detecting anomalies in wafer maps.

Visualization of anomaly regions (heatmaps).

A comparative analysis with a supervised baseline (e.g., CNN classifier) to show the benefit of unsupervised detection on novel defects.

Documentation of challenges, limitations, and potential improvements for industrial deployment.

Even if the system is not perfect, achieving a working prototype will demonstrate the feasibility of anomaly detection in semiconductor yield management, and serve as a valuable starting point for future optimization.

Remarks

In an ideal scenario, we would obtain access to real production wafer data and evaluate our model’s performance under realistic fab conditions, potentially benchmarking it against supervised classifiers to provide a direct comparison. However, if such data is not accessible, the project will rely primarily on the WM-811K dataset as a proof-of-concept.

Furthermore, this project plan should be treated as a living document: depending on practical challenges, unforeseen obstacles, and experimental outcomes during the semester, the scope and implementation details may evolve. Any such adjustments will be communicated promptly to ensure alignment with course expectations.
