## Part 1: Problem Overview

This project focuses on developing an unsupervised anomaly detection system for semiconductor wafer maps. In the semiconductor industry, even minor undetected defects on wafers can result in severe yield loss, especially when the issue is only discovered after dozens or even hundreds of wafers have already been manufactured and discarded. 

While existing research, such as supervised CNN-based classification models (e.g., by Stanford teams), have shown strong performance in controlled environments, they often fail in real-world industrial use cases when confronted with novel, 'zero-shot' defect types not present in the training data.. This is primarily because real-world manufacturing defects are frequently novel or previously unseen, making supervised methods less effective due to the lack of labeled examples for rare anomalies.

Our goal is to design a system that can leverage patterns from known “good” wafers to identify previously unseen or subtle anomalies early in the process. Ideally, such a system could help prevent large-scale quality loss by detecting potential yield-impacting anomalies as soon as they emerge.

This project is directly inspired by my firsthand experience with yield loss challenges during my tenure as a Manufacturing Quality Engineer at TSMC. It aims to bridge the gap between academic research and industrial application by developing a practical solution for early detection of novel defects.It not only fulfills the course requirement but also provides a valuable opportunity to build a research-grade project that may enhance my resume and career prospects. If the work yields meaningful or novel results, I would also be interested in exploring publication opportunities in relevant industrial or computer vision venues.

Part 2: Proposed High-Level Solution 

This project aims to build an unsupervised anomaly detection system for semiconductor wafer images. Unlike supervised learning, which focuses on classifying known defect types, our goal is to rely solely on normal samples, train the model to learn the structural patterns of normal wafers, and then identify images that deviate from these patterns as potential anomalies. This approach is tailored to the real-world industrial environment where unknown defects are frequent but labels are scarce.

While many different types of defects exist in the actual wafer production process, they often share certain common characteristics: disruptions to the symmetry, repetitiveness, or spatial structure of the wafer image; localized brightness abrupt changes, texture breaks, and discontinuous patterns. While these characteristics are visually perceptible, they are often difficult to formalize using handcrafted rules. This is where machine learning, particularly representation learning methods, lies in their value.

We have not yet finalized the specific algorithms we will use, but our initial focus is on unsupervised/semi-supervised learning to ensure that they are as close to real-world industrial applications as possible. In terms of method selection, we may employ a denoising autoencoder, a structure more sensitive to anomalies. Through an "input-reconstruction" mechanism, it can improve the model's ability to detect subtle structural deviations while learning from normal wafers. We may also explore strategies such as principal component analysis (PCA) or clustering to construct a representation space for normal samples and identify anomalies based on distance or density.

Notably, we require a model mechanism that can "see through phenomena to the essence" to address the more severe risk distribution characteristics of industrial production: the cost of false negatives (missed detections) far outweighs the cost of false positives (false alarms). Therefore, our primary objective is to maximize recall (minimize false negatives), even at the cost of a higher false positive rate. The operational threshold will be tuned via ROC curve analysis to find an acceptable balance for a hypothetical production line. During the model design and training phases, we prioritize maximizing recall to ensure that no potential defects are missed. Furthermore, to demonstrate the balance of the model's overall performance, we will also report precision and false positive rates in our experiments, and plot receiver operating characteristic (ROC) curves to provide a more comprehensive evaluation metric reference.

The output of this system may include: (1) an anomaly score for each image; (2) a residual map highlighting abnormal areas; or (3) a binary classification judgment (normal vs. abnormal) based on a score threshold. We will further test different output methods and threshold selection strategies in subsequent experiments.

In addition, we hope that the model has a certain degree of invariance: for example, it is insensitive to slight brightness changes, geometric rotations, or small translations of the image. This helps to reduce false positives and allows the model to focus on anomalies that are truly structurally destructive rather than triggering alarms due to irrelevant disturbances.

[Current statement] : Since this project is still in the exploratory stage, the specific model structure, feature extraction method, anomaly score calculation mechanism, and final output form are still under investigation and may be dynamically adjusted based on actual results. We will also continue to evaluate the degree of match between existing public datasets (such as WM-811K) and real industrial scenarios, and consider supplementing other data or fine-tuning the objectives when necessary.
Part 3 Datasets
The primary dataset planned for this project is the WM-811K Wafer Map dataset, a publicly available benchmark dataset commonly used in semiconductor defect detection research. It consists of over 800,000 wafer map images, most of which are labeled as “normal,” along with a limited number of samples across several defect categories such as edge-loc, center, donut, scratch, and others. The images are grayscale with relatively low resolution (typically 28×28 pixels), which mimics real-world data constraints in terms of compactness and low storage cost.

Given the industrial motivation of this project, the training set will primarily consist of normal wafers only, enabling the model to learn the structure and distribution of healthy patterns. The validation set may include both normal samples and a small subset of defect cases to support threshold tuning and parameter selection. The test set will include a broader mix of defect types—including some classes not seen during training—to evaluate the model’s generalization capability in detecting novel anomalies.

One notable challenge of this dataset is the high imbalance between normal and defective samples, which reflects real-world distributions but also increases the risk of bias toward over-generalizing “normal” patterns. Furthermore, some defect categories exhibit very subtle visual cues, requiring the model to learn highly sensitive structural representations to distinguish them. Preprocessing steps such as normalization, potential resizing, and possibly grayscale binarization will be considered to reduce irrelevant noise and enhance structural contrast.

In addition to using WM-811K, I plan to reach out to real-world industry partners or internal university research labs to explore access to authentic wafer data. If granted, such datasets would serve not only to validate the proposed model in a realistic deployment scenario, but also potentially allow for experimental deployment or feedback from domain experts. The overarching goal is to develop a solution that is not only academically valid, but also practically viable in high-precision industrial environments.
-----------------------------------------0929 Update---------------------------------------------------
Part 4: Feature & Invariance Considerations
Our model aims to detect structural anomalies in wafer maps, such as symmetry breaks, irregular patterns, or local pixel disruptions. These features are more meaningful than raw pixel values and better reflect actual production defects.
At the same time, the model should be robust to irrelevant variations like minor changes in brightness, slight rotations, or small shifts. These factors are common in real-world data and should not trigger false alarms.
We prioritize high sensitivity to subtle defects while maintaining invariance to uninformative noise, ensuring the model can detect real problems without overreacting to harmless variation.


This project will be completed individually.




