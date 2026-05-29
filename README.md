# Machine Learning: Clustering, Neural Networks & Dimensionality Reduction

This project explores three fundamental machine learning paradigms (clustering, regression, and classification) and investigates how dimensionality reduction techniques (PCA vs. Autoencoders) impact model performance and computational efficiency.

## Datasets & Tasks
The project evaluates models across three distinct datasets:
* **Human Activity Recognition (HAR) - Clustering:** Sensor data (561 features) clustered using K-Means to identify 6 physical activities.
* **California Housing - Regression:** Census data used to predict median house values using a Feedforward Neural Network (FNN).
* **Fashion-MNIST - Classification:** Grayscale image classification (10 categories) using a Convolutional Neural Network (CNN).

## Methodology & Dimensionality Reduction
For each task, the original data was compared against reduced representations targeted to maintain ~95% of the variance:
* **PCA (Principal Component Analysis):** Used as a linear dimensionality reduction baseline.
* **Autoencoders:** Neural networks trained to compress and reconstruct data, capturing non-linear relationships.

### Models Used:
* **K-Means** (k=6) optimized via silhouette score.
* **FNN** (128-64-32-1) with ReLU, BatchNorm, and Dropout.
* **CNN** (Conv2D, MaxPool, Dense) for image feature extraction.

## Key Findings
* **Clustering (HAR):** PCA (50D) achieved the best clustering quality, outperforming both the original (561D) and Autoencoder representations. Linear reduction proved highly effective for sensor data with strong linear correlations.
* **Regression (Housing):** The original features (13D) performed best ($R^2=0.814$). However, the Autoencoder (8D) achieved a very close performance ($R^2=0.805$), capturing non-linear price relationships better than PCA while reducing dimensions by 38%.
* **Classification (Fashion-MNIST):** The Original model (28x28) achieved 89.5% accuracy. The Autoencoder representation (14x14) significantly outperformed PCA, preserving visual structures and achieving 86.9% accuracy despite a 75% pixel reduction.

## Authors
Project developed by:
* Fabiano Cacioli 
* Luca Buonomini
