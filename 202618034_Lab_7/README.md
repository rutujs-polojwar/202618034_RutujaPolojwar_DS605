# DS605 Lab 7 – Image Representation, PCA, Eigenfaces and Face Recognition

**Dataset:** AT&T (ORL) Database of Faces – 40 subjects × 10 grayscale images, 112 × 92 px (Kaggle `kasikrit/att-database-of-faces`, loaded with `kagglehub`).
**Constraint:** PCA/eigenfaces use NumPy only (`np.linalg.eigh`, `np.linalg.svd`); scikit-learn is used only in Part D.

## Run

```bash
pip install numpy pandas matplotlib pillow scikit-learn kagglehub jupyter
jupyter notebook Lab_7.ipynb   # run all cells
```

## Approach and results

**A – Representation.** Images read as 8-bit grayscale, flattened to 10,304-dim vectors (scaled to [0,1]) → `X` (400 × 10304). Seeded stratified split: 8 train / 2 test per subject (320 / 80).

**B – PCA.** Mean face and centering from the training set only. Covariance eigendecomposition (Gram-matrix trick) and SVD agree: max eigenvalue difference 1.3e-13, |cos| of top-50 directions = 1.0.

- **k = 100** components keep ≥ 90 % variance (95 %: 164, 99 %: 267); PC1 = 17.7 %.
- Reconstruction of test faces at k = 100: PSNR 23.4 dB (k = 20: 21.1 dB, k = 319: 24.6 dB).
- Storage (320 faces, float64): pixels 26.4 MB vs PCA 8.6 MB incl. basis (3.1×); per image 10,304 → 100 numbers (103×). The saving grows with dataset size (10,000 faces: ≈ 50×).

**C – Nearest-neighbour recognition.**

| Representation        | Dims   | Accuracy        | Match time |
| --------------------- | ------ | --------------- | ---------- |
| Raw pixels, Euclidean | 10,304 | 96.25 % (77/80) | 260 ms     |
| PCA k=100, Euclidean  | 100    | 96.25 % (77/80) | 5.4 ms     |
| PCA k=100, cosine     | 100    | 96.25 %         | –          |

Raw and PCA matching give identical predictions; PCA is ~48× faster. Accuracy peaks at 97.5 % with only 10–15 components. Over 10 random splits: raw 97.9 ± 1.2 %, PCA 98.1 ± 1.4 %.

**D – Traditional ML on PCA features.** Hyper-parameters and k tuned by 4-fold CV on the training set; final choice by CV score (ties → fewer components).

| Method                                   | Features | CV acc      | Test acc    |
| ---------------------------------------- | -------- | ----------- | ----------- |
| Raw pixels + 1-NN                        | 10,304   | –           | 96.25 %     |
| PCA(k=100) + 1-NN                        | 100      | –           | 96.25 %     |
| PCA(k=100) + KNN                         | 100      | 97.19 %     | 96.25 %     |
| PCA(k=100) + Logistic Regression (C=100) | 100      | 98.75 %     | 97.50 %     |
| PCA(k=120) + SVM RBF (C=10)              | 120      | 98.44 %     | 97.50 %     |
| **PCA(k=60) + SVM linear (C=0.1)**       | **60**   | **98.75 %** | **97.50 %** |

## Observations

1. Faces are highly compressible: 100 of 319 components hold 90 % variance, and ~10 are enough to reach ≈ 96–97 % accuracy.
2. Covariance and SVD give the same eigenvalues and directions (up to sign).
3. PCA matches raw-pixel accuracy at 1/100 of the dimensionality and ~48× faster.
4. Logistic Regression and SVMs score 97.5 % vs 96.25 % for KNN/nearest-neighbour, with higher CV accuracy. This is one test image out of 80, within the split-to-split noise (~1.2–1.4 %), so all methods perform similarly on ORL.
5. The selected model (linear SVM, 60 components, 2 errors out of 80) is the most compact, a 172× reduction from raw pixels. Standardising the PCA scores was never chosen, since it inflates noisy low-variance components.
6. Limitations: ORL is small and controlled, and PCA is linear and sensitive to pose and lighting.
