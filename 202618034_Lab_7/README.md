# DS605 Lab 7 – Image Representation, PCA, Eigenfaces and Face Recognition

Dataset: **AT&T (ORL) Database of Faces** – 40 subjects × 10 grayscale images, 112 × 92 px.

## How to run

1. Download the AT&T/ORL face images and unzip them (e.g. `att_faces/s1/1.pgm … s40/10.pgm`).
2. Open `DS605_Lab7_PCA_Eigenfaces.ipynb` and set the `path` variable in the first code cell to that folder.
3. Run all cells. Figures are written to `figures/`, tables to `results/`.

`pip install -r requirements.txt`

## Approach

| Part                       | What is done                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A** Image representation | Images read as 8-bit grayscale matrices, intensities inspected, flattened to 10 304-dim vectors, image matrix `X (400×10304)` + labels. Seeded, per-subject stratified 80/20 split (8 train / 2 test per subject).                                                                                                                                                                                                                                                                                                                         |
| **B** PCA / eigenfaces     | Mean face from the training set, mean-centering. Principal directions by (1) eigendecomposition of the covariance via the Turk–Pentland Gram-matrix trick (`np.linalg.eigh`) and (2) SVD (`np.linalg.svd`); the two are compared (eigenvalues, directions, orthonormality, `‖Cu−λu‖`, timing). Cumulative explained variance → smallest _k_ with ≥ 90 %. Eigenfaces visualised, train/test projected, reduced representation saved (`results/pca_representation.npz`), faces reconstructed, MSE/PSNR and storage compared with raw pixels. |
| **C** Recognition          | Nearest stored training face (Euclidean; cosine as extra) in PCA space vs raw pixels: accuracy, timing, correct/incorrect match figures, accuracy vs _k_, and a 10-split robustness check.                                                                                                                                                                                                                                                                                                                                                 |
| **D** Traditional ML       | k-NN, Logistic Regression, linear SVM and RBF SVM on PCA scores. Hyper-parameters (and whether to standardise) tuned by stratified 4-fold CV on the training set for several values of _k_; configuration chosen by CV score (ties → fewer components); test set only used for final reporting.                                                                                                                                                                                                                                            |

**Constraint:** all PCA/eigenface computation uses NumPy only; scikit-learn is used only in Part D.

## Deliverables in this repo

- `DS605_Lab7_PCA_Eigenfaces.ipynb` – runnable notebook
- `figures/` – explained variance, eigenfaces, reconstructions, matches, comparison plots
- `results/` – `ml_comparison_table.csv`, `nn_accuracy_vs_k.csv`, `reconstruction_error.csv`, `storage_comparison.csv`, `summary.md`, …

## Results and observations

After running the notebook, paste the contents of `results/summary.md` here (it contains the key numbers and the comparison table), then add your own observations. The notebook's final "Observations and discussion" cell lists the points to check against your numbers.
