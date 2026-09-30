# DS605 – Lab 6: Image and Text Feature Extraction and Classification

**Course:** DS605 – Fundamentals of Machine Learning
**Topic:** Converting raw image and text data into numerical features and classifying them with traditional ML models (no CNNs, no deep learning, no pretrained embeddings).

| Task | Dataset | Size |
|------|---------|------|
| Image classification (Crack / NonCrack) | [Asphalt Crack Dataset – Mendeley Data](https://data.mendeley.com/) (448×448 version) | 400 images (200 crack, 200 non-crack) |
| Email spam classification | [Email Spam Classification Dataset – Kaggle](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv) | 5,172 emails (3,672 non-spam, 1,500 spam) |

---

## Repository Structure

```text
.
├── Lab_6_Part_A.ipynb                 # Image feature extraction + classification
├── Lab_6_Part_B.ipynb                 # Email spam classification
├── Lab_6_Part_C.ipynb                 # Representation improvement (spam feature selection + Decision Tree on images)
├── cracks_data.csv                    # Extracted image-feature table (400 rows)
├── cracks_results_comparison.csv      # Image model evaluation results
├── DS605_Lab6_...Asphalt_Final.pdf    # Assignment statement
└── README.md
```

> Dataset folders (`448/Cracks`, `448/NonCracks`, `emails.csv`) are not included in the repo. See *How to Run* below.

---

## Workflow

```text
Raw Image/Text -> Preprocessing -> Feature Extraction / Vectorization
               -> Train-Test Split -> ML Model -> Evaluation -> Representation Improvement
```

---

## Part A – Image Feature Extraction and Classification

**Notebook:** `Lab_6_Part_A.ipynb`

### Preprocessing
- Images are read with **OpenCV** (`cv2.imread`) and converted to grayscale (`cv2.COLOR_BGR2GRAY`).
- Inspection of a sample image: colour shape `(448, 448, 3)`, grayscale shape `(448, 448)`, dtype `uint8`, intensity range 0–255.
- The 448 version of the dataset is already at a common size, so no additional resizing was applied.

### Features extracted (one row per image)
| Feature | Description |
|---------|-------------|
| `mean_brightness` | Mean grayscale intensity |
| `contrast` | Standard deviation of grayscale intensities |
| `dark_pixel_ratio` | Fraction of pixels with intensity < 50 |
| `bright_pixel_ratio` | Fraction of pixels with intensity > 100 |
| `edge_density` | Fraction of pixels marked as edges by **Canny** (thresholds 50 and 100) |

All features are computed with NumPy/OpenCV only. The final table (`cracks_data.csv`) has shape **400 × 7** (5 features + `label` + `image_name`). Label `1` = Crack, `0` = NonCrack.

### Visualizations
The notebook shows a sample crack image, its grayscale version, and its Canny edge map (sample edge density ≈ 0.377).

### Train-test split
80% train / 20% test (320 / 80 images), `stratify=y`, `random_state=42`.

### Results (test set, 80 images)

| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Prediction Time (s) |
|-------|---------:|----------:|-------:|---------:|------------------:|--------------------:|
| Logistic Regression | 0.6625 | 0.6757 | 0.6250 | 0.6494 | 0.0584 | 0.0022 |
| Decision Tree *(added in Part C)* | 0.8500 | 0.8889 | 0.8000 | 0.8421 | 0.0062 | 0.0018 |
| **Random Forest** (100 trees) | **0.9125** | **0.9231** | **0.9000** | **0.9114** | 0.2831 | 0.0190 |

**Confusion matrices** (rows = actual, columns = predicted, order: NonCrack, Crack):

| Model | Matrix |
|-------|--------|
| Logistic Regression | `[[28, 12], [15, 25]]` |
| Decision Tree | `[[36, 4], [8, 32]]` |
| Random Forest | `[[37, 3], [4, 36]]` |

### Observations
- Logistic Regression performs poorly (66%), suggesting the relationship between the five global features and the label is **non-linear**, which a linear decision boundary cannot capture well.
- The tree-based models do much better. Random Forest is the most accurate (91%) and has the most balanced precision and recall, at the cost of roughly 5× the training time of Logistic Regression and 3× the training time of the Decision Tree.
- The single Decision Tree is a good speed/accuracy compromise: it is the fastest to train (~6 ms) and reaches 85% accuracy.
- Each image is reduced to only **5 numbers**, so the representation is extremely compact and fast. However, these are global statistics, so they do not capture *where* or *what shape* a crack is.
- Edge density varies only in a narrow range (≈ 0.316–0.394) across images, which suggests the chosen Canny thresholds (50, 100) also pick up a lot of asphalt surface texture. Tuning the thresholds is a natural next step.

---

## Part B – Email Spam Classification

**Notebook:** `Lab_6_Part_B.ipynb`

### Dataset inspection
- Shape: 5,172 rows × 3,002 columns (`Email No.`, 3,000 word-count columns, `Prediction`).
- No missing values and no text (object) columns other than the ID.
- Class distribution: **71.0% non-spam (3,672)** vs **29.0% spam (1,500)**. Classes are moderately imbalanced, so a stratified split is used.
- `Email No.` is dropped (identifier only); `Prediction` is the target (`0` = non-spam, `1` = spam).

### Important dataset limitation
The Kaggle file `emails.csv` does **not contain the raw email text**. It is already a pre-computed bag-of-words matrix with 3,000 word-count features per email. Because `CountVectorizer` and `TfidfVectorizer` both require raw text, they **could not be applied to this file**, and no artificial text was fabricated. The pre-computed 3,000-feature word-count representation was used directly instead. If a raw-text version of the dataset is used, the same classification and evaluation code can be reused with the two vectorizers.

### Model and results
Logistic Regression (`max_iter=1000`), 80/20 stratified split (4,137 train / 1,035 test), `random_state=42`.

| Representation | # Features | Accuracy | Precision | Recall | F1 | Training Time (s) | Prediction Time (s) |
|----------------|-----------:|---------:|----------:|-------:|---:|------------------:|--------------------:|
| Provided word-count matrix | 3,000 | 0.9826 | 0.9578 | 0.9833 | 0.9704 | ~6.4 | ~0.13 |

Confusion matrix: `[[722, 13], [5, 295]]` (TN = 722, FP = 13, FN = 5, TP = 295).

### Observations
- Logistic Regression is very strong on this representation: ~98% accuracy and ~98% spam recall, with only 5 spam emails missed.
- The main cost is dimensionality: 3,000 features make training noticeably slower than the image task. This motivates the feature reduction tested in Part C.

---

## Part C – Improving the Representation

**Notebook:** `Lab_6_Part_C.ipynb`

### Change 1 – Feature selection for spam (dimensionality reduction)
**Justification:** 3,000 word features make training slow, and many words are uninformative for spam detection. `SelectKBest(chi2, k=500)` keeps the 500 features most associated with the target. The split is performed **before** selection and the selector is fitted on training data only, to avoid test-set leakage.

| Representation | # Features | Accuracy | Precision | Recall | F1 | Training Time (s) | Prediction Time (s) |
|----------------|-----------:|---------:|----------:|-------:|---:|------------------:|--------------------:|
| Original | 3,000 | 0.9826 | 0.9578 | 0.9833 | 0.9704 | 6.599 | 0.0785 |
| Improved (chi² top-500) | 500 | 0.9652 | 0.9342 | 0.9467 | 0.9404 | 3.070 | 0.0032 |

- **Dimensionality reduced by 83.33%** (3,000 → 500).
- Training time dropped by ~53% and prediction time by ~96%.
- Accuracy fell by ~1.7 percentage points and F1 by ~3 points. False negatives rose from 5 to 16 and false positives from 13 to 20 (`[[715, 20], [16, 284]]`).

### Change 2 – Model choice for images
**Justification:** Logistic Regression was weak on the image features, so a **Decision Tree** was added as a non-linear model. It improved accuracy from 66.25% to 85.0% with almost no extra cost, and trained faster than Random Forest (see the Part A results table).

### Trade-off discussion
| Aspect | Observation |
|--------|-------------|
| **Dimensionality vs. performance** | Cutting spam features by 83% costs only a small drop in accuracy (98.3% → 96.5%), so most predictive information lives in a small subset of words. Some useful signal is still lost, particularly in spam recall (98.3% → 94.7%). |
| **Dimensionality vs. computation** | Fewer features give much faster training and near-instant prediction. This matters for large datasets or real-time filtering. |
| **Model complexity vs. performance (images)** | Non-linear models (Decision Tree, Random Forest) clearly outperform the linear baseline on the 5-feature image representation. Random Forest gives the best accuracy but is the slowest of the three. |
| **Which to choose?** | If missing spam is costly, keep the full 3,000-feature model. If speed or memory matters more, the 500-feature model is a reasonable trade-off. |

---

## Constraints Followed
- No CNNs, deep-learning models, or pretrained image embeddings.
- Image features are derived directly from raw pixels using NumPy and OpenCV.
- Scikit-learn is used only for traditional ML models, splitting, feature selection, and metrics.

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib opencv-python scikit-learn kagglehub jupyter
   ```
2. **Image data (Part A):** download the Asphalt Crack dataset (448 version) and arrange it as:
   ```text
   448/
   ├── Cracks/
   └── NonCracks/
   ```
   placed next to the notebook.
3. **Email data (Parts B & C):** `Lab_6_Part_B.ipynb` downloads the dataset via `kagglehub`. Alternatively, download `emails.csv` from Kaggle manually. Note that Part C currently loads the file from a hard-coded local path, so update the `path` variable to match your machine.
4. **Run the notebooks in order:** Part A → Part B → Part C. Part A creates `cracks_data.csv` and `cracks_results_comparison.csv`, which Part C reads to add the Decision Tree result.

All random operations use `random_state=42`, so results are reproducible (timings will vary slightly between machines).

---

## Limitations
- The image dataset is small (400 images) and the test set has only 80 images, so metrics can vary noticeably with a different split.
- Only five global image features are used; they ignore crack shape and location.
- Canny thresholds and the dark/bright intensity thresholds were fixed, not tuned.
- CountVectorizer and TF-IDF comparison was not possible because the provided email file has no raw text (see Part B).
- Reported timings come from single runs and are only indicative.
