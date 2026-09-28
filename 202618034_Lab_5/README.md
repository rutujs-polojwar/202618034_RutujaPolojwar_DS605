# Machine Learning with Scikit-learn and From Scratch

## DS605 — Fundamentals of Machine Learning

This project implements regression and classification models using both **Scikit-learn** and **NumPy/Pandas from scratch** on the **UCI Productivity Prediction of Garment Employees** dataset.

The purpose of the project is to understand the complete machine-learning workflow and compare a library-based implementation with a manual implementation.

---

## 📌 Project Objectives

The project focuses on:

- Data preprocessing
- Missing-value handling
- Categorical feature encoding
- Feature scaling
- Train-test splitting
- Linear Regression
- Logistic Regression
- Model evaluation
- Training and prediction time comparison
- NumPy/Pandas implementation from scratch
- Manual gradient-descent optimization
- Comparison between Scikit-learn and manual implementations

---

## 📊 Dataset

The project uses the **UCI Productivity Prediction of Garment Employees** dataset.

The dataset contains information about garment-production teams, including features such as:

- Team
- Targeted productivity
- SMV
- WIP
- Overtime
- Incentive
- Idle time
- Number of workers
- Department
- Quarter
- Day
- Date
- and other production-related attributes.

The dataset contains **1,197 observations and 14 input features**.

### Missing Values

The `wip` feature contains missing values.

The missing values are handled using **median imputation**, with the median calculated only from the training data.

---

# 🎯 Machine Learning Tasks

Two separate machine-learning problems are solved.

## 1. Regression

The goal is to predict:

```text
actual_productivity
```

using **Linear Regression**.

### Evaluation Metrics

The regression models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 2. Classification

A binary target called `MeetsTarget` is created:

```text
MeetsTarget = 1
if actual_productivity >= targeted_productivity

MeetsTarget = 0
otherwise
```

The classification task predicts this binary target using **Logistic Regression**.

`actual_productivity` is not used as an input feature for classification because it is used to construct the target itself.

### Evaluation Metrics

The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

---

# ⚙️ Preprocessing

The following preprocessing steps are used.

### Numerical Features

1. Missing values are replaced using the training-set median.
2. Numerical features are standardized.

### Categorical Features

The following columns are treated as categorical:

```text
date
quarter
department
day
```

Categorical values are:

1. Imputed using the training-set mode if necessary.
2. Converted into numerical features using one-hot encoding.

### Data Leakage Prevention

All preprocessing parameters are learned from the **training set only**.

The test set is transformed using those already-learned parameters.

This ensures that information from the test set does not influence model training.

---

# 🧪 Part A — Scikit-learn Implementation

The Scikit-learn implementation uses:

- `ColumnTransformer`
- `Pipeline`
- `SimpleImputer`
- `OneHotEncoder`
- `StandardScaler`
- `LinearRegression`
- `LogisticRegression`

The preprocessing pipeline is fitted on the training data and then applied to the test data.

The same fixed train-test split is used for both regression and classification.

---

# 🔢 Part B — NumPy/Pandas From-Scratch Implementation

The second implementation recreates the machine-learning workflow using NumPy and Pandas.

No Scikit-learn model, preprocessing, or metric function is used in the manual implementation.

## Manual Preprocessing

The following operations are implemented manually:

- Median imputation
- Mode imputation
- Standardization
- One-hot encoding using Pandas
- Feature-matrix construction

---

## Manual Linear Regression

Linear Regression is implemented using the pseudo-inverse solution:

```text
θ = pinv(X) @ y
```

An additional column of ones is added to the feature matrix to represent the intercept.

Predictions are then obtained using:

```text
ŷ = Xθ
```

---

## Manual Logistic Regression

Logistic Regression is implemented using:

1. Linear combination
2. Sigmoid function
3. Gradient calculation
4. Gradient descent
5. Probability prediction
6. Thresholding at 0.5

The sigmoid function is:

```text
σ(z) = 1 / (1 + e^(-z))
```

The gradient is calculated using vectorized NumPy operations.

---

# 📈 Results

The experiments were performed using the same train-test split for both implementations.

## Linear Regression

| Model                          |      MAE |     RMSE |       R² |
| ------------------------------ | -------: | -------: | -------: |
| Scikit-learn Linear Regression | 0.112381 | 0.150048 | 0.152084 |
| Manual Linear Regression       | 0.112381 | 0.150048 | 0.152076 |

The regression results are almost identical.

This is expected because both implementations solve the same least-squares problem using the same processed data.

---

## Logistic Regression

| Model                            | Accuracy | Precision |   Recall |       F1 |
| -------------------------------- | -------: | --------: | -------: | -------: |
| Scikit-learn Logistic Regression | 0.741667 |  0.769953 | 0.926554 | 0.841026 |
| Manual Logistic Regression       | 0.754167 |  0.768182 | 0.954802 | 0.851385 |

The manual Logistic Regression produces slightly different results because its parameters are obtained using a manually implemented gradient-descent procedure, whereas Scikit-learn uses its own optimized solver and convergence procedure.

---

# ⏱️ Execution Time

The measured execution times were:

| Model                            | Training Time (s) | Prediction Time (s) |
| -------------------------------- | ----------------: | ------------------: |
| Scikit-learn Linear Regression   |          0.012088 |            0.000536 |
| Manual Linear Regression         |          0.046075 |            0.000145 |
| Scikit-learn Logistic Regression |          0.038209 |            0.000674 |
| Manual Logistic Regression       |          0.727859 |            0.000256 |

The exact execution times can vary depending on the machine and runtime environment.

### Observation

Scikit-learn's Logistic Regression trains considerably faster than the manual implementation.

The manual implementation performs thousands of gradient-descent iterations explicitly, while Scikit-learn uses an optimized solver implementation.

Prediction is much faster than training for both approaches because prediction mainly involves matrix operations.

---

# 🚀 Part C — Optimized Manual Logistic Regression

An additional experiment was performed by changing the Logistic Regression hyperparameters.

### Original configuration

```text
Learning rate = 0.01
Epochs        = 10,000
```

### Optimized experiment

```text
Learning rate = 0.05
Epochs        = 20,000
```

The resulting performance was:

| Metric        | Optimized Manual Logistic Regression |
| ------------- | -----------------------------------: |
| Accuracy      |                             0.733333 |
| Precision     |                             0.772947 |
| Recall        |                             0.903955 |
| F1            |                             0.833333 |
| Training Time |                           1.414946 s |

Interestingly, increasing the learning rate and number of epochs **did not improve the test-set performance**.

Compared with the original manual implementation:

```text
Original F1       = 0.851385
Optimized F1      = 0.833333
```

and:

```text
Original training time  = 0.727859 s
Optimized training time = 1.414946 s
```

This demonstrates that increasing the number of iterations or learning rate does not automatically result in better generalization. Hyperparameters need to be selected based on convergence and validation performance rather than simply increasing their values.

---

# 🔍 Key Observations

### 1. Linear Regression implementations are very similar

The Scikit-learn and manual Linear Regression models produce almost identical MAE, RMSE, and R² values.

This shows that the mathematical solution implemented manually is consistent with the library implementation.

### 2. Logistic Regression results differ slightly

The manual Logistic Regression uses our own gradient-descent implementation, while Scikit-learn uses a different optimized solver.

Therefore, the learned parameters and predictions are not required to be exactly identical.

### 3. Scikit-learn is faster for model training

The manual Logistic Regression requires many gradient-descent iterations.

Scikit-learn's implementation is optimized and therefore trains substantially faster.

### 4. Vectorization improves manual implementation

The gradient calculation is performed using NumPy matrix operations instead of iterating through individual samples.

This makes the manual implementation considerably more efficient than a completely element-by-element Python implementation.

### 5. More iterations do not guarantee better performance

The optimization experiment used 20,000 epochs instead of 10,000, but the test-set F1 score decreased.

Therefore, increasing epochs and learning rate blindly is not necessarily an effective optimization strategy.

### 6. The same train-test split is important

Both approaches use the same training and testing samples.

This makes the comparison between Scikit-learn and the manual implementation more meaningful.

---

# 📁 Project Structure

A recommended repository structure is:

```text
ML-Lab/
│
├── Lab_6.ipynb
├── README.md
└── data/
    └── garment_productivity.csv
```

If the dataset is obtained programmatically using `ucimlrepo`, the dataset file does not need to be stored in the repository.

---

# 🛠️ Requirements

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn ucimlrepo jupyter
```

---

# ▶️ How to Run

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project:

```bash
cd <repository-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Lab_6.ipynb
```

and run the cells from top to bottom.

---

# 📚 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- UCI Machine Learning Repository
- Jupyter Notebook

---

# 👤 Author

**[Your Name]**

DS605 — Fundamentals of Machine Learning

---

# 📄 Assignment

This project was completed as part of the **DS605 — Fundamentals of Machine Learning** laboratory assignment on Machine Learning with Scikit-learn and From Scratch.
