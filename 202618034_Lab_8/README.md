# DS605 Lab 8 - Decision Trees & SVMs (UCI Banknote Authentication)

Decision Tree and Linear SVM implemented **from scratch (NumPy/Pandas)** and compared with Scikit-learn; kernel SVMs explored on nonlinear data.
Data: 1,372 samples, 4 features (variance, skewness, curtosis, entropy), 2 classes. One stratified 80/20 split (1097 train / 275 test, `random_state=42`) is used everywhere. 2D plots use **variance + skewness**; final evaluation uses **all 4 features**.

## Run
```bash
pip install -r requirements.txt
jupyter notebook DS605_Lab8_Decision_Tree_SVM.ipynb   # Run All
```
The dataset downloads automatically. Figures are saved to `plots/`, tables to `results/`.

## Repository
| Path | Content |
|---|---|
| `DS605_Lab8_Decision_Tree_SVM.ipynb` | Full implementation with outputs (`.py` = same code as a script) |
| `plots/` | Boundaries, manual tree drawing, margin plots, kernel plots, final comparison |
| `results/` | CSV tables (tree rules, depth/C/lr/iteration studies, final metrics) |

## Final comparison (4 features, test set)
| Model | Acc | Precision | Recall | F1 | Train (s) | Predict (ms) |
|---|---|---|---|---|---|---|
| Manual Decision Tree (gini, depth 5) | 0.9927 | 0.9839 | 1.0 | 0.9919 | 0.0184 | 0.19 |
| Sklearn Decision Tree (same params) | 0.9927 | 0.9839 | 1.0 | 0.9919 | 0.0043 | 0.26 |
| Manual Linear SVM (C=10) | 0.9709 | 0.9385 | 1.0 | 0.9683 | 0.2931 | 0.02 |
| Sklearn Linear SVM (C=10) | 0.9855 | 0.9683 | 1.0 | 0.9839 | 0.0152 | 0.56 |
| Sklearn RBF SVM | 1.0000 | 1.0000 | 1.0 | 1.0000 | 0.0071 | 2.77 |

## Key observations
**Decision Tree (Part A)**
- The manual tree matches Scikit-learn: identical accuracy and leaf counts for depth ≤ 5, and 99-100% prediction agreement overall (tiny differences only from tie-breaking between equal splits).
- On two features, test accuracy peaks at depth 4-5 (~0.93) while train accuracy keeps rising to 1.0 - deeper trees overfit. Gini and entropy behave almost the same.
- On four features, depth 5 reaches 0.993 test accuracy with just 20 leaves; deeper trees add nothing.

**Linear SVM (Part B)**
- Manual SVM converges to nearly the same solution as Scikit-learn (cosine similarity of `w` ≈ 1.0, 98.9% agreement, objective 313.2 vs 312.0).
- **C:** larger C → narrower margin (width 3.40 → 0.92) and fewer margin violators (911 → 337); test accuracy barely changes (~0.87-0.89).
- **Learning rate:** 0.0005 is far too slow (objective ≈ 1002 after 1000 iterations); ≥ 0.5 reaches the optimum (≈ 312).
- **Iterations:** objective drops from 712 (5 iters) to 313 (5000); accuracy plateaus early. Two features cannot be separated linearly (~0.87-0.88), unlike four features (0.97-0.99).

**Kernel SVM (Part C)**
- `make_circles`: linear kernel fails (test 0.59) while polynomial and RBF reach 1.00. `make_moons`: linear 0.89, polynomial 0.92, RBF 0.93.
- Nonlinear kernels implicitly map data to a higher-dimensional space where a linear separator exists, giving curved boundaries in the original space.

**Overall (Part D)**
- All models have recall 1.0; every error is a false positive. RBF SVM is best (100% on this test set, a small split so treat as "near-perfect"), followed by both trees (0.993).
- **Interpretability:** the tree is readable rule-by-rule; linear SVM has only 5 parameters; RBF SVM (91 support vectors) is a black box.
- **Scaling:** SVMs need standardisation (training statistics); trees are scale-invariant.
- **Overfitting/complexity:** unconstrained trees reach 100% train accuracy; RBF `gamma=50` overfits (test 0.982, 982 support vectors); linear SVM is stable across C.
- **Speed:** manual SVM trains slowest (0.29 s, plain Python loop) but predicts fastest; RBF predicts slowest because it evaluates every support vector.
