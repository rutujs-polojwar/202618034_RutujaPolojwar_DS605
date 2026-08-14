# Lab 2 – NumPy, Pandas and Data Analysis

## Overview

This assignment focuses on using Python for numerical computing, data manipulation, statistical analysis, and visualization.

The work is divided into two parts:

- **Part A:** NumPy operations, vectorized programming, linear algebra, and normal distribution.
- **Part B:** Exploratory Data Analysis (EDA) using the Titanic dataset.

## Part A – NumPy

The following concepts were implemented:

- Creation and manipulation of 1D, 2D, and 3D NumPy arrays.
- Statistical operations such as minimum, maximum, mean, median, and standard deviation.
- Array creation using `arange()`, `zeros()`, `ones()`, and `linspace()`.
- Indexing, slicing, reshaping, and flattening arrays.
- Matrix addition, element-wise multiplication, and matrix multiplication.
- Transpose, determinant, and inverse of matrices.
- Generation and analysis of 1,000 values from a normal distribution.
- Visualization of the normal distribution using a histogram.

## Part B – Titanic Dataset Analysis

The Titanic dataset was explored using Pandas and visualization libraries.

The analysis included:

- Dataset inspection and descriptive statistics.
- Missing-value count and percentage analysis.
- Missing-value imputation using mean, median, mode, and random values where applicable.
- Detection of Fare outliers using the IQR method.
- Feature engineering using `FamilySize` and `IsAlone`.
- Grouping and pivot-table analysis based on Sex and Passenger Class.
- Correlation analysis using a heatmap.
- Visualization of survival rates by Sex.
- Age vs Fare analysis with survival status.

## Key Findings

- Female passengers had a much higher survival rate (**74.20%**) than male passengers (**18.89%**).
- First-class passengers had the highest survival rate (**62.96%**), while third-class passengers had the lowest (**24.24%**).
- Female first-class passengers had the highest survival rate (**96.81%**), while male third-class passengers had the lowest (**13.54%**).
- `Pclass` and `Fare` showed the strongest negative correlation (**-0.5495**).
- `SibSp` and `Parch` showed the strongest positive correlation (**0.4148**).
- `Cabin` had the highest percentage of missing values (**77.10%**).

## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Conclusion

The assignment demonstrates the use of Python and its data-analysis libraries for numerical computation, statistical analysis, data cleaning, feature engineering, and visualization.
