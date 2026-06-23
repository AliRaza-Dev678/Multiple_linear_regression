# 💼 Multiple Linear Regression — Salary Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A supervised machine learning project implementing **Multiple Linear Regression** to predict employee salaries based on multiple input features — covering the full regression pipeline from EDA and assumption checking to model training, coefficient interpretation, and performance evaluation.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What is Multiple Linear Regression?](#-what-is-multiple-linear-regression)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Concepts Covered](#-key-concepts-covered)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

Salary prediction is a classic and highly practical regression problem. This project builds a **Multiple Linear Regression** model to estimate employee salaries from multiple features such as years of experience, education level, job role, and more. Unlike simple linear regression (one predictor), MLR models the combined effect of several variables simultaneously — making it far more reflective of real-world scenarios. The complete pipeline is implemented in a Jupyter Notebook.

---

## 🧠 What is Multiple Linear Regression?

**Multiple Linear Regression (MLR)** extends simple linear regression to model the relationship between a **continuous target variable** and **two or more predictor features**.

```
ŷ = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

| Symbol | Meaning |
|---|---|
| `ŷ` | Predicted target value (e.g., salary) |
| `β₀` | Intercept — predicted value when all features are zero |
| `β₁...βₙ` | Coefficients — change in `ŷ` for a one-unit increase in each feature |
| `X₁...Xₙ` | Input features (e.g., experience, education) |
| `ε` | Error term — irreducible noise in the data |

**How the model learns:**
The model minimizes the **Residual Sum of Squares (RSS)** using **Ordinary Least Squares (OLS)** — finding the coefficient values that produce the smallest total prediction error across all training samples.

**Key assumptions of Multiple Linear Regression:**

| Assumption | Description |
|---|---|
| **Linearity** | The relationship between features and target is linear |
| **Independence** | Observations are independent of each other |
| **Homoscedasticity** | Residuals have constant variance across all predictions |
| **Normality of Residuals** | Residuals are approximately normally distributed |
| **No Multicollinearity** | Predictor features are not highly correlated with each other |

---

## 📊 Dataset

The **Salary Dataset** contains employee records with multiple features used to predict their salary.

| Property | Details |
|---|---|
| **Task** | Regression (Continuous target prediction) |
| **Target Variable** | Salary (in currency units) |
| **Features** | Multiple employee attributes (e.g., experience, education, role) |
| **Type** | Structured tabular data |

### Likely Features

| Feature | Description |
|---|---|
| `Years of Experience` | Total years the employee has worked |
| `Education Level` | Highest degree attained |
| `Job Role / Title` | Position or department |
| `Age` | Age of the employee |
| `Gender` | Employee gender |
| **`Salary`** | **Target variable — annual salary to predict** |

> 📖 Refer to the notebook for the exact feature list, data types, and exploratory analysis outputs.

---

## ⚙️ Methodology

### 1. Data Loading & Exploration
- Load the Salary Dataset using Pandas
- Inspect shape, data types, and descriptive statistics
- Check for missing values and duplicate records
- Understand the distribution of the target variable (salary)

### 2. Exploratory Data Analysis (EDA)
- **Scatter plots** to visualize relationships between each feature and salary
- **Histograms** to check feature distributions
- **Correlation heatmap** to identify multicollinearity between predictors
- **Box plots** to detect outliers in key features

### 3. Data Preprocessing
- Handle missing values (imputation or removal)
- Encode categorical variables using **Label Encoding** or **One-Hot Encoding**
- Apply **feature scaling** (StandardScaler / MinMaxScaler) where needed
- Split data into **training (80%) and test (20%) sets**

### 4. Assumption Checking
- Verify **linearity** via scatter plots of features vs target
- Check **multicollinearity** using a correlation matrix or VIF (Variance Inflation Factor)
- Inspect **residual plots** for homoscedasticity
- Test **normality of residuals** using Q-Q plots or histograms

### 5. Model Training
- Instantiate and fit a **Linear Regression** model from scikit-learn
- The model uses **Ordinary Least Squares (OLS)** to estimate coefficients
- Inspect learned **coefficients and intercept** to understand feature impact

### 6. Prediction & Evaluation
- Predict salaries on the test set
- Evaluate using regression metrics:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score (Coefficient of Determination)**

### 7. Visualization
- **Actual vs Predicted** salary scatter plot
- **Residual plot** to assess error patterns
- **Feature coefficient bar chart** to show which features drive salary most

---

## 📚 Key Concepts Covered

| Concept | Description |
|---|---|
| **Multiple Linear Regression** | Regression with two or more predictor features |
| **Ordinary Least Squares (OLS)** | Method for estimating model coefficients by minimizing RSS |
| **Coefficients (β)** | Quantify the effect of each feature on the target |
| **R² Score** | Proportion of variance in salary explained by the model (0–1) |
| **Residuals** | Difference between actual and predicted values |
| **Multicollinearity** | When predictor features are highly correlated — can distort coefficients |
| **Homoscedasticity** | Uniform spread of residuals — a key regression assumption |
| **Feature Encoding** | Converting categorical variables to numeric form for the model |
| **MAE / RMSE** | Metrics measuring average prediction error in original salary units |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Jupyter Notebook** | Interactive development environment |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Regression plots and residual charts |
| **Seaborn** | Correlation heatmaps and distribution plots |
| **scikit-learn** | Linear Regression model, preprocessing, and evaluation metrics |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ and pip installed.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AliRaza-Dev678/Multiple_linear_regression.git
cd Multiple_linear_regression

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook Multiple_linear_regression.ipynb
```

Open the notebook in your browser and run all cells from top to bottom.

---

## 📁 Project Structure

```
Multiple_linear_regression/
│
├── Multiple_linear_regression.ipynb    # Main notebook — full pipeline
└── README.md                           # Project documentation
```

---

## 📈 Model Performance

Multiple Linear Regression performance on salary data depends on how well the selected features explain salary variance. A well-fitted model typically achieves:

| Metric | Description |
|---|---|
| **MAE** | Average absolute error between predicted and actual salary |
| **MSE** | Average squared error — penalizes large errors more heavily |
| **RMSE** | Square root of MSE — expressed in the same units as salary |
| **R² Score** | Closer to 1.0 = better fit; indicates % of salary variance explained |

> A strong MLR model on the Salary Dataset typically achieves **R² ≥ 0.85**, depending on the features included.
>
> Run the notebook to see exact metric values for this dataset.

---

## 🔭 Future Improvements

- Apply **Polynomial Regression** to capture non-linear salary relationships
- Use **Ridge** and **Lasso Regression** to handle multicollinearity and perform feature selection
- Compute **Variance Inflation Factor (VIF)** to formally detect and resolve multicollinearity
- Implement **cross-validation (k-fold)** for more robust R² estimates
- Add **interaction terms** to capture combined effects of features (e.g., Experience × Education)
- Compare against non-linear models — **Random Forest Regressor**, **Gradient Boosting**
- Deploy the model as a **Streamlit salary prediction app**

---

## 👤 Author

**Ali Raza**

[![GitHub](https://img.shields.io/badge/GitHub-AliRaza--Dev678-black?style=flat-square&logo=github)](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
