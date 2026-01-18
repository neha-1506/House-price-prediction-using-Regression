# House-price-prediction-using-Regression

---

# Case Study: Predicting House Prices Using Linear Regression

---

## 1. Introduction

In the real estate industry, accurately estimating house prices is essential for buyers, sellers, investors, and policy makers. Property valuation depends on multiple measurable factors such as house size, number of rooms, age of the building, neighborhood facilities, and safety conditions. With the increasing availability of structured housing data, **data-driven predictive models** can assist in making informed pricing decisions.

This case study focuses on developing a **Linear Regression model** to predict house prices based solely on **numerical property and neighborhood attributes**. Linear Regression is chosen due to its simplicity, interpretability, and effectiveness in modeling linear relationships between variables. The project also aims to identify the most influential factors affecting house prices and evaluate model performance using standard regression metrics.

---

## 2. Case Study Background

A real estate analytics firm wants to estimate the selling price of residential houses using historical property data. Each property is described using numerical features related to:

* Physical characteristics of the house
* Accessibility and location
* Nearby amenities
* Safety of the neighborhood

The firm wants a model that:

* Produces accurate price predictions
* Explains how each factor affects the price
* Can be easily interpreted by business stakeholders

---

## 3. Problem Statement

To build a predictive model that estimates the **house price (in $1000s)** using numerical features such as area, number of rooms, distance from city center, and neighborhood conditions.

---

## 4. Objectives

* Build a Linear Regression model to predict house prices
* Analyze the impact of each numerical feature on price
* Evaluate model performance using:

  * R² Score
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)

---

## 5. Dataset Description

| Feature                 | Description                        |
| ----------------------- | ---------------------------------- |
| area_sqft               | Total built-up area in square feet |
| bedrooms                | Number of bedrooms                 |
| bathrooms               | Number of bathrooms                |
| age_years               | Age of the house in years          |
| distance_city_center_km | Distance to city center            |
| num_schools_nearby      | Schools within 3 km                |
| crime_rate_index        | Crime index (higher = worse)       |
| price                   | House price in $1000s              |

---

## 6. Methodology (Case Study Approach)

1. Data Loading and Understanding
2. Data Cleaning and Validation
3. Exploratory Data Analysis (EDA)
4. Correlation Analysis
5. Train-Test Split
6. Model Building using Linear Regression
7. Model Evaluation
8. Interpretation of Results

---

# 7. Implementation (With Code and Explanation)

---

## Step 1: Importing Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
```

**Explanation:**
These libraries are used for data manipulation, numerical calculations, visualization, and clean output generation.

---

## Step 2: Loading the Dataset

```python
data = pd.read_excel("Houses.xlsx", index_col=0)
data.head()
```

**Explanation:**
The dataset is loaded from an Excel file. The first column is treated as an index.

---

## Step 3: Understanding Data Structure

```python
data.info()
```

**Explanation:**
Confirms that all variables are numerical and checks for missing values.

---

## Step 4: Checking Missing Values

```python
data.isnull().sum()
```

**Explanation:**
Ensures the dataset is complete. No missing values means no imputation is required.

---

## Step 5: Descriptive Statistics

```python
data.describe()
```

**Explanation:**
Provides statistical insights such as mean, minimum, maximum, and variability of features.

---

## Step 6: Exploratory Data Analysis (EDA)

### Distribution of House Prices

```python
sns.histplot(data["price"], kde=True)
plt.title("Distribution of House Prices")
plt.show()
```

**Explanation:**
Visualizes the distribution of house prices and checks normality.

---

### Pairwise Relationships

```python
sns.pairplot(data)
plt.show()
```

**Explanation:**
Identifies linear relationships and outliers between variables.

---

## Step 7: Correlation Analysis

```python
corr = data.corr(method="pearson")

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
```

**Explanation:**
Highlights which factors have strong positive or negative relationships with price.

---

## Step 8: Defining Features and Target Variable

```python
X = data.drop("price", axis=1)
y = data["price"]
```

**Explanation:**
Separates independent variables from the target variable.

---

## Step 9: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Explanation:**
Splits the dataset into training and testing sets to evaluate model performance on unseen data.

---

## Step 10: Model Building

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

**Explanation:**
Fits a Linear Regression model to the training data.

---

## Step 11: Model Coefficients Interpretation

```python
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients
```

**Explanation:**
Shows how each feature impacts the house price.

---

## Step 12: Making Predictions

```python
y_pred = model.predict(X_test)
```

**Explanation:**
Generates price predictions for the test dataset.

---

## Step 13: Model Evaluation

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
```

**Explanation:**
Evaluates accuracy and prediction error.

---

## 8. Results and Insights

* Area and number of bedrooms have a strong positive impact on house price
* Distance from city center and crime rate negatively affect prices
* The model explains a significant portion of price variability
* Error metrics indicate acceptable prediction accuracy

---

## 9. Conclusion

This case study demonstrates the successful application of Linear Regression for predicting house prices using numerical data. The model provides both accurate predictions and meaningful insights into the factors influencing property valuation. Due to its interpretability and simplicity, this model is suitable for business and academic applications.

---

## 10. Future Scope

* Include categorical variables such as location type
* Apply advanced models like Ridge, Lasso, or Random Forest
* Perform feature scaling and cross-validation


