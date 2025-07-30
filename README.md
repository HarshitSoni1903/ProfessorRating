# Bias & Performance Analysis in Student Ratings

This project analyzes patterns and potential biases in student ratings of university professors, based on various attributes like gender, course tags, and qualitative feedback. Using a combination of statistical tests, regression models, and classification metrics, it explores questions such as:
- Are there gender-based differences in ratings?
- Which features significantly influence overall scores?
- Can student sentiment be predicted from rating and tag data?

Each question is implemented in a dedicated module. The `main.py` script serves as the orchestrator, loading datasets and executing all analysis steps sequentially.

---

## Project Structure

---

## Shared Utilities (`imports.py`)

This file contains:

- **Core Libraries**:  
  - `pandas`, `numpy` for data manipulation  
  - `matplotlib` for visualization  
  - `scipy.stats` for statistical tests  
  - `sklearn` for regression and classification  
  - `imblearn` for oversampling via SMOTE

- **Helper Function**:
  - `ecdf(data)`: Returns x, y arrays for empirical cumulative distribution plot.

- **Feature Mappings**:
  - `numname`: Maps column indices to human-readable names for numerical features.
  - `tagname`: Maps indices to tag descriptions like `"Tough grader"`, `"Hilarious"`, etc.

---

## Main Script (`main.py`)

The main execution pipeline does the following:

1. **Data Loading**:  
   - `rmpCapstoneNum.csv` → numerical features  
   - `rmpCapstoneTags.csv` → tag indicators  
   - `rmpCapstoneQual.csv` → sentiment scores from text

2. **Module Execution**:  
   - `q1`: Mann-Whitney U test and ECDF to analyze rating bias by gender  
   - `q4`: Identifies top/bottom 3 tag features by p-value  
   - `q5`: Mann-Whitney U test on difficulty scores  
   - `q7`, `q8`, `q9`: Linear regression with different sets of predictors  
   - `q10`: Logistic regression with AUROC and classification report  
   - `ec`: SMOTE-based oversampling and logistic regression using sentiment labels

3. **Outputs Printed**:  
   - P-values and significance statements  
   - Cohen’s d effect sizes  
   - RMSE and R² values  
   - Most influential features by model weights  
   - AUROC, confusion matrix, and classification report for the classification task

---
## Module Descriptions

### `q1.py` – Q1 to Q3: Gender Bias in Ratings

This module investigates whether male instructors receive systematically higher ratings than female instructors. It performs three distinct analyses:

- **Q1: Mann-Whitney U Test**
  - Tests the null hypothesis that male and female ratings come from the same distribution.
  - Uses `alternative='greater'` to check for pro-male bias.
  - Filters out ambiguous gender data (e.g., both male and female indicators set identically).
  - Visualizes results via a **boxplot**.

- **Q2: Kolmogorov-Smirnov Test**
  - Assesses whether the full distribution of ratings differs significantly.
  - Visualizes results using **Empirical CDFs** for male and female instructors.

- **Q3: Cohen’s d Effect Size**
  - Measures the magnitude of the difference in average ratings.
  - Bar plots are generated to compare both **mean** and **standard deviation** between groups.

**Returns**:  
Test statistics and p-values from the U test and KS test, and Cohen's d.

---
### `q4.py` – Q4: Tag Feature Significance by Gender

This module examines whether specific tag-based features are more commonly associated with one gender over the other by:

- Merging numerical data (rating counts and gender) with tag indicators.
- Normalizing tag counts by the number of ratings per entry.
- Filtering data to include only clearly gendered records and those with more than 5 ratings.
- Running **Mann-Whitney U tests** for each tag to compare male vs. female distributions.
- Storing results in a dictionary mapping p-values to tag indices (adjusted for column alignment).

**Used in `main.py` to**:
- Identify the **Top 3 most significant** and **Bottom 3 least significant** tag features based on p-value ranking.

---
### `q5.py` – Q5 & Q6: Gender Bias in Perceived Difficulty

This module explores whether there's a significant difference in the perceived difficulty ratings of male vs. female instructors.

- **Q5: Kolmogorov-Smirnov Test**
  - Filters records to include only instructors with distinct gender labels and more than 5 ratings.
  - Compares the difficulty score distribution (column index 1) for male and female instructors.
  - Uses a two-sided **KS test** to check for differences in distribution.

- **Q6: Cohen’s d**
  - Computes the effect size to quantify the magnitude of the rating difference.

**Returns**:  
- KS test p-value  
- Cohen’s d value
---
### `q7.py` – Q7: Linear Regression on Numeric Features

This module builds a **linear regression model** to predict the average rating (column 0) using all other numerical features.

- Preprocessing:
  - Filters out rows with null values and ambiguous gender data.
  - Splits the dataset into training and testing subsets.
  - Scales features using `StandardScaler`.

- Modeling:
  - Trains a linear regression model using `sklearn.LinearRegression`.
  - Predicts on the test set and evaluates model performance using:
    - **RMSE (Root Mean Squared Error)**
    - **R² (Coefficient of Determination)**

- Visualization:
  - Plots model weights to show the influence of each feature.
  - Identifies the feature with the largest absolute weight (most impact on prediction).

**Returns**:  
RMSE, R² score, index of the most impactful feature, and its weight.

---
### `q8.py` – Q8: Linear Regression Using Tag Features

This module builds a linear regression model to predict average ratings based on tag-based features (e.g., "Inspirational", "Tough Grader").

- **Preprocessing**:
  - Combines relevant numerical columns (e.g., average rating, gender indicators, rating count) with binary tag data.
  - Normalizes tag values by the number of ratings for each entry.
  - Filters out ambiguous gender cases and entries with fewer than 5 ratings.

- **Modeling**:
  - Uses `LinearRegression` with `StandardScaler` for feature normalization.
  - Splits data into training and testing sets.
  - Trains the model and evaluates it using **RMSE** and **R² score**.

- **Visualization**:
  - Displays a bar chart of model weights for each tag.
  - Identifies the tag with the highest absolute weight (strongest influence on rating prediction).

**Returns**:  
RMSE, R² score, most influential tag index, and its weight.
---

### `q9.py` – Q9: Predicting Difficulty Using Tag Features

This module builds a linear regression model to predict perceived **difficulty ratings** (instead of average ratings) using tag-based features.

- **Preprocessing**:
  - Merges difficulty score (column 1), rating count, gender indicators, and tag data.
  - Filters for valid gender labels and minimum rating thresholds.
  - Normalizes tag data by number of ratings.

- **Modeling**:
  - Splits the data into training and test sets.
  - Applies feature scaling and trains a `LinearRegression` model.
  - Evaluates using **RMSE** and **R²** metrics.

- **Visualization**:
  - Displays a bar plot of model weights for all tag features.
  - Identifies the most impactful tag by absolute weight.

**Returns**:  
RMSE, R² score, index of the most impactful tag, and its weight.

---

### `q10.py` – Q10: Logistic Regression on Pepper Indicator

This module builds a **logistic regression model** to predict whether a professor received a "pepper" (a binary popularity marker) using both numeric and tag-based features.

- **Preprocessing**:
  - Merges numerical and tag features.
  - Uses column 3 (pepper indicator) as the binary target.
  - Splits the data into training and test sets.
  - Applies **SMOTE** to handle class imbalance in the target.
  - Standardizes features using `StandardScaler`.

- **Modeling**:
  - Trains a `LogisticRegression` classifier.
  - Evaluates model performance with:
    - **ROC Curve** and **AUC score**
    - **Confusion Matrix**
    - **Classification Report** (precision, recall, F1-score)

- **Visualization**:
  - Side-by-side plots of ROC curve and confusion matrix for interpretability.

**Returns**:  
AUROC score and classification report.
---

### `ec.py` – Extra Credit: STEM vs Non-STEM Sentiment Bias

This module explores whether there's a difference in average ratings between STEM and non-STEM instructors based on qualitative field labels.

- **STEM Classification**:
  - Identifies STEM fields by matching keywords like `"comput"`, `"math"`, `"physics"`, `"tech"`, `"info"`, etc. in course descriptions.
  - Creates two groups: `STEM` and `Non-STEM`.

- **Statistical Testing**:
  - Applies **Mann-Whitney U test** to compare rating distributions between the two groups.

- **Sampling Strategy**:
  - Applies **random undersampling** to reduce class imbalance before re-running ECDF plots and analysis.

- **Visualization**:
  - Side-by-side **Empirical CDF plots** (before and after undersampling) for visual comparison of rating distributions.

**Returns**:  
Mann-Whitney U test statistic and p-value.

---
## How to Run

```bash
python main.py

