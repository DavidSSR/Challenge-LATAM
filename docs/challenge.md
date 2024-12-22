# All Parts Software Engineer (ML & LLMs) Challenge

## Overview

This is the first part of the **Software Engineer (ML & LLMs) Application Challenge**. In this phase, the model from the `.ipynb` notebook was transcribed into the `model.py` file. Each function in the `model.py` file is properly documented to explain its functionality. Additional lines were added to enable testing.

## Model Selection and Analysis

A **Logistic Regression** model was chosen for training and prediction. The decision was based on the following observations:

### 1. Linearity
- The data was found to be linear.
- Using a statistical linearity test from scipy, the p-value for each attribute of the top 10 features was less than 0.05, confirming linearity.
- A low p-value (< 0.05) indicates that the null hypothesis—"no linear relationship between the feature and the target variable"—can be rejected, suggesting a significant linear relationship between the features and the target variable (delay).

### 2. Feature Correlation
- Using the .corr() function of a DataFrame, most correlation values were close to 0, indicating minimal or no correlation among features.
- Low feature correlation ensures that multicollinearity is not a concern, making the data suitable for logistic regression.

### 3. Model Justification
- Logistic Regression works well with linear data and non-correlated features.
- It is relatively fast to train and deploy, making it an efficient choice.

## Bug Fixes

Several bugs were identified and addressed during this phase:

### 1. `exploration.ipynb`
- The code for plotting required the `x` and `y` attributes to be declared. These attributes were added to each plot to ensure proper functionality.

### 2. `model.py`
- A typo was found in the `preprocess` function where the `Union` object was incorrectly defined using parentheses `()` instead of brackets `[]`. This was corrected to ensure compatibility and proper operation.