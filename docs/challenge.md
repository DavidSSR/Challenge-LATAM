# All Parts Software Engineer (ML & LLMs) Challenge

## # Part I

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

# Part II

The second part involves deploying the model as an **API** using **FastAPI** to predict flight delays. The API consumes a request containing flight data and returns the predicted delays. If any of the values in the request are incorrect, such as invalid operators, flight types, or months, the API will raise an `HttpException` with a 400 status code. Otherwise, it processes the data, runs the prediction using the trained model, and returns the result.

Additionally, the following methods were implemented in the **`api.py`** file:

- **`check_opera_exists`, `check_tipovuelo_exists`, `check_mes_exists`**: These validation methods ensure that the values for each of the required columns (`OPERA`, `TIPOVUELO`, `MES`) are valid. If any value is incorrect, an `HttpException` is raised.

- **`post_predict`**: This method handles the incoming POST request, validates the flight data, preprocesses it, and returns the delay predictions from the model.

# Part III

In this section, you will find the Dockerized app deployed on GCP. The Dockerfile is configured to use Python 3.9 to avoid errors and additional build time related to package dependencies like numpy and pandas. This Python version helps optimize the process of building and deploying the image to GCP Container Registry. The app is deployed at the following URL: https://challenge-latam-1006538092866.us-central1.run.app .

During the stress test, there were no failures, with an average time of 450ms per request, handling 4477 requests in total. Below are the response time percentiles:


## Response time percentiles (approximated)

| Type      | Name      | 50% | 66% | 75% | 80% | 90% | 95% | 98% | 99% | 99.9% | 99.99% | 100% | # reqs |
|-----------|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-------|--------|------|--------|
| POST      | /predict  | 360 | 450 | 500 | 560 | 630 | 680 | 740 | 760 | 800   | 900    | 900  | 4477   |
| Aggregated|           | 360 | 450 | 500 | 560 | 630 | 680 | 740 | 760 | 800   | 900    | 900  | 4477   |


# Part IV
## CI/CD Pipeline Overview
This pipeline automates testing, building, and deployment for the develop and main branches.

### CI Pipeline
**Trigger:** Runs on push to the develop branch.

1. **Build & Test:**
   - Build `Dockerfile.tests` image to run tests.

2. **Build & Push Dev Image:**
   - Build and push development image to GCP Artifact Registry.

3. **Deploy to Cloud Run (Dev):**
   - Deploy image to Cloud Run for development testing.

4. **Stress Test:**
   - Build and run `Dockerfile.stress` for performance testing.

5. **Release Branch:**
   - Create release branch and open a PR to main.

### CD Pipeline
**Trigger:** Runs when the release PR is merged into main.

1. **Build & Push Prod Image:**
   - Build and push production image to GCP Artifact Registry.

2. **Deploy to Cloud Run (Prod):**
   - Deploy image to Cloud Run for production.

3. **Output Prod URL:**
   - Print the deployed production URL for verification.

This CI/CD setup automates testing, building, and deployment, ensuring production-ready code is always delivered efficiently.
