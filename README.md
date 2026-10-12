
# Predicting Diabetes Risk Using NHANES Data

**End-to-End Healthcare Machine Learning Project | Python · scikit-learn · TensorFlow/Keras · Flask**

## Project Overview

This project uses publicly available data from the CDC's [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/) to estimate the probability of **diabetes or borderline diabetes** using demographic, behavioral, and physical characteristics.

The objective was to develop and evaluate machine learning models using information individuals can provide without laboratory testing, then deploy the selected model through an interactive Flask web application.

The project demonstrates an end-to-end data science workflow, including data integration, exploratory analysis, preprocessing, model comparison, hyperparameter optimization, probability calibration, and deployment.

## Dataset and Preprocessing

Six NHANES datasets were integrated:

- **DEMO_L:** Demographics
- **BMX_L:** Body measurements
- **SMQ_L:** Smoking history
- **ALQ_L:** Alcohol consumption history
- **PAQ_L:** Physical activity
- **DIQ_L:** Diabetes questionnaire (target variable)

Predictors included age, sex, race/ethnicity, BMI, income relative to the poverty line, smoking history, alcohol consumption history, physical activity, and sedentary behavior.

**Data preparation included:**

- Cleaning NHANES response codes and handling missing values.
- Merging datasets and restricting the analysis to adults.
- Feature engineering and categorical encoding.
- Numerical preprocessing and custom income imputation.
- Stratified 80/20 training and testing split.

Model comparisons were performed using five-fold cross-validation on the training data.

## Models Evaluated

Five machine learning model types were trained and evaluated:

1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Decision Tree Classifier
4. Gradient Boosting
5. Neural Network (TensorFlow/Keras)

Performance was evaluated using ROC AUC, Log Loss, and Brier Score to assess both discrimination and probability prediction quality.

### Initial Model Comparison

| Model | ROC AUC | Log Loss | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.774 | 0.377 | 0.119 |
| Support Vector Classifier | 0.678 | 0.424 | 0.130 |
| Decision Tree | 0.736 | 0.719 | 0.131 |
| Gradient Boosting | **0.776** | 0.378 | 0.119 |

A TensorFlow/Keras neural network was also evaluated using hidden layers of 32 and 64 neurons. Increasing the network capacity did not improve performance.

### Hyperparameter Optimization

GridSearchCV with five-fold cross-validation was used to optimize Gradient Boosting and Logistic Regression.

| Optimized Model | ROC AUC | Log Loss | Brier Score |
|---|---:|---:|---:|
| **Gradient Boosting** | **0.781** | **0.373** | **0.117** |
| Logistic Regression | 0.775 | 0.377 | 0.118 |
| Neural Network (32 neurons) | 0.772 | 0.379 | 0.119 |

Gradient Boosting achieved the strongest overall performance and was selected as the final model.

## Final Model Performance

The optimized Gradient Boosting model was retrained using the complete training dataset and evaluated on the held-out test set.

| Metric | Test Score |
|---|---:|
| **ROC AUC** | **0.780** |
| **Log Loss** | **0.371** |
| **Brier Score** | **0.115** |

The final model demonstrated moderate discrimination between individuals with and without positive status, while producing reasonably well-calibrated probability estimates.

### Calibration Analysis

Calibration was evaluated by comparing predicted probabilities against observed diabetes rates.

- Predicted and observed rates were generally within 2 percentage points across the most populated probability ranges.
- Most predictions fell below 50% probability.
- Higher probability ranges contained relatively few observations, limiting confidence in calibration at those levels.

These findings suggest that the model provides reasonably useful probability estimates, particularly within the ranges most frequently represented in the test data.

## Model Experimentation

Hypothetical individuals were created to investigate how changes in demographic, behavioral, and physical characteristics affected predicted probabilities.

Key observations included:

- **Age and BMI** produced substantial changes in predicted diabetes probability.
- Combinations of multiple risk factors generally produced higher predicted probabilities.
- Changes in physical activity and sedentary behavior influenced predictions even when other characteristics remained constant.

For example, two otherwise identical hypothetical 65-year-old individuals received predicted probabilities of **34.7% and 54.9%** when exercise frequency and sedentary behavior were varied.

These experiments illustrate patterns learned by the model and do not establish causal relationships.

## Flask Web Application

An interactive **Diabetes Risk Estimator** was developed using Flask to demonstrate deployment of the final Gradient Boosting model.

The application allows users to enter demographic, behavioral, and physical characteristics and receive an estimated probability of diabetes or borderline diabetes.

A simplified preprocessing pipeline was combined with the trained model and serialized using joblib.

The deployment pipeline was validated against the original modeling workflow using **35 hypothetical observations**, producing matching predictions.

## Project Structure

- **Jupyter Notebook (`diabetes-project.ipynb`)** — Data preprocessing, exploratory analysis, model development, and evaluation.
- **`diabetes-project.html`** — Complete analysis and results, viewable without running Jupyter.
- **`flask/`** — Flask application, HTML templates, CSS, and serialized model.
- **`data/`** — NHANES datasets.
- **`codebooks/`** — NHANES documentation of the variables.
- **`images/`** — Flask app visualizations.
- **`requirements.txt`** — Python dependencies.
- **`environment.yml`** — Conda environment configuration.

## Running the Flask Application

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Navigate to the Flask application directory:

```bash
cd flask
python app.py
```

Open the local address displayed in the terminal.

The project was developed on an Apple Silicon Mac. TensorFlow installation requirements may differ across operating systems.

## Limitations

This project is intended for educational and portfolio purposes and has not been clinically validated.

Model predictions should not be used for medical diagnosis or treatment decisions.

Although the model demonstrated reasonable calibration across common predicted probability ranges, relatively few test observations received probabilities above 50%, limiting confidence in higher-risk predictions.

## Author

**Craig J. Reynolds, MB(ASCP)**

M.S. Data Science — Eastern University
B.S. Molecular Biology — University of Washington
B.A. Chemistry — University of Washington

Background in molecular oncology, molecular diagnostics, and clinical laboratory science.

**Data Source:** [CDC National Health and Nutrition Examination Survey](https://www.cdc.gov/nchs/nhanes/)
