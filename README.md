# Customer Churn Prediction

## Overview
This project focuses on predicting customer churn using machine learning techniques. The dataset used is the **Telco Customer Churn Dataset**, which contains various attributes related to customer demographics, services, and account details. The analysis includes data preprocessing, visualization, feature engineering, model training, and evaluation.

## Dataset
- The dataset consists of customer information such as tenure, monthly charges, total charges, contract type, and internet service type.
- The target variable is **Churn**, which indicates whether a customer has left the company or not.
- The dataset is preprocessed to handle missing values, categorical encoding, and feature scaling.

## Project Workflow
### **Task 1: Data Preparation**
1. **Loading Data:** Read the CSV file into a Pandas DataFrame.
2. **Cleaning Data:**
   - Removed the `customerID` column as it's not relevant to churn prediction.
   - Converted `TotalCharges` from object type to numeric.
   - Checked and removed null values.
3. **Data Visualization:**
   - Histogram and boxplots for numerical features.
   - Count plots for categorical features.
   - Correlation heatmap.
4. **Feature Engineering:**
   - One-hot encoding for categorical variables.
   - Label encoding for ordinal categories.
   - Standardization of numerical features using `StandardScaler`.

### **Task 2: Data Splitting**
- Splitting the dataset into **training (80%)** and **testing (20%)** sets.
- Handling class imbalance using **undersampling** and **SMOTE (Synthetic Minority Over-sampling Technique)**.

### **Task 3: Model Selection & Training**
Several machine learning models were implemented and evaluated:

#### **1. Random Forest Classifier**
- Justification: Handles mixed data types well, prevents overfitting, provides feature importance analysis.
- Hyperparameter tuning with different numbers of estimators.
- Evaluation:
  - Confusion matrix and classification report.
  - Cross-validation to check model stability.
  - Individual decision tree analysis.

#### **2. Decision Tree Classifier (with Pruning)**
- Cost Complexity Pruning (`ccp_alpha`) to prevent overfitting.
- Visualization of pruning effect on accuracy.

#### **3. Gradient Boosting Classifier**
- Grid Search for hyperparameter tuning (`learning_rate` and `max_depth`).
- Training accuracy vs. test accuracy comparison.
- Evaluation using cross-validation and confusion matrix.

#### **4. Logistic Regression**
- Effect of `max_iter` on accuracy.
- Hyperparameter tuning for regularization (`C`).

### **Task 4: Model Evaluation**
Each model was evaluated using:
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score, Accuracy)**
- **Cross-Validation Performance**
- **Feature Importance Analysis**

## Results & Insights
- **Random Forest** performed the best in terms of accuracy and interpretability.
- **Gradient Boosting** showed good results with fine-tuned hyperparameters.
- **Logistic Regression** was less effective due to the non-linearity in the dataset.
- **Decision Tree with pruning** helped prevent overfitting and provided explainability.

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/diyanayakE15/Customer-Churn-Predictions
   cd customer-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to execute the model training and evaluation.

## Future Improvements
- Exploring deep learning models such as Neural Networks.
- Implementing additional feature selection techniques.
- Deploying the best model as a web service.

## Author
**Diya Uday Nayak**  
[LinkedIn](https://www.linkedin.com/in/diyan-6151/) | [GitHub](https://github.com/diyanayakE15/)

