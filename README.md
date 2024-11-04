# credit-risk-classification
- Module 20 Challenge
- Steph Abegg

# Credit Risk Analysis Report

## Overview of the Analysis

In this challenge, we used a dataset of historical lending activity from a peer-to-peer lending services company to build a machine-learning model that can identify the creditworthiness of borrowers. We used logistic regression to model creditworthiness. Logistic regression is a statistical model used for binary classification, which predicts the probability that a given input belongs to one of two classes. In this challenge, the predicted variable was the status of a loan, which is a categorial variable (0 means that the loan is healthy, and 1 means that the loan is at a high risk of defaulting). 

The analysis is in [credit_risk_classification.ipynb](Credit_Risk/credit_risk_classification.ipynb)

The steps of the analysis are detailed below:

1. The data is in `lending_data.csv`, which contains `loan_size`, `interest_rate`, `borrower_income`, `debt_to_income`, `num_of_accounts`, `derogatory_marks`, `total_debt`, and `loan_status`. The `loan_status` column contains either 0 or 1, where 0 means that the loan is healthy, and 1 means that the loan is at a high risk of defaulting. The data was stored the data in a DataFrame. A screenshot of the first five rows of the DataFrame is shown below.

   <img src="Credit_Risk/images/dataframe.png" width=900>

2. The loan status label set (i.e. 0 or 1) from the `loan_status` column were stored in the `y` variable and the features DataFrame (all the columns except `loan_status`) were stored in the `X` variable. The "balance" of the labels were checked with `value_counts`. In this dataset, 75036 loans were healthy and 2500 were high-risk.

3. The `train_test_split` module from `sklearn` was used to split the data into training and testing variables: `X_train`, `X_test`, `y_train`, and `y_test`. A `random_state` of 1 was assigned to the function to ensure that the train/test split is consistent, i.e. the same data points are assigned to the training and testing sets across multiple runs of code. A `stratify=y` was used to keep the same proportion of classes in `y` across `y_train` and `y_test`. This is particularly useful for classification tasks where you may have imbalanced classes, such as in this dataset where there are far more healthy loans than high-risk loans.

4. A logistic regression model, called `lr_model`, was created using `LogisticRegression()` from the `sklearn` library. The model was fit with the training data, `X_train` and `y_train`. A `random_state` of 1 was assigned to the model to make the model consistent between runs of the code. 

5. The logistic regression model was then used to make predictions of the testing data labels, `y_pred`, with `predict()` using the testing feature data, `X_test`, as input.

5. A confusion matrix for the model was generated using `confusion_matrix()` from `sklearn`, based on `y_test` and `y_pred`.

6. A classification report for the model was obtained with `classification_report()` from `sklearn`, based on `y_test` and `y_pred`.


## Results

Confusion matrix:

|      | Predicted Positive   | Predicted Negative      |
| ------------- | ------------- | ------------- |
| Actual Positive  | 18673  | 86  |
| Actual Negative  | 32  | 593  |

Classification report:

<img src="Credit_Risk/images/classification_report.png" width=500>

   - Accuracy: 0.994.
   - Precision: for healthy loans the precision is 0.998, for high-risk loans the precision is 0.873.
   - Recall: for healthy loans the recall score is 0.995, for high-risk loans the recall score is 0.949.
   - f1-score: for healthy loans the f1-score is 0.997, for high-risk loans the f1-score is 0.910.

## Summary

The accuracy score represents the proportion of correctly predicted labels (both true positives and true negatives) out of the total number of predictions in the test set. The accuracy score of the logistic regression model is 0.994. So the model is quite accurate. (However, it is important to note the imbalance in the training and testing data where there are far more healthy loans than high-risk loans; the results discussed below suggest that the healthy loans are predicted with higher accuracy than the high-risk loans, so the high accuracy of the model is somewhat a result of there being more healthy loans.)

The precision quantifies how many of the positive predictions made by a logistic regression model are actually correct. In other words, precision tells you the proportion of true positives out of all the predicted positives (both true and false). The logistic regression model had precision scores of 0.998 for healthy loans and 0.873 for high-risk loans. So, the logistic regression model is better for predicting healthy loans than high-risk loans.

Recall, also known as sensitivity or true positive rate, is a metric that quantifies how well a logistic regression model can identify actual positives from the dataset. In other words, it measures the proportion of true positives that were correctly predicted by the model out of all actual positive cases.  The logistic regression model had recall scores of 0.995 for healthy loans and 0.949 for high-risk loans. 

The F1-score is a metric that combines both precision and recall into a single value, providing a balance between the two. It is especially useful when you want to ensure that both false positives and false negatives are minimized, and it's often used when you have an imbalanced dataset (such as the one we have, where there are far more healthy loans than high-risk loans). The logistic regression model had f1-scores of 0.997 for healthy loans and 0.910 for high-risk loans. 

Model performance depends on the problem we are trying to solve. If the company wishes to predict the healthy loans the logistic regression model does quite well and the results of the model are reliably actionable. But if the company wants to predict high-risk loans, then the logistic regression model does not perform quite as well, having lower precision, recall, and F1-scores. Unfortunately, it is ultimately more important to predict high-risk loans. If the company wishes to predict high-risk loans, then the model can be used for a general prediction with 87-94 percent reliability, but other factors should be considered in the decision-making process.

