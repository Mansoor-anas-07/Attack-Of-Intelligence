
# Retail Customer Churn Analysis


## Contributors

1. Krishna Shah [KU2407U703]
2. Harsh Gupta [KU2407U757]
3. Yash Jha [KU2407U756]
4. Mansoor Anas [KU2407U696]

##

## Description
A brief description of what this project does and who it's for
Identify customers likely to churn and visualize factors influencing churn.

## Table of content
1.	[Introduction](#Introduction)
2.	[Objective](#Objetive)
3. [Tools and Libraries](#tools-and-libraries)
4. [Data Sources](#data-sources)
5. [Execution Steps](#execution-steps)
6. [Challenges Faced](#challenges-faced)
7. [Lessons Learned](#lessons-learned)
## Introduction

Customer churn refers to the loss of customers who stop using a company's services or products. Identifying customers likely to churn is essential for businesses to take proactive measures and improve retention. This project focuses on analyzing customer data to predict which customers are at risk of leaving. By using machine learning techniques and data analysis, we identify key factors influencing churn, such as customer demographics, usage patterns, and service interactions. Visualizing these factors helps businesses understand why customers leave and enables them to implement strategies to reduce churn and enhance customer loyalty.	
## Objective 

•  Enhance customer retention strategies by identifying high-risk customers early.
•  Understand the drivers of churn to make informed decisions on improving products, services, and customer experiences.
•  Reduce churn rates through targeted interventions, such as personalized offers or customer support enhancements.

## Tools and libraries

**Python 3.x**
- **NumPy** - For numerical computing and array manipulation.
- **Pandas** - For data manipulation and analysis.
- **Matplotlib** - For data visualization and plotting.
- **Seaborn** - For enhanced data visualization.
- **Scikit-learn** - For machine learning algorithms and model evaluation.
- **Imbalanced-learn** - For handling class imbalance using techniques like SMOTE.


## Data source

1.Customer Demographics:

Age
Gender
Tenure (how long the customer has been with the company)
Income/Salary
Account Information:

2.Subscription plan or service type
Contract type (e.g., monthly, yearly)
Payment method (e.g., credit card, bank transfer)
Account status (active, inactive, suspended)
Customer Behavior:

3.Usage frequency (e.g., number of logins, transactions)
Interaction with customer service (e.g., number of complaints, call frequency)

4.Product or service features used
Customer engagement (e.g., participation in loyalty programs)
Churn Indicator:

Whether the customer has churned (binary target variable: 1 for churned, 0 for active)

**Retail Customer Churn Data**: The dataset used for this project is a publicly available retail customer churn dataset that contains customer demographics, service usage, and churn status.
- **Source**: [Kaggle - Customer Churn Prediction](https://www.kaggle.com/)

## Execution Steps

 ## Step 1:
  Clone the Repository
Clone the project repository from GitHub:
git clone https://github.com/[Group_Name]/Retail_Customer_Churn_Analysis.git

## Step 2: 
Install Required Libraries
pip install -r requirements.txt

## Step 3:
 Load the Data
Load the dataset using Pandas:
import pandas as pd
data = pd.read_csv('data/customer_churn.csv')

## Step 4:
 Preprocess the Data
Clean the dataset by handling missing values, encoding categorical features, and scaling numerical features.
Split the data into training and testing sets.

## Step 5:
 Train Machine Learning Models
Train models like Logistic Regression, Random Forest, and XGBoost.
Use techniques like cross-validation and hyperparameter tuning to optimize model performance.

## Step 6: 
Evaluate the Model
Evaluate the model performance using metrics like accuracy, precision, recall, and F1-score.
Implement a confusion matrix and ROC-AUC curve for model evaluation.

## Step 7: 
Visualize Factors Influencing Churn
Generate visualizations such as:
Feature importance plots to understand the factors influencing churn.
Correlation heatmaps to see relationships between variables.
Bar charts showing churn rates across different demographic groups.

## Step 8: 
Interpret Results
Interpret the results to identify:

Key factors contributing to churn (e.g., low usage, low customer satisfaction).
Customers at high risk of churning and potential actions to retain them.


## Summary 
The model successfully predicted customer churn with an accuracy of 85%. The key factors influencing churn were:

Customer service interaction: Customers with frequent service complaints were more likely to churn.

Usage frequency: Low engagement with the service led to higher churn rates.

Contract type: Monthly contracts had higher churn rates than yearly contracts.

The visualizations showed that customers in certain age groups and income brackets were more likely to churn, providing actionable insights for marketing strategies.
## Challenges Faced

Imbalanced Dataset:

1. The dataset was highly imbalanced, with fewer churned customers than active ones.
Solution: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.
Missing Data:

2. Some columns had missing values.
Solution: Imputed missing values with mean/median imputation for numerical columns and mode imputation for categorical columns.
Overfitting:

3. Some models overfitted the training data.
Solution: Used regularization techniques and cross-validation to reduce overfitting and improve generalization.
Feature Selection:

4. Identifying the most important features influencing churn.


## Lessons Learned


~Data Preprocessing: The importance of cleaning and transforming 
 data (handling missing values, encoding categorical features)
 for effective analysis.

~Feature Engineering: Identifying key features that influence   
 churn and improving model performance through feature selection.
 Modeling: Building and evaluating multiple models (Logistic    
 Regression, Random Forest) to predict churn and choosing the 
 best one.

~Data Visualization: Using Matplotlib and Seaborn to visualize 
 trends and factors contributing to churn, aiding in better 
 decision-making.

~Imbalanced Data: Addressing class imbalance using techniques  
 like SMOTE to ensure accurate predictions.
