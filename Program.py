# --- Import necessary libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# --- Load the dataset ---
dataset_path = 'C:/Users/USer/Documents/AOI/customer_churn_dataset-training-master.csv'  # Update path if necessary
churn_data = pd.read_csv(dataset_path)

# --- Check and clean column names ---
churn_data.columns = churn_data.columns.str.strip()  # Remove leading/trailing spaces
print("Columns in the dataset:", churn_data.columns)

# --- Handle 'TotalCharges' column ---
# Check if 'TotalCharges' exists, if not print a message and skip processing
if 'TotalCharges' in churn_data.columns:
    churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
else:
    print("'TotalCharges' column is not found in the dataset. Skipping this column.")

# --- Separate numeric and categorical columns ---
numeric_columns = churn_data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = churn_data.select_dtypes(include=['object']).columns

# --- Handle missing values ---
# Fill missing values for numeric columns with the mean of the column
churn_data[numeric_columns] = churn_data[numeric_columns].fillna(churn_data[numeric_columns].mean())

# Fill missing values for categorical columns with the mode (most frequent value)
churn_data[categorical_columns] = churn_data[categorical_columns].fillna(churn_data[categorical_columns].mode().iloc[0])

# --- Encode categorical variables ---
label_encoder = LabelEncoder()
churn_data['Gender'] = label_encoder.fit_transform(churn_data['Gender'])
churn_data['Contract Length'] = label_encoder.fit_transform(churn_data['Contract Length'])
churn_data['Subscription Type'] = label_encoder.fit_transform(churn_data['Subscription Type'])
churn_data['Churn'] = label_encoder.fit_transform(churn_data['Churn'])

# --- Define Features and Target ---
X = churn_data.drop('Churn', axis=1)  # Features
y = churn_data['Churn']  # Target variable

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train the Model with Class Weights ---
# Using 'balanced' class weight to address class imbalance
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Feature Importance ---
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Features for Churn Prediction")
plt.show()

# --- Optional: Evaluate Model on the Test Set ---
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
