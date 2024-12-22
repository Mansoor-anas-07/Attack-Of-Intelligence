import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Datasets ---
# Update file paths if necessary
training_data = pd.read_csv('C:/Users/USer/Documents/AOI/customer_churn_dataset-training-master.csv')
testing_data = pd.read_csv('C:/Users/USer/Documents/AOI/customer_churn_dataset-testing-master.csv')

# --- Preprocessing ---
# Combine training and testing datasets for preprocessing consistency
training_data['is_train'] = 1
testing_data['is_train'] = 0
combined_data = pd.concat([training_data, testing_data], ignore_index=True)

# Encode categorical variables
combined_data = pd.get_dummies(
    combined_data, 
    columns=['Gender', 'Subscription Type', 'Contract Length'], 
    drop_first=True
)

# Check for missing values and handle them
print("Missing values summary:\n", combined_data.isnull().sum())

# Drop rows with missing target values
combined_data.dropna(subset=['Churn'], inplace=True)

# Fill missing numerical features with median (if any)
combined_data.fillna(combined_data.median(), inplace=True)

# Split back into training and testing
training_data = combined_data[combined_data['is_train'] == 1].drop('is_train', axis=1)
testing_data = combined_data[combined_data['is_train'] == 0].drop('is_train', axis=1)

# Separate features and target
X_train = training_data.drop(['CustomerID', 'Churn'], axis=1)
y_train = training_data['Churn']

X_test = testing_data.drop(['CustomerID', 'Churn'], axis=1)
y_test = testing_data['Churn']

# --- Train Model ---
model = RandomForestClassifier(random_state=42)
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
