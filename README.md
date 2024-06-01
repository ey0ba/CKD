Report on Predictive Modeling for Chronic Kidney Disease (CKD)

General Overview

In this report, we describe the steps taken to build a predictive model for Chronic Kidney Disease (CKD) using a dataset with 200 samples and 60 features. The primary goal was to preprocess the data, identify important features, and build robust models using Random Forest. Additionally, cross-validation was performed to ensure the model's robustness.

Methods

Data Preprocessing
Loading the Dataset: The dataset was loaded into a pandas DataFrame for processing.

Exploratory Data Analysis:
•	Checked the shape and columns of the dataset.
•	Identified unique values and the number of unique values for each column to understand the data better.
•	Detected and counted missing values in columns.
•	Identified and handled a row named 'discrete' which was present in all columns.
•	Handling Missing Values: Dropped rows with missing values.
•	
Column Classification:
Classified columns as binary, categorical, or numerical based on their unique values and data types.
Encoding:
•	Used LabelEncoder for binary columns.
•	Applied one-hot encoding for categorical columns.
Correlation Analysis:
•	Calculated the correlation matrix.
•	Visualized the correlation matrix using a heatmap.
•	Identified features with high correlation with the target variable 'class'.
•	Dropped features with correlation less than a specified threshold.
Feature Selection
•	Identified Important Features: Selected features that had a significant correlation with the target variable.
•	Dropped the 'affected' Column: Removed the 'affected' column as it had a correlation value of 1, indicating it is perfectly correlated with the target variable.
Model Building

Defining Features and Target: Defined the feature set (X) and target variable (y).
Data Splitting: Split the data into training and test sets using an 80-20 split.
Model Training and Evaluation
Random Forest Classifier:
•	Trained the model on the training set.
•	Predicted on the test set and calculated the accuracy.
•	Random Forest Accuracy: 95%
•	Feature Importance: Plotted the feature importances obtained from the Random Forest model.
•	
Cross-Validation
•	Performed cross-validation to ensure the robustness of the model:
•	Cross-validation scores: [0.95, 0.95, 0.925, 0.975, 0.95]
•	Mean cross-validation score: 95%

Code and Results

Data Preprocessing and EDA

import pandas as pd

df = pd.read_csv('D:\\beka\\pro1\\datascience\\PYTHON FOR DATA SCIENCE\\datasets\\ckd.csv')

df.shape
df.describe
df.columns

# Exploring unique values and their counts
for column in df.columns:
    unique_values = df[column].unique()
    num_unique = len(unique_values)
    print(f'Column: {column}')
    print(f'Data Type: {df[column].dtype}')
    print(f'Number of Unique Values: {num_unique}')

# Checking for missing values
missing_values_count = df['bp (Diastolic)'].isnull().sum()
print(f"Missing values in 'bp (Diastolic)': {missing_values_count}")

# Handling 'discrete' values
columns_to_check = ['bp (Diastolic)', 'bp limit', 'sg', 'al', 'class', 'rbc', 'su', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc', 'wbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'grf', 'stage', 'affected', 'age']

for column in columns_to_check:
    df[column] = df[column].replace('discrete', pd.NA)

df.dropna(inplace=True)

# Classifying columns
def classify_column_type(column):
    unique_values = df[column].unique()
    num_unique = len(unique_values)
    
    if df[column].dtype == 'object':
        if num_unique == 2:
            return 'Binary'
        else:
            return 'Categorical'
    elif df[column].dtype in ['int64', 'float64']:
        if num_unique == 2:
            return 'Binary'
        else:
            return 'Numerical'
    else:
        return 'Unknown'

# Classify each column
column_types = {}
for column in df.columns:
    column_types[column] = classify_column_type(column)

# Display the column types
for column, col_type in column_types.items():
    print(f"Column: {column}, Type: {col_type}")



Encoding


from sklearn.preprocessing import LabelEncoder
    
# Define binary columns
binary_columns = ['bp (Diastolic)', 'class', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'affected']

# Initialize LabelEncoder
label_encoders = {}

# Apply LabelEncoder to each binary column
for column in binary_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define categorical columns
categorical_columns = ['bp limit', 'sg', 'al', 'su', 'bgr', 'bu', 'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc', 'wbcc', 'grf', 'stage', 'age']

# Apply one-hot encoding to the categorical columns with prefix and drop the first category
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, prefix=categorical_columns)

# Display the encoded DataFrame
print(df_encoded.head())

df = df_encoded




Correlation Analysis and Feature Selection


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap for correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Extract the correlation values for the target variable 'class'
corr_with_target = corr_matrix['class'].abs().sort_values(ascending=False)

# Display the column names with their correlation values
print(corr_with_target)

# Set the correlation threshold
correlation_threshold = 0.4

# Calculate the correlation matrix and get the absolute correlations with the target variable 'class'
correlations = df.corr()['class'].abs()

# Identify features with correlation less than the threshold
low_correlation_features = correlations[correlations < correlation_threshold].index

# Drop the low correlation features
df = df.drop(columns=low_correlation_features)

# Display the remaining features
print("Remaining features after dropping low correlation ones:")
print(df.columns)

# Remove the column named 'affected' because it has a correlation value of 1
df = df.drop(columns=['affected'])





Model Building



# Define the features (X) and the target (y)
X = df.drop('class', axis=1)
y = df['class']

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



Model Training and Evaluation


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Get feature importances from the random forest model
feature_importances = rf_model.feature_importances_

# Create a DataFrame for visualization
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importance')
plt.show()








Cross-Validation


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores and the mean score
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")




Results
•	Random Forest Accuracy: 95%
•	Cross-validation scores: [0.95, 0.95, 0.925, 0.975, 0.95]
•	Mean cross-validation score: 95%
•	Top Features: 'al_< 0', 'dm', 'htn', 'sg_≥ 1.023', 'bp limit_2', etc.











Feature Importance
The bar plot of feature importance shows the most significant features in predicting CKD, with 'al_< 0' being the most important.
 


Conclusion
The predictive model for CKD using Random Forest has shown high accuracy (95%) and robustness through cross-validation. The preprocessing steps, including handling missing values, encoding categorical variables, and feature selection based on correlation, were crucial in building an effective model. The most important features identified can provide insights into the key indicators of CKD, aiding in early detection and better management of the disease.


Abbreviations and Description
al = (albumin):  This represents the level of albumin in the urine. Albumin is a type of protein, and its presence in urine (albuminuria) can be an indicator of kidney damage.
dm = (Diabetes Mellitus): This indicates whether the patient has diabetes mellitus, a major risk factor for developing CKD.
htn = (Hypertension): This indicates whether the patient has hypertension (high blood pressure), a common condition that can cause or exacerbate kidney damage.
sg = (Specific Gravity): This measures the concentration of solutes in the urine. It is used to assess the kidney's ability to concentrate urine.

bp limit = (Blood Pressure Limit): This likely refers to a categorical representation of blood pressure limits. It could indicate different ranges or stages of blood pressure readings, which are crucial in evaluating the impact of blood pressure on kidney function.

