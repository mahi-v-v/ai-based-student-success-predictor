import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

# --- 1. LOAD AND PREPARE DATA ---
print("Loading data...")
df = pd.read_csv('Students.csv')

# Define the NEW passing score and create the target variable
PASSING_SCORE = 70
df['Pass_Fail'] = np.where(df['Exam_Score'] >= PASSING_SCORE, 1, 0)

# Check the balance of the new data
print("Data balance with passing score of 70:")
print(df['Pass_Fail'].value_counts())


# Select features (X) and target (y)
features = df.drop(columns=['Student_ID', 'Name', 'Exam_Score', 'Pass_Fail'])
target = df['Pass_Fail']

# Convert categorical variables into a numerical format
features_encoded = pd.get_dummies(features)

print("\nData has been prepared for training.")

# --- 2. TRAIN THE MODEL ---
# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

print("Training the model...")
model.fit(features_encoded, target)
print("Model training complete.")

# --- 3. SAVE THE MODEL ---
# Define the filename for your new model
filename = 'trained_model.sav'

# Save the trained model to the file using pickle
pickle.dump(model, open(filename, 'wb'))

print(f"\nSuccess! New model has been saved as '{filename}'")
