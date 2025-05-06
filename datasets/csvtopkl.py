import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os

data_dest = os.path.join(os.getcwd(), "raw-data") # *CODE MODIFIED HERE*

# Make sure to create sub-folders with these names and put the respective data there
behavioral_fdir = os.path.join(data_dest, "Behavioral")

# Step 1: Read the CSV file
csv_file_path = os.path.join(behavioral_fdir, "OversampledSwipeData.csv")  # Replace with your CSV file path
data = pd.read_csv(csv_file_path,index_col=0)

# Step 2: Split the data into training and testing sets
# Assuming the last column is the target/label column
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target
print(X.shape,y.shape)

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Combine data and labels in a dictionary
train_data = {'data': X_train, 'labels': y_train}
test_data = {'data': X_test, 'labels': y_test}

# Step 4: Save training and testing data as .pkl files
train_pkl_path = os.path.join(behavioral_fdir,"Swipetrain_data.pkl")
test_pkl_path = os.path.join(behavioral_fdir,"Swipetest_data.pkl")

with open(train_pkl_path, "wb") as train_file:
    pickle.dump(train_data, train_file)

with open(test_pkl_path, "wb") as test_file:
    pickle.dump(test_data, test_file)

print(f"Training data saved to {train_pkl_path}")
print(f"Testing data saved to {test_pkl_path}")